import pandas as pd
import numpy as np
import os
from src.data.ghz_factors import GHZFactorBuilder
from src.utils.data_utils import clean_infs

class ORCADataLoader:
    def __init__(self, data_root: str, cache_dir: str = None):
        self.builder = GHZFactorBuilder(data_root)

        # Default cache dir is sibling to data_root
        if cache_dir is None:
            parent = os.path.dirname(data_root.rstrip('/'))
            self.cache_dir = os.path.join(parent, "orca_feature_cache")
        else:
            self.cache_dir = cache_dir

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Created cache directory: {self.cache_dir}")
        else:
            print(f"Using cache directory: {self.cache_dir}")

    def _get_cache_path(self, year):
        return os.path.join(self.cache_dir, f"orca_features_{year}.parquet")

    def load_data(self, start_year: int, end_year: int):
        """
        Smart Load: Check cache for each year.
        Process only missing years (with 1 year buffer for lag features).
        """
        years = range(start_year, end_year + 1)
        dfs = []
        missing_years = []

        # 1. Check Cache
        for y in years:
            path = self._get_cache_path(y)
            if os.path.exists(path):
                # print(f"Found cache for {y}")
                dfs.append((y, pd.read_parquet(path)))
            else:
                missing_years.append(y)

        # 2. Process Missing
        if missing_years:
            min_miss = min(missing_years)
            max_miss = max(missing_years)
            print(f"Cache miss for years: {missing_years}. Processing {min_miss}-{max_miss}...")

            # Load Raw with Buffer (Need previous year for momentum)
            # If rolling window is 24 months, strictly we need 2 years buffer?
            # data_loader code uses 24 months.
            # safe assumption: 2 years buffer if possible, or just 2000 start.
            buffer_start = max(1990, min_miss - 2)

            # We need to re-initialize builder or clear it?
            # GHZFactorBuilder handles load_data by Accumulating?
            # No, it usually re-reads.

            print(f"Loading Raw Data ({buffer_start}-{max_miss}) for Feature Engineering...")
            self.builder.load_data(buffer_start, max_miss)

            # Compute ALL features for this chunk
            processed_chunk = self.build_orca_features() # This returns features for buffer_start to max_miss

            if not processed_chunk.empty:
                processed_chunk['year'] = processed_chunk['date'].dt.year

                # Save each missing year to cache and append to dfs
                for y in missing_years:
                    yearly_data = processed_chunk[processed_chunk['year'] == y].copy()
                    if not yearly_data.empty:
                        # Drop temp year col
                        yearly_data = yearly_data.drop(columns=['year'])

                        save_path = self._get_cache_path(y)
                        yearly_data.to_parquet(save_path)
                        print(f"Saved cache: {save_path}")

                        dfs.append((y, yearly_data))
                    else:
                        print(f"Warning: No data generated for missing year {y}")

        # 3. Concatenate and Return (sorted by year for consistency)
        dfs.sort(key=lambda x: x[0])
        self.cached_features = pd.concat([d[1] for d in dfs], ignore_index=True)

        # Ensure sorted
        if 'date' in self.cached_features.columns:
            self.cached_features = self.cached_features.sort_values(['date', 'permno'])

    def get_features(self):
        return self.cached_features

    def build_orca_features(self) -> pd.DataFrame:
        """
        Build the 36 features specified in Kim et al. (2025):
        - 24 Momentum Features (mom_1 to mom_24)
        - 12 Fundamental Features (Quarterly)

        Returns:
            pd.DataFrame: Aligned features with index (permno, date) (monthly frequency)
        """
        # 1. Process Quarterly Fundamentals using precise date logic from GHZ
        # We reuse process_quarterly to get 'valid_from' and aligned data
        print("Processing Quarterly Data...")
        fundq = self.builder.process_quarterly()

        # Merge raw features back to valid dates
        q_keys = ['gvkey', 'datadate']
        # Use copy to avoid SettingWithCopy warnings and ensure we have raw data
        fundq_raw = self.builder.fundq.copy()

        # Fix Merge Error: Ensure datadate is datetime
        fundq_raw['datadate'] = pd.to_datetime(fundq_raw['datadate'])
        if 'datadate' in fundq.columns:
            fundq['datadate'] = pd.to_datetime(fundq['datadate'])

        # Derive Missing Columns (Proxies if not in WRDS pull)
        # Required: ATQ, LTQ, DLCQ, DLTTQ, SEQQ, CHEQ, SALEQ, NIQ, OIADPQ, PIQ, DPQ, EPSPXQ

        # 1. NIQ (Net Income) -> IBQ (Income Before components)
        if 'niq' not in fundq_raw.columns and 'ibq' in fundq_raw.columns:
            fundq_raw['niq'] = fundq_raw['ibq']

        # 2. DLTTQ (Long Term Debt) -> LTQ - LCTQ (Total - Current)
        if 'dlttq' not in fundq_raw.columns and 'ltq' in fundq_raw.columns and 'lctq' in fundq_raw.columns:
             fundq_raw['dlttq'] = fundq_raw['ltq'] - fundq_raw['lctq']

        # 3. OIADPQ (Op Income) -> SALEQ - COGSQ - XSGAQ
        if 'oiadpq' not in fundq_raw.columns:
            # Need saleq, cogsq, xsgaq
            if all(c in fundq_raw.columns for c in ['saleq', 'cogsq', 'xsgaq']):
                fundq_raw['oiadpq'] = fundq_raw['saleq'] - fundq_raw['cogsq'].fillna(0) - fundq_raw['xsgaq'].fillna(0)
            else:
                fundq_raw['oiadpq'] = 0 # Fallback

        # 4. PIQ (Pretax) -> IBQ + TXTQ
        if 'piq' not in fundq_raw.columns and 'ibq' in fundq_raw.columns and 'txtq' in fundq_raw.columns:
            fundq_raw['piq'] = fundq_raw['ibq'] + fundq_raw['txtq'].fillna(0)

        # 5. EPSPXQ -> IBQ / CSHOQ
        if 'epspxq' not in fundq_raw.columns and 'ibq' in fundq_raw.columns and 'cshoq' in fundq_raw.columns:
            fundq_raw['epspxq'] = fundq_raw['ibq'] / fundq_raw['cshoq'].replace(0, np.nan)

        # 6. DPQ -> 0 (Missing)
        if 'dpq' not in fundq_raw.columns:
            fundq_raw['dpq'] = 0.0

        funda_cols = [
            'atq', 'ltq', 'dlcq', 'dlttq', 'seqq', 'cheq',
            'saleq', 'niq', 'oiadpq', 'piq', 'dpq', 'epspxq'
        ]

        # Ensure all exist now (or created as nan/0)
        for c in funda_cols:
            if c not in fundq_raw.columns:
                fundq_raw[c] = 0.0

        # Merge valid_from from processed fundq
        if 'valid_from' in fundq.columns:
            # fundq has valid_from, gvkey, datadate, cusip
            fundq_aligned = fundq[['gvkey', 'datadate', 'valid_from', 'cusip']].merge(
                fundq_raw[q_keys + funda_cols],
                on=q_keys,
                how='left'
            )
        else:
             # Fallback if process_quarterly failed or empty
            print("Warning: process_quarterly returned empty or invalid data.")
            return pd.DataFrame()

        # 2. Process Monthly Price/Momentum
        print("Processing Monthly Price Data...")
        # GHZ process_crsp helps cleaning but we need custom momentum
        msf = self.builder.process_crsp()

        if msf.empty:
            return pd.DataFrame()

        # Re-calculate specific ORCA momentum:
        # mom_1 = r_{t-1}
        # mom_i = prod(1+r_{t-k}) - 1 for k=1..i

        # Sort by permno, date
        msf = msf.sort_values(['permno', 'date'])

        # We need raw returns. process_crsp output 'mom1m' which is r_{t-1}.
        # But for full range 1..24, let's recompute from 'ret'.
        # Note: process_crsp outputs 'ret' column.

        g = msf.groupby('permno')['ret']

        # mom_1 is just shift(1)
        msf['mom_1'] = g.shift(1)

        # For mom_2 to mom_24:
        # It's cumulative return over past i months.
        # mom_i at time t is return from t-i to t-1.

        # Precompute log returns for summation: log(1+r)
        # Handling NaNs/Infs done in safe_log inside builder but let's be sure
        log_ret = np.log1p(msf['ret'].fillna(0)) # approximation for speed, ensure no -1

        # We need rolling sum of log returns, then exp.
        # Window size i, shift 1 (to not include current month t)

        for i in range(2, 25): # 2 to 24
            # Rolling window i, shifted by 1
            # rolling_sum of log_ret
            # We use transform to keep alignment
            # Note: shift(1) is applied first to exclude current month.
            roll_log = msf.groupby('permno')['ret'].transform(
                lambda x: np.log1p(x).shift(1).rolling(window=i, min_periods=i).sum()
            )
            msf[f'mom_{i}'] = np.exp(roll_log) - 1

        # 3. Merge Monthly and Quarterly
        # We need to link PERMNO (MSF) to GVKEY (Fundq)
        # Use GHZ merge logic?
        # GHZ uses merge_asof on date with backwards direction.

        # We need a Link Table. GHZFactorBuilder has 'stocknames'.
        # If stocknames empty, we can't link well (fallback to CUSIP if available in process_crsp?)
        # process_crsp does NOT return CUSIP currently. process_annual/quarterly return CUSIP.
        # We need to bridge via 'stocknames' (permno, ncusip/cusip, date range).

        # Let's perform the merge.
        print("Merging Monthly and Quarterly Data...")

        # Link via stocknames if available
        if self.builder.stocknames is not None:
            # CCM Linking
            # stocknames: permno, namedt, nameendt, ncusip, ticker...
            link = self.builder.stocknames.copy()

            # Robustness Checks for Stocknames
            if 'namedt' not in link.columns:
                link['namedt'] = pd.Timestamp('1900-01-01')
            else:
                link['namedt'] = pd.to_datetime(link['namedt'])

            if 'nameendt' not in link.columns:
                link['nameendt'] = pd.Timestamp('2100-01-01')
            else:
                link['nameendt'] = pd.to_datetime(link['nameendt']).fillna(pd.Timestamp('2100-01-01'))

            # Use gvkey-permno link if available?
            # WRDS stocknames usually has PERMNO and NCUSIP. Compustat has CUSIP.
            # CUSIPs change.
            # Simplified approach: Merge MSF with Link table to get NCUSIP, then merge with Compustat CUSIP.
            # ORCA/GHZ usually assumes CCM keys (gvkey-permno link table "ccmxpf_lnkhist").
            # The current pull_wrds pulls `crsp.stocknames`. This gives NCUSIP.
            # Compustat `fundq` has CUSIP.
            # NCUSIP (CRSP) is historical CUSIP. Compustat CUSIP is header? Or historical?
            # It's usually Header. This is a weak link but let's try.
            # Better: If the user has `ccmxpf_lnkhist` data? Not pulled.

            # Warning: Linking might be weak without ccmxpf_lnkhist.
            # We will use NCUSIP (first 6 digits) match.
            pass

        # For now, let's assume we proceed with MSF data as the base (since we trade stocks)
        # and attach fundamentals via `merge_asof` on Date, if we can map IDs.

        # If we can't link, we return just momentum? No, paper requires fundamentals.
        # The `merge_factors.py` assumes `ghz_factors` (with PERMNO) and `daily` (with PERMNO).
        # Wait, `ghz_factors.py` returns data with `permno`?
        # Check `process_annual`: returns `cusip`.
        # Check `process_crsp`: returns `permno`.
        # `merge_and_align` in `ghz_factors.py` handles the linking!
        # It uses `stocknames` to link `annual` (cusip) to `market` (permno).

        # SO: We should use `self.builder.merge_and_align`.
        # But `merge_and_align` takes `annual` and `quarterly` and computes GHZ factors.
        # We need to supply our CUSTOM annual/quarterly frames to a similar logic.

        # Refactor strategy:
        # 1. Prepare raw feature frames (fundq_aligned, msf with mom).
        # 2. Use a modified merge logic based on `merge_and_align` but for our columns.

        # Align column names for merge
        fundq_aligned = fundq_aligned.rename(columns={'valid_from': 'date'}) # It becomes the effective date

        # We reuse the logic in builder.merge_and_align but passing our frames
        # However `merge_and_align` is harders to reuse directly as it calculates factors inside.
        # We will copy the linking logic.

        merged = self._merge_custom(msf, fundq_aligned)

        return merged

    def _merge_custom(self, msf, fundq):
        # Implement simplified merge_and_align logic
        # 1. Link fundq (GVKEY/CUSIP) to PERMNO using stocknames
        if self.builder.stocknames is None:
            print("Stocknames missing. Cannot link.")
            return msf # Return only price data?

        link = self.builder.stocknames.copy()

        # Robustness Checks (Duplicate of above but needed for local context if called independently)
        if 'namedt' not in link.columns:
            link['namedt'] = pd.Timestamp('1900-01-01')
        else:
            link['namedt'] = pd.to_datetime(link['namedt'])

        if 'nameendt' not in link.columns:
            link['nameendt'] = pd.Timestamp('2100-01-01')
        else:
            link['nameendt'] = pd.to_datetime(link['nameendt']).fillna(pd.Timestamp('2100-01-01'))

        # Prepare 6-digit CUSIPs
        link['ncusip6'] = link['ncusip'].astype(str).str[:6]
        fundq['cusip6'] = fundq['cusip'].astype(str).str[:6]

        # Link fundq to permno
        # Since fundq has 'date' (valid_from), we merge on CUSIP and checking date ranges
        # This is expensive (Cartesian header).
        # Optimized:
        # MERGE fundq with Link on CUSIP6
        # Filter where date between namedt and nameendt

        fundq_perm = pd.merge(fundq, link[['permno', 'ncusip6', 'namedt', 'nameendt']],
                              left_on='cusip6', right_on='ncusip6', how='inner')

        # Filter date range
        mask = (fundq_perm['date'] >= fundq_perm['namedt']) & (fundq_perm['date'] <= fundq_perm['nameendt'])
        fundq_perm = fundq_perm[mask]

        # Now we have fundq with PERMNO and DATE (valid_from)
        # We can merge_asof with MSF

        # MSF is the backbone (monthly)
        msf = msf.sort_values(['date'])
        fundq_perm = fundq_perm.sort_values(['date'])

        # Drop overlapping cols
        # Drop overlapping cols and metadata
        meta_cols = ['namedt', 'nameendt', 'cusip6', 'ncusip6', 'cusip', 'ncusip']
        cols_to_use = [c for c in fundq_perm.columns
                       if c not in msf.columns
                       and c != 'permno'
                       and c not in meta_cols]

        # CLEAN KEYS for merge_asof
        # msf: date, permno must be non-null
        msf = msf.dropna(subset=['date', 'permno'])
        # fundq_perm: date, permno must be non-null
        fundq_perm = fundq_perm.dropna(subset=['date', 'permno'])

        # Ensure correct types
        msf['permno'] = msf['permno'].astype(int)
        fundq_perm['permno'] = fundq_perm['permno'].astype(int)

        # merge_asof
        merged = pd.merge_asof(
            msf.sort_values('date'),
            fundq_perm[['permno', 'date'] + cols_to_use].sort_values('date'),
            on='date',
            by='permno',
            direction='backward',
            tolerance=pd.Timedelta(days=90) # Quarterly data
        )

        return merged
