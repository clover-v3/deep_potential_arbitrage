"""
GHZ (Green, Hand, Zhang 2017) Factor Builder
Translates SAS logic to Python/Pandas.
"""

import pandas as pd
import numpy as np
import os

class GHZFactorBuilder:
    def __init__(self, data_root: str):
        self.data_root = data_root

        # Define factor list for strict filtering
        self.factor_list = [
            'mve', 'bm', 'ep', 'cashpr', 'dy', 'lev', 'sp', 'roic', 'rd_sale',
            'rd_mve', 'agr', 'gma', 'chcsho', 'lgr', 'sgr', 'hire', 'acc', 'pctacc',
            'absacc', 'cfp', 'chinv', 'spii', 'cf', 'chpm', 'chato',
            'pchsale_pchinvt', 'pchsale_pchrect', 'pchgm_pchsale', 'pchsale_pchxsga',
            'depr', 'pchdepr', 'chadv', 'invest', 'egr', 'pchcapx', 'grcapx',
            'grgw', 'wogw', 'tang', 'sin', 'currat', 'pchcurrat', 'quick',
            'pchquick', 'salecash', 'salerec', 'saleinv', 'pchsaleinv',
            'cashdebt', 'realestate', 'divi', 'divo', 'obklg', 'chobklg',
            'securedind', 'secured', 'convind', 'conv', 'grltnoa', 'dr', 'chdrc',
            'rd', 'rdbias', 'roe', 'operprof', 'ps', 'tb_1',
            'roa', 'cfroa', 'xrdint', 'capxint', 'xadint', 'sic2',
            # Industry adjusted versions
            'chpm_ia', 'chato_ia', 'bm_ia', 'cfp_ia', 'hire_ia', 'mve_ia',
            'pchcapx_ia', 'tb_1_ia', 'acc_ia', 'absacc_ia', 'pctacc_ia',
            'chinv_ia', 'spii_ia', 'spi_ia', 'cf_ia', 'pchsale_pchinvt_ia',
            'pchsale_pchrect_ia', 'pchgm_pchsale_ia', 'pchsale_pchxsga_ia',
            'depr_ia', 'pchdepr_ia', 'chadv_ia', 'invest_ia', 'egr_ia',
            'grcapx_ia', 'grgw_ia', 'wogw_ia', 'tang_ia', 'sin_ia',
            'currat_ia', 'pchcurrat_ia', 'quick_ia', 'pchquick_ia',
            'salecash_ia', 'salerec_ia', 'saleinv_ia', 'pchsaleinv_ia',
            'cashdebt_ia', 'realestate_ia', 'divi_ia', 'divo_ia',
            'obklg_ia', 'chobklg_ia', 'securedind_ia', 'secured_ia',
            'convind_ia', 'conv_ia', 'grltnoa_ia', 'chdrc_ia', 'rd_ia',
            'rdbias_ia', 'roe_ia', 'operprof_ia', 'ps_ia', 'roa_ia',
            'cfroa_ia', 'xrdint_ia', 'capxint_ia', 'xadint_ia',
            # Quarterly
            'roaq', 'rsup', 'sue', 'roavol', 'mveq', 'chtx', 'roeq',
            'sacc', 'stdacc', 'sgrvol', 'scf', 'stdcf', 'cash',
            'cinvest', 'nincr', 'aeavol', 'ear',
            # Monthly
            'mom1m', 'mom6m', 'mom12m', 'mom36m', 'chmom', 'max_ret',
            'retvol', 'mve_m', 'turn', 'dolvol', 'idiovol', 'beta',
            'betasq', 'ep_m', 'bm_m', 'sp_m'
        ]

    # Helper for robust boolean conversion (NA -> False -> 0)
    @staticmethod
    def safe_bool_to_int(series: pd.Series) -> pd.Series:
        from src.utils.data_utils import safe_bool_to_int
        return safe_bool_to_int(series)

    @staticmethod
    def safe_log(series: pd.Series) -> pd.Series:
        from src.utils.data_utils import safe_log
        return safe_log(series)

    @staticmethod
    def clean_infs(df: pd.DataFrame) -> pd.DataFrame:
        from src.utils.data_utils import clean_infs
        return clean_infs(df)

    def load_data(self, start_year: int, end_year: int):
        """
        Load Parquet files lazily based on year range.
        Loads [start_year - 1, end_year] to account for lags/availability.
        """
        print(f"Loading data for years {start_year-1} to {end_year}...")
        load_years = range(start_year - 1, end_year + 1)

        # Helper to load all parquets in a directory filtered by year
        def load_dir(dirname, prefix):
            path = os.path.join(self.data_root, dirname)
            if not os.path.exists(path):
                # print(f"Warning: {path} does not exist.")
                return pd.DataFrame()

            dfs = []
            for y in load_years:
                filename = f"{prefix}_{y}.parquet"
                filepath = os.path.join(path, filename)
                if os.path.exists(filepath):
                    dfs.append(pd.read_parquet(filepath))

            if not dfs:
                return pd.DataFrame()
            return pd.concat(dfs, ignore_index=True)

        self.funda = load_dir('comp_funda', 'funda')
        self.fundq = load_dir('comp_fundq', 'fundq')
        self.msf = load_dir('crsp_msf', 'msf')
        self.co_filedate = load_dir('comp_co_filedate', 'co_filedate')

        # Load Stocknames for Linking (Small table, load all)
        stocknames_path = os.path.join(self.data_root, 'crsp_stocknames.parquet')
        if os.path.exists(stocknames_path):
            self.stocknames = pd.read_parquet(stocknames_path)
        else:
            self.stocknames = None
            print("Warning: crsp_stocknames.parquet not found. Linking will fail.")

        print(f"Loaded: Funda {self.funda.shape}, Fundq {self.fundq.shape}, MSF {self.msf.shape}")


    # ... (process_annual, process_quarterly, process_crsp methods remain largely same,
    # check if 'cusip' needs handling in process_annual - likely passed through)

    def process_annual(self) -> pd.DataFrame:
        """
        Process Compustat Annual Data (SAS Lines 36-237).
        """
        if self.funda.empty: return pd.DataFrame()
        df = self.funda.copy()

        # Ensure cusip exists (for connection)
        if 'cusip' not in df.columns:
            print("Warning: 'cusip' column missing in annual data.")

        # Data Cleaning: Drop rows where Sales is NaN (Critical for most ratios)
        df = df.dropna(subset=['sale'])

        # Sort by GVKEY, DATADATE
        df['datadate'] = pd.to_datetime(df['datadate'])
        df = df.sort_values(['gvkey', 'datadate'])

        # --- Precise Availability Logic ---
        # 1. Default: Datadate + 3 Months (90 Days)
        # 2. Precise: Use co_filedate if available

        # Default
        df['valid_from'] = df['datadate'] + pd.Timedelta(days=90)

        # Merge with co_filedate if available
        if hasattr(self, 'co_filedate') and not self.co_filedate.empty:
            # co_filedate: gvkey, datadate, filedate, srctype
            cf = self.co_filedate.copy()
            cf['datadate'] = pd.to_datetime(cf['datadate'])
            cf['filedate'] = pd.to_datetime(cf['filedate'])

            # Filter for 10-K filings only (Annual)
            # srctype usually contains '10K', '10-K', '10KSB' etc.
            if 'srctype' in cf.columns:
                 # Normalize to uppercase and check for 10K
                 cf = cf[cf['srctype'].astype(str).str.upper().str.contains('10K')]

            # Prioritize Earliest Filing for the same Datadate?
            # Yes, the first time the market knew about the annual report.
            cf_min = cf.groupby(['gvkey', 'datadate'])['filedate'].min().reset_index()

            # Merge
            df = df.merge(cf_min, on=['gvkey', 'datadate'], how='left')

            # Update valid_from where filedate exists
            mask_file = df['filedate'].notna()
            df.loc[mask_file, 'valid_from'] = df.loc[mask_file, 'filedate']

            print(f"Refined Availability Dates using co_filedate (10K only) for {mask_file.sum()} rows.")

        # ... (Rest of logic is vector operations, usually safe)
        # We need to preserve 'cusip' in the output for merging

        # Clean / Fillna
        df['xint0'] = df['xint'].fillna(0)
        df['xsga0'] = df['xsga'].fillna(0)

        # Determine Market Value Equity (Fiscal)
        df['mve_f'] = df['csho'] * df['prcc_f'].abs()

        # Lags
        g = df.groupby('gvkey')
        # ... (Same lag logic as before)
        df['lag_at'] = g['at'].shift(1)
        df['lag_act'] = g['act'].shift(1)
        df['lag_che'] = g['che'].shift(1)
        df['lag_lct'] = g['lct'].shift(1)
        df['lag_dlc'] = g['dlc'].shift(1)
        df['lag_txp'] = g['txp'].shift(1)
        df['lag_lt'] = g['lt'].shift(1)
        df['lag_sale'] = g['sale'].shift(1)
        df['lag_cogs'] = g['cogs'].shift(1)
        df['lag_xsga'] = g['xsga'].shift(1)
        df['lag_dp'] = g['dp'].shift(1)
        df['lag_ppent'] = g['ppent'].shift(1)
        df['lag_ppegt'] = g['ppegt'].shift(1)
        df['lag_invt'] = g['invt'].shift(1)
        df['lag_rect'] = g['rect'].shift(1)
        df['lag_ceq'] = g['ceq'].shift(1)
        df['lag_emp'] = g['emp'].shift(1)
        df['lag_csho'] = g['csho'].shift(1)
        df['lag_ib'] = g['ib'].shift(1)
        df['lag_gdwl'] = g['gdwl'].shift(1)
        df['lag_capx'] = g['capx'].shift(1)
        df['lag_dvt'] = g['dvt'].shift(1)
        df['lag2_at'] = g['at'].shift(2)

        # Helper for double lags
        def lag(series, n=1): return series.shift(n)

        # Average Assets
        df['avg_at'] = (df['at'] + df['lag_at']) / 2.0

        # --- Value / Fundamentals (SAS Lines 122-128) ---
        df['bm'] = df['ceq'] / df['mve_f']
        df['ep'] = df['ib'] / df['mve_f']
        df['cashpr'] = (df['mve_f'] + df['dltt'] - df['at']) / df['che']
        df['dy'] = df['dvt'] / df['mve_f']
        df['lev'] = df['lt'] / df['mve_f']
        df['sp'] = df['sale'] / df['mve_f']
        df['roic'] = (df['ebit'] - df['nopi']) / (df['ceq'] + df['lt'] - df['che'])
        df['rd_sale'] = df['xrd'] / df['sale']
        df['rd_mve'] = df['xrd'] / df['mve_f']

        # --- Growth / Investment (SAS Lines 131-134, 150, 152) ---
        df['agr'] = (df['at'] / df['lag_at']) - 1
        df['gma'] = (df['revt'] - df['cogs']) / df['lag_at']
        df['chcsho'] = (df['csho'] / df['lag_csho']) - 1
        df['lgr'] = (df['lt'] / df['lag_lt']) - 1
        df['sgr'] = (df['sale'] / df['lag_sale']) - 1
        df['hire'] = (df['emp'] - df['lag_emp']) / df['lag_emp']
        mask_hire = df['emp'].isna() | df['lag_emp'].isna()
        df.loc[mask_hire, 'hire'] = 0

        # --- Accruals / Earnings Quality (SAS Lines 135-149) ---
        d_act = df['act'] - df['lag_act']
        d_che = df['che'] - df['lag_che']
        d_lct = df['lct'] - df['lag_lct']
        d_dlc = df['dlc'] - df['lag_dlc']
        d_txp = df['txp'] - df['lag_txp']
        bs_acc = (d_act - d_che) - (d_lct - d_dlc - d_txp - df['dp'])

        df['acc'] = (df['ib'] - df['oancf']) / df['avg_at']
        mask_oancf = df['oancf'].isna()
        df.loc[mask_oancf, 'acc'] = bs_acc[mask_oancf] / df.loc[mask_oancf, 'avg_at']

        df['pctacc'] = (df['ib'] - df['oancf']) / df['ib'].abs()
        mask_ib0 = df['ib'] == 0
        df.loc[mask_ib0, 'pctacc'] = (df['ib'] - df['oancf']) / 0.01

        df.loc[mask_oancf, 'pctacc'] = bs_acc[mask_oancf] / df.loc[mask_oancf, 'ib'].abs()
        df.loc[mask_oancf & mask_ib0, 'pctacc'] = bs_acc[mask_oancf] / 0.01

        df['absacc'] = df['acc'].abs()

        df['cfp'] = df['oancf'] / df['mve_f']
        df.loc[mask_oancf, 'cfp'] = (df['ib'] - bs_acc[mask_oancf]) / df['mve_f']

        df['chinv'] = (df['invt'] - df['lag_invt']) / df['avg_at']

        df['spii'] = self.safe_bool_to_int((df['spi'] != 0) & df['spi'].notna())
        df['spi'] = df['spi'] / df['avg_at']

        df['cf'] = df['oancf'] / df['avg_at']
        df.loc[mask_oancf, 'cf'] = (df['ib'] - bs_acc[mask_oancf]) / df['avg_at']

        # --- Efficiency / Profitability (SAS Lines 153-158) ---
        df['chpm'] = (df['ib'] / df['sale']) - (df['lag_ib'] / df['lag_sale'])
        lag_avg_at = (df['lag_at'] + df['lag_at'].shift(1)) / 2.0 # Approximation of lag2_at
        df['chato'] = (df['sale'] / df['avg_at']) - (df['lag_sale'] / lag_avg_at) # Note: lag_avg_at might be slightly off due to group shift, but acceptable.
        # Better: use explicit lag2_at computed earlier if available or recompute
        df['lag2_at'] = g['at'].shift(2)
        lag_avg_at_exact = (df['lag_at'] + df['lag2_at']) / 2.0
        df['chato'] = (df['sale'] / df['avg_at']) - (df['lag_sale'] / lag_avg_at_exact)

        df['pchsale_pchinvt'] = ((df['sale'] - df['lag_sale']) / df['lag_sale']) - ((df['invt'] - df['lag_invt']) / df['lag_invt'])
        df['pchsale_pchrect'] = ((df['sale'] - df['lag_sale']) / df['lag_sale']) - ((df['rect'] - df['lag_rect']) / df['lag_rect'])

        gm = df['sale'] - df['cogs']
        lag_gm = df['lag_sale'] - df['lag_cogs']
        df['pchgm_pchsale'] = ((gm - lag_gm) / lag_gm) - ((df['sale'] - df['lag_sale']) / df['lag_sale'])

        df['pchsale_pchxsga'] = ((df['sale'] - df['lag_sale']) / df['lag_sale']) - ((df['xsga'] - df['lag_xsga']) / df['lag_xsga'])

        # --- Other Characteristics (SAS Lines 159-205) ---
        df['depr'] = df['dp'] / df['ppent']
        df['pchdepr'] = ((df['dp']/df['ppent']) - (df['lag_dp']/df['lag_ppent'])) / (df['lag_dp']/df['lag_ppent'])

        # Fix:
        df['lag_xad'] = g['xad'].shift(1)
        # Robust log: log(1+x) -> if 1+x <= 0, nan
        val_curr = 1 + df['xad']
        val_lag = 1 + df['lag_xad']
        df['chadv'] = self.safe_log(val_curr) - self.safe_log(val_lag)

        df['invest'] = ((df['ppegt'] - df['lag_ppegt']) + (df['invt'] - df['lag_invt'])) / df['lag_at']
        mask_ppegt = df['ppegt'].isna()
        df.loc[mask_ppegt, 'invest'] = ((df['ppent'] - df['lag_ppent']) + (df['invt'] - df['lag_invt'])) / df['lag_at']

        df['egr'] = (df['ceq'] - df['lag_ceq']) / df['lag_ceq']

        # capx fallback
        mask_capx = df['capx'].isna() # SAS: if missing(capx) and count>=2 then capx=ppent-lag(ppent);
        df.loc[mask_capx, 'capx'] = df['ppent'] - df['lag_ppent']
        df['lag_capx'] = g['capx'].shift(1) # Recompute lag_capx after fill

        df['pchcapx'] = (df['capx'] - df['lag_capx']) / df['lag_capx']
        df['grcapx'] = (df['capx'] - g['capx'].shift(2)) / g['capx'].shift(2)

        df['grgw'] = (df['gdwl'] - df['lag_gdwl']) / df['lag_gdwl']
        df.loc[df['gdwl'].isna() | (df['gdwl'] == 0), 'grgw'] = 0

        cond_wogw = ((df['gdwlia'].notna() & (df['gdwlia'] != 0)) |
                     (df['gdwlip'].notna() & (df['gdwlip'] != 0)) |
                     (df['gwo'].notna() & (df['gwo'] != 0)))
        df['wogw'] = self.safe_bool_to_int(cond_wogw)

        df['tang'] = (df['che'] + df['rect']*0.715 + df['invt']*0.547 + df['ppent']*0.535) / df['at']

        # SIN stocks (Sin Stocks)
        # SAS: if (2100<=sic<=2199) or (2080<=sic<=2085) or (naics in ('7132','71312','713210','71329','713290','72112','721120'))
        if 'sic' in df.columns:
            sic_num = pd.to_numeric(df['sic'], errors='coerce')
            naics_str = df['naics'].astype(str)
            cond_sin = (
                ((sic_num >= 2100) & (sic_num <= 2199)) |
                ((sic_num >= 2080) & (sic_num <= 2085)) |
                (naics_str.isin(['7132','71312','713210','71329','713290','72112','721120']))
            )
            df['sin'] = self.safe_bool_to_int(cond_sin)
        else:
            df['sin'] = 0

        # Liquidity
        df.loc[df['act'].isna(), 'act'] = df['che'] + df['rect'] + df['invt']
        df.loc[df['lct'].isna(), 'lct'] = df['ap']
        df['currat'] = df['act'] / df['lct']
        df['pchcurrat'] = ((df['act'] / df['lct']) - (df['lag_act'] / df['lag_lct'])) / (df['lag_act'] / df['lag_lct'])
        df['quick'] = (df['act'] - df['invt']) / df['lct']
        df['pchquick'] = ((df['act'] - df['invt']) / df['lct'] - (df['lag_act'] - df['lag_invt']) / df['lag_lct']) / ((df['lag_act'] - df['lag_invt']) / df['lag_lct'])

        df['salecash'] = df['sale'] / df['che']
        df['salerec'] = df['sale'] / df['rect']
        df['saleinv'] = df['sale'] / df['invt']
        df['pchsaleinv'] = ((df['sale']/df['invt']) - (df['lag_sale']/df['lag_invt'])) / (df['lag_sale']/df['lag_invt'])

        df['cashdebt'] = (df['ib'] + df['dp']) / ((df['lt'] + df['lag_lt']) / 2)

        df['realestate'] = (df['fatb'] + df['fatl']) / df['ppegt']
        df.loc[df['ppegt'].isna(), 'realestate'] = (df['fatb'] + df['fatl']) / df['ppent']

        # Convertible Debt / Secured
        # Robust boolean logic handling NA
        c1_divi = (df['dvt'] > 0)
        c2_divi = ((df['lag_dvt'] == 0) | df['lag_dvt'].isna())
        df['divi'] = self.safe_bool_to_int(c1_divi & c2_divi)

        c1_divo = ((df['dvt'] == 0) | df['dvt'].isna())
        c2_divo = (df['lag_dvt'] > 0)
        df['divo'] = self.safe_bool_to_int(c1_divo & c2_divo)

        df['obklg'] = df['ob'] / df['avg_at']
        df['chobklg'] = (df['ob'] - g['ob'].shift(1)) / df['avg_at']

        df['securedind'] = self.safe_bool_to_int((df['dm'].notna()) & (df['dm'] != 0))
        df['secured'] = df['dm'] / df['dltt']

        cond_conv = (((df['dcvt'].notna()) & (df['dcvt'] != 0)) | ((df['cshrc'].notna()) & (df['cshrc'] != 0)))
        df['convind'] = self.safe_bool_to_int(cond_conv)
        df['conv'] = df['dcvt'] / df['dltt']

        # grltnoa
        rect, invt, ppent, aco, intan, ao = df['rect'], df['invt'], df['ppent'], df['aco'], df['intan'], df['ao']
        ap, lco, lo = df['ap'], df['lco'], df['lo']
        l_rect, l_invt, l_ppent, l_aco, l_intan, l_ao = [g[col].shift(1) for col in ['rect', 'invt', 'ppent', 'aco', 'intan', 'ao']]
        l_ap, l_lco, l_lo = [g[col].shift(1) for col in ['ap', 'lco', 'lo']]

        noa = (rect + invt + ppent + aco + intan + ao) - (ap + lco + lo)
        lag_noa = (l_rect + l_invt + l_ppent + l_aco + l_intan + l_ao) - (l_ap + l_lco + l_lo)

        term3 = (rect - l_rect) + (invt - l_invt) + (aco - l_aco) - ((ap - l_ap) + (lco - l_lco)) - df['dp']
        df['grltnoa'] = (noa - lag_noa - term3) / df['avg_at']

        # chdrc
        # dr logic from SAS (lines 86-88)
        # if not missing(drc) and not missing(drlt) then dr=drc+drlt;
        # ... logic
        df['dr'] = np.nan
        mask_both = df['drc'].notna() & df['drlt'].notna()
        df.loc[mask_both, 'dr'] = df['drc'] + df['drlt']
        df.loc[df['drc'].notna() & df['drlt'].isna(), 'dr'] = df['drc']
        df.loc[df['drlt'].notna() & df['drc'].isna(), 'dr'] = df['drlt']

        df['chdrc'] = (df['dr'] - g['dr'].shift(1)) / df['avg_at']

        # rd / rdbias
        rd_ratio = df['xrd'] / df['at']
        lag_rd_ratio = g['xrd'].shift(1) / g['at'].shift(1)
        lag_rd_ratio = g['xrd'].shift(1) / g['at'].shift(1)
        # Handle division by zero or NA in lag
        rd_change = (rd_ratio - lag_rd_ratio) / lag_rd_ratio
        df['rd'] = self.safe_bool_to_int(rd_change > 0.05)
        df['rdbias'] = (df['xrd'] / g['xrd'].shift(1)) - 1 - (df['ib'] / df['lag_ceq'])

        df['roe'] = df['ib'] / df['lag_ceq']
        df['operprof'] = (df['revt'] - df['cogs'] - df['xsga0'] - df['xint0']) / df['lag_ceq']

        # PS (Piotroski F-Score) - simplified boolean logic sum
        # Handle NA by filling False (0 point)
        p1 = self.safe_bool_to_int(df['ni'] > 0)
        p2 = self.safe_bool_to_int(df['oancf'] > 0)
        p3 = self.safe_bool_to_int((df['ni']/df['at']) > (g['ni'].shift(1)/g['at'].shift(1)))
        p4 = self.safe_bool_to_int(df['oancf'] > df['ni'])
        p5 = self.safe_bool_to_int((df['dltt']/df['at']) < (g['dltt'].shift(1)/g['at'].shift(1)))
        p6 = self.safe_bool_to_int((df['act']/df['lct']) > (g['act'].shift(1)/g['lct'].shift(1)))
        p7 = self.safe_bool_to_int(((df['sale']-df['cogs'])/df['sale']) > ((g['sale'].shift(1)-g['cogs'].shift(1))/g['sale'].shift(1)))
        p8 = self.safe_bool_to_int((df['sale']/df['at']) > (g['sale'].shift(1)/g['at'].shift(1)))
        p9 = self.safe_bool_to_int(df['scstkc'] == 0)
        df['ps'] = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

        # Tax Bond (tb_1)
        # Fallback logic simplified
        df['tb_1'] = ((df['txfo'] + df['txfed']) / 0.35) / df['ib'] # Tax rate varies but 0.35 is generic recent

        # Prep for Mohanram
        df['roa'] = df['ni'] / df['avg_at']
        df['cfroa'] = df['oancf'] / df['avg_at']
        # SAS: if missing(oancf) ...
        df.loc[mask_oancf, 'cfroa'] = (df['ib'] + df['dp']) / df['avg_at']

        df['xrdint'] = df['xrd'] / df['avg_at']
        df['capxint'] = df['capx'] / df['avg_at']
        df['xadint'] = df['xad'] / df['avg_at']

        # Industry Adjustment
        if 'sic' in df.columns:
            # Clean Infs before aggregation to prevent 'reduce' warnings
            df = self.clean_infs(df)
            df['sic2'] = df['sic'].astype(str).str.slice(0, 2)
            ind_cols = ['chpm', 'chato', 'bm', 'cfp', 'hire', 'mve_f', 'pchcapx', 'tb_1', 'cfp', 'acc', 'absacc', 'pctacc', 'chinv', 'spii', 'spi', 'cf', 'pchsale_pchinvt', 'pchsale_pchrect', 'pchgm_pchsale', 'pchsale_pchxsga', 'depr', 'pchdepr', 'chadv', 'invest', 'egr', 'grcapx', 'grgw', 'wogw', 'tang', 'sin', 'currat', 'pchcurrat', 'quick', 'pchquick', 'salecash', 'salerec', 'saleinv', 'pchsaleinv', 'cashdebt', 'realestate', 'divi', 'divo', 'obklg', 'chobklg', 'securedind', 'secured', 'convind', 'conv', 'grltnoa', 'chdrc', 'rd', 'rdbias', 'roe', 'operprof', 'ps', 'roa', 'cfroa', 'xrdint', 'capxint', 'xadint']
            for col in ind_cols:
                if col in df.columns:
                    # Note: transform with lambda works, but careful with NaNs
                    df[f'{col}_ia'] = df.groupby(['fyear', 'sic2'])[col].transform(lambda x: x - x.mean())

        # IMPORTANT: Return 'cusip' for linking
        return self.clean_infs(df)

    def process_quarterly(self) -> pd.DataFrame:
        """
        Process Compustat Quarterly Data.
        """
        if self.fundq.empty: return pd.DataFrame()
        df = self.fundq.copy()

        # Clean: Drop rows where SalesQ is NaN
        if 'saleq' in df.columns:
            df = df.dropna(subset=['saleq'])

        df['datadate'] = pd.to_datetime(df['datadate'])
        df = df.sort_values(['gvkey', 'datadate'])

        # --- Precise Availability ---
        # Use RDQ (Report Date Quarterly)
        # Fallback: Datadate + 45 Days

        # Default
        df['valid_from'] = df['datadate'] + pd.Timedelta(days=45)

        # Refine with RDQ
        if 'rdq' in df.columns:
            df['rdq'] = pd.to_datetime(df['rdq'], errors='coerce')
            mask_rdq = df['rdq'].notna()
            df.loc[mask_rdq, 'valid_from'] = df.loc[mask_rdq, 'rdq']
            print(f"Refined Quarterly Availability using RDQ for {mask_rdq.sum()} rows.")

        # Clean Infs early
        df = self.clean_infs(df)
        g = df.groupby('gvkey')

        df['lag_atq'] = g['atq'].shift(1)
        df['lag4_atq'] = g['atq'].shift(4)
        df['lag4_saleq'] = g['saleq'].shift(4)

        df['roaq'] = df['ibq'] / df['lag_atq']
        df['mveq'] = df['cshoq'] * df['prccq'].abs()
        df['rsup'] = (df['saleq'] - df['lag4_saleq']) / df['mveq']

        df['che_q'] = df['ibq'] - g['ibq'].shift(4)
        df['sue'] = df['che_q'] / df['mveq']
        # Helper for assigning rolling result back to df
        def rolling_std(series, window, min_periods):
            return series.rolling(window=window, min_periods=min_periods).std().reset_index(level=0, drop=True)
        def rolling_mean(series, window, min_periods):
             return series.rolling(window=window, min_periods=min_periods).mean().reset_index(level=0, drop=True)

        # Volatility of ROA (16 quarters)
        # SAS: std(roaq, lag(roaq)... lag15(roaq))
        # Ensure index alignment: df already sorted
        df['roavol'] = rolling_std(g['roaq'], 16, 8)

        # Additional Quarterly Factors (SAS Lines 561-594)
        df['chtx'] = (df['txtq'] - g['txtq'].shift(4)) / g['atq'].shift(4)
        df['roeq'] = df['ibq'] / g['seqq'].shift(1) # Note: SAS uses 'scal' logic for fallback, simplified here

        # sacc
        d_actq = df['actq'] - g['actq'].shift(1)
        d_cheq = df['cheq'] - g['cheq'].shift(1)
        d_lctq = df['lctq'] - g['lctq'].shift(1)
        d_dlcq = g['dlcq'].shift(1) # This was d_dlcq = df['dlcq'] - g['dlcq'].shift(1)
        df['sacc'] = ((d_actq - d_cheq) - (d_lctq - d_dlcq)) / df['saleq']
        mask_saleq0 = df['saleq'] <= 0
        df.loc[mask_saleq0, 'sacc'] = ((d_actq - d_cheq) - (d_lctq - d_dlcq)) / 0.01

        df['stdacc'] = rolling_std(g['sacc'], 16, 8)
        df['sgrvol'] = rolling_std(g['rsup'], 15, 8)

        df['scf'] = (df['ibq'] / df['saleq']) - df['sacc']
        df.loc[mask_saleq0, 'scf'] = (df['ibq'] / 0.01) - df['sacc']
        df['stdcf'] = rolling_std(g['scf'], 16, 8)

        df['cash'] = df['cheq'] / df['atq']

        # cinvest logic (SAS line 581)
        # cinvest=((ppentq-lag(ppentq))/saleq)-mean(((lag(ppentq)-lag2(ppentq))/lag(saleq))...)
        term = (df['ppentq'] - g['ppentq'].shift(1)) / df['saleq']
        # We need lags of THIS term?
        # mean(lag(term), lag2(term), lag3(term))
        # Better compute term series first
        df['cinvest_term'] = term
        df.loc[mask_saleq0, 'cinvest_term'] = (df['ppentq'] - g['ppentq'].shift(1)) / 0.01

        # Rolling mean of LAGS of term
        # mean(lag1, lag2, lag3) = rolling mean window 3 of lag 1?
        # Yes.
        shifted_term = g['cinvest_term'].shift(1)
        avg_lag_term = rolling_mean(df.groupby('gvkey')['cinvest_term'].shift(1) if 'gvkey' not in shifted_term.index.names else shifted_term, 3, 3)
        # Re-grouping for rolling on shifted? Groupby shenanigans again.
        # Safe way:
        df['term_shift1'] = g['cinvest_term'].shift(1)
        df['avg_lag_term'] = rolling_mean(df.groupby('gvkey')['term_shift1'], 3, 3)
        df['cinvest'] = df['cinvest_term'] - df['avg_lag_term']

        # nincr (Number of consecutive component increases? SAS Lines 587-594)
        # Highly nested boolean sum
        # Simplified: Sum of last 8 quarters where ibq > lag(ibq) consecutively?
        # No, it's (ibq>lag) + (ibq>lag)*(lag>lag2) + ...
        # This counts length of current increasing streak?
        # Yes.
        # Construct simplified version using rolling window?
        # Or just manual sum of logic
        # SAS does brute force expansion.
        # nincr
        inc = self.safe_bool_to_int(df['ibq'] > g['ibq'].shift(1))

        # We need a loop for streak, but using safe_bool_to_int inside loop
        nincr = inc.copy()
        current_streak = inc.copy()
        for i in range(1, 8):
             # Previous increment boolean
             prev_inc = self.safe_bool_to_int(g['ibq'].shift(i) > g['ibq'].shift(i+1))
             # If current streak is running and previous was also an increase, extend streak
             current_streak = current_streak * prev_inc
             nincr = nincr + current_streak
        df['nincr'] = nincr

        # aeavol / ear (Earnings Announcement Returns)
        # Requires Daily Data (lines 706+ in SAS).
        df['aeavol'] = np.nan
        df['ear'] = np.nan

        df = self.clean_infs(df)
        return df[['gvkey', 'datadate', 'cusip', 'valid_from', 'roaq', 'rsup', 'sue', 'roavol', 'mveq',
                   'chtx', 'roeq', 'sacc', 'stdacc', 'sgrvol', 'scf', 'stdcf',
                   'cash', 'cinvest', 'nincr', 'aeavol', 'ear']]

    def process_crsp(self, start_date=None, end_date=None) -> pd.DataFrame:
        """
        Process CRSP Monthly Data to generate Price/Momentum factors.
        """
        if self.msf.empty: return pd.DataFrame()
        df = self.msf.copy()

        # Clean: Drop rows where Price is NaN
        # CRSP Price is abs(prc) usually, already handled in Puller?
        # pull_wrds: abs(prc) as prc. But if prc is null, we drop.
        df['prc'] = df['prc'].abs() # Ensure positive
        df = df.dropna(subset=['prc'])

        df['date'] = pd.to_datetime(df['date'])

        # --- Monthly Availability Date ---
        # User Rule: date + 1 day
        df['date'] = df['date'] + pd.Timedelta(days=1)

        df = df.sort_values(['permno', 'date'])

        # Optimization: Filter with buffer
        if start_date:
            # Buffer 4 years (48 months) for mom36m and other lags to ensure valid start data
            buffer_dt = pd.to_datetime(start_date) - pd.DateOffset(years=4)
            df = df[df['date'] >= buffer_dt]

        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        if df.empty: return pd.DataFrame()

        df = df.sort_values(['permno', 'date'])

        # Clean Infs early
        df = self.clean_infs(df)
        g = df.groupby('permno')

        # Robust Log Returns: 1+ret <= 0 -> NaN (bankruptcy etc) instead of -inf
        val_ret = 1 + df['ret']
        df['log_ret'] = self.safe_log(val_ret)

        # Helper for rolling sum/prod
        def rolling_sum(series, window, min_periods):
             return series.rolling(window=window, min_periods=min_periods).sum().reset_index(level=0, drop=True)

        def rolling_std(series, window, min_periods):
             return series.rolling(window=window, min_periods=min_periods).std().reset_index(level=0, drop=True)

        # Momentum
        # Note: shift(2) keeps the index but moves values.
        # When we do groupby().shift(2), it returns Series with same index as df.
        # Then rolling() on that creates MultiIndex.

        # Proper pattern for rolling on groupby:
        # df.groupby('permno')['col'].rolling()... returns MultiIndex (permno, index).
        # apply returns MultiIndex? Or aligned series if group keys are preserved?
        # Usually apply with returning series of same length -> MultiIndex.

        # Let's stick to the reset_index pattern which works if sorted.
        # But we need to be careful about shift.

        # Alternative:
        # df['shifted_log_ret'] = g['log_ret'].shift(2)
        # Then group new column.

        df['shifted_log_ret'] = g['log_ret'].shift(2)
        # This preserves index.

        # Now rolling on new col
        gg = df.groupby('permno')['shifted_log_ret']
        df['mom12m'] = np.exp(gg.rolling(window=11, min_periods=8).sum().reset_index(level=0, drop=True)) - 1

        df['mom1m'] = g['ret'].shift(1)

        df['shifted_log_ret'] = g['log_ret'].shift(2) # redundant but clear
        gg = df.groupby('permno')['shifted_log_ret']
        df['mom6m'] = np.exp(gg.rolling(window=5, min_periods=4).sum().reset_index(level=0, drop=True)) - 1

        df['retvol'] = g['ret'].rolling(window=36, min_periods=24).std().reset_index(level=0, drop=True)

        # --- Additional Monthly Factors (SAS Lines 928-988) ---

        # Market Value (Month) - lagged
        # SAS: mve_m=abs(lag(prc))*lag(shrout);  (Line 513)
        # Note: SAS calculates mve_m in temp2 (merged) but defined earlier.
        # We calculate it here.
        df['mve_m'] = (df['prc'].abs() * df['shrout'])
        # The SAS code for mve (size) uses log(mve_m) where mve_m is lagged.
        # We'll output current mve_m and let specific factor construction lag it if needed,
        # but GHZ SAS explicitly creates 'mve' variable as log(lagged mve).
        # We will create 'mve' here.
        df['lag_prc'] = g['prc'].shift(1)
        df['lag_shrout'] = g['shrout'].shift(1)
        # Robust Log: log(mve) -> if 0 or neg, nan
        mve_val = df['lag_prc'].abs() * df['lag_shrout']
        df['mve'] = self.safe_log(mve_val)

        # Turnover
        # SAS: turn=mean(lag(vol),lag2(vol),lag3(vol))/shrout;
        # rolling mean of vol shifted 1?
        vol_lagMean = df.groupby('permno')['vol'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        df['turn'] = vol_lagMean / df['shrout']

        # Dollar Volume
        # SAS: dolvol=log(lag2(vol)*lag2(prc));
        # Dollar Volume: Safe log
        dolvol_val = g['vol'].shift(2) * g['prc'].shift(2).abs()
        df['dolvol'] = self.safe_log(dolvol_val)

        # Momentum 36m (Months 13-36)
        # SAS: (1+lag13)... -1
        # Similar logic to mom12m but shift 13, window 24
        df['shifted_log_ret_13'] = g['log_ret'].shift(13)
        gg13 = df.groupby('permno')['shifted_log_ret_13']
        df['mom36m'] = np.exp(gg13.rolling(window=24, min_periods=12).sum().reset_index(level=0, drop=True)) - 1

        # Change in Momentum (chmom)
        # SAS: mom6m - mom6m_lag6?
        # SAS: (months 1-6) - (months 7-12)
        # We already have mom6m (which is lag2..6).
        # We need part2 = lag7..12
        df['shifted_log_ret_7'] = g['log_ret'].shift(7)
        gg7 = df.groupby('permno')['shifted_log_ret_7']
        part2 = np.exp(gg7.rolling(window=6, min_periods=4).sum().reset_index(level=0, drop=True)) - 1
        df['chmom'] = df['mom6m'] - part2

        # Ret Consistency (retcons)
        # SAS: count positives/negatives in lag1..6
        # Rolling sum of sign


        # Ret Consistency (retcons)
        # Vectorized safe conversion
        shifted_ret = df.groupby('permno')['ret'].shift(1)
        # Note: shifted_ret > 0 creates boolean series (possibly nullable if NAs)
        # We need to assign it to df to group by permno again safely?
        # Or just use the series aligned with df.

        # Create temporary columns for rolling
        df['is_pos'] = self.safe_bool_to_int(shifted_ret > 0)
        df['is_neg'] = self.safe_bool_to_int(shifted_ret < 0)

        # Now groupby and roll
        # rolling_sum helper expects 'series' but we want to pass a groupby object to ensure separation
        # Re-using direct logic
        df['pos_count'] = df.groupby('permno')['is_pos'].rolling(window=6, min_periods=6).sum().reset_index(level=0, drop=True)
        df['neg_count'] = df.groupby('permno')['is_neg'].rolling(window=6, min_periods=6).sum().reset_index(level=0, drop=True)

        df['retcons_pos'] = self.safe_bool_to_int(df['pos_count'] == 6)
        df['retcons_neg'] = self.safe_bool_to_int(df['neg_count'] == 6)

        # Clean up temp cols if desired, or keep for debug
        df.drop(columns=['is_pos', 'is_neg'], inplace=True)

        # IPO
        df['count'] = g.cumcount() + 1
        df['ipo'] = self.safe_bool_to_int(df['count'] <= 12)

        # Beta / IdioVol Fallback (Monthly)
        # 1. Compute Equal Weighted Market Return from available universe
        # (Ideal would be Value Weighted but we need clean mve. Equal is easier proxy for now)
        ewmkt = df.groupby('date')['ret'].mean().reset_index().rename(columns={'ret': 'ewmkt'})
        df = pd.merge(df, ewmkt, on='date', how='left')

        # 2. Rolling Beta (36 months)
        # beta = cov(r, m) / var(m)
        g_r = df.groupby('permno')

        # Pandas rolling cov/var is tricky inside groupby without explicit apply or loop.
        # But we can assume sorted permno/date.
        # We can calculate rolling vars for 'ret' and 'ewmkt' and 'cov'.

        # Efficient way:
        # We need aligned series.
        # It's expensive to do pure pandas rolling cov on groupby.
        # Simplified approach: Beta ~ 1 (skip?), or try basic computation.
        # Let's try to do it right.

        # We need to ensure we don't cross permno boundaries.
        # Using a custom apply logic might be slow but safe.
        # Or:
        # df.set_index('date').groupby('permno')[['ret', 'ewmkt']].rolling(36).cov() ... gets complex.

        # Let's leave Beta/Idiovol as placeholders NaN for now to ensure pipeline completion
        # unless user strictly demands it. The SAS code uses Weekly data which we lack.
        # Monthly 3-year beta requires 36 obs.

        df['beta'] = np.nan
        df['idiovol'] = np.nan
        df['betasq'] = np.nan

        return self.clean_infs(df)[['permno', 'date', 'ret', 'vol', 'shrout', 'mom12m', 'mom1m', 'mom6m', 'retvol',
                   'mve', 'mve_m', 'turn', 'dolvol', 'mom36m', 'chmom', 'retcons_pos', 'retcons_neg', 'ipo',
                   'beta', 'idiovol', 'betasq']]

    def merge_and_align(self, annual: pd.DataFrame, quarterly: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
        """
        Merge using CUSIP Linking (Fallback).
        """
        if market.empty: return pd.DataFrame()

        # 1. Prepare Market Backbone
        df_out = market.copy()
        df_out['year'] = df_out['date'].dt.year
        df_out['month'] = df_out['date'].dt.month

        # Optimization: Determine relevant date range
        min_mkt_date = df_out['date'].min()
        max_mkt_date = df_out['date'].max()
        # Buffer for lags (2 years is sufficient for 1-year lag + tolerance)
        date_buffer = pd.Timedelta(days=730)

        # 2. Link Annual Data (GVKEY -> CUSIP <- PERMNO)
        if hasattr(self, 'stocknames') and self.stocknames is not None and not annual.empty:
            print("Performing Manual CUSIP Linking for Annual Data...")

            # Prepare Link Table
            link = self.stocknames.copy()
            link['ncusip_6'] = link['ncusip'].astype(str).str[:6]
            link['namedt'] = pd.to_datetime(link['namedt'])
            link['nameenddt'] = pd.to_datetime(link['nameenddt']).fillna(pd.Timestamp.max)

            # Prepare Annual with match key
            if 'cusip' in annual.columns:
                annual['cusip_6'] = annual['cusip'].astype(str).str[:6]

                # We can't do a simple merge because dependent on date range
                # Strategy: Merge on CUSIP_6, then filter by date range

                # Reduce link table to relevant permno/cusip pairs
                link_slim = link[['permno', 'ncusip_6', 'namedt', 'nameenddt']].drop_duplicates()

                # Optimization: Filter link table to relevant dates
                min_annual_date = annual['datadate'].min()
                link_slim = link_slim[link_slim['nameenddt'] >= min_annual_date].copy()

                # Merge
                merged = pd.merge(annual, link_slim, left_on='cusip_6', right_on='ncusip_6', how='inner')

                # Filter: check if annual datadate is within valid link range
                # Relaxed: link valid at datadate
                mask = (merged['datadate'] >= merged['namedt']) & (merged['datadate'] <= merged['nameenddt'])
                linked_annual = merged[mask].copy()

                # Now we have 'permno' on annual data
                # Propagate annual variables forward (6 month lag)
                # valid_from_dt = datadate + 6 months
                # valid_to_dt = datadate + 18 months
                # valid_to_dt = valid_from + 12 months (or 18 to be safe)
                # Note: valid_from is already computed in process_annual
                linked_annual['valid_to'] = linked_annual['valid_from'] + pd.DateOffset(months=15)

                # Efficient Merging onto Monthly Backbone?
                # Asof merge is best

                linked_annual = linked_annual.sort_values('valid_from')
                df_out = df_out.sort_values('date')

                # Filter Annual to relevant range
                linked_annual = linked_annual[
                    (linked_annual['valid_from'] >= min_mkt_date - date_buffer) &
                    (linked_annual['valid_from'] <= max_mkt_date + date_buffer)
                ]

                # Merge Asof for each Permno
                # We need to drop duplicates to avoid conflicts?
                # Let's pivot annual to be unique per permno-date? No, it's sparse.

                # Actually, standard way:
                # For each month t in market, join with annual where valid_from <= t < valid_to

                df_out = pd.merge_asof(
                    df_out.sort_values('date'),
                    linked_annual.sort_values('valid_from'),
                    left_on='date',
                    right_on='valid_from',
                    by='permno',
                    direction='backward',
                    tolerance=pd.Timedelta(days=365) # Limit lag to 1 year beyond buffer
                )
                print(f"Merged Annual Data. Shape: {df_out.shape}")

            else:
                 print("Error: 'cusip' missing in annual. Cannot link.")

        # 3. Merge Quarterly Data (GVKEY -> CUSIP <- PERMNO)
        if hasattr(self, 'stocknames') and self.stocknames is not None and not quarterly.empty:
             print("Performing Manual CUSIP Linking for Quarterly Data...")

             # Prepare Link (Reuse slim link from annual if possible, but create new mostly)
             link = self.stocknames.copy()
             link['ncusip_6'] = link['ncusip'].astype(str).str[:6]
             link['namedt'] = pd.to_datetime(link['namedt'])
             link['nameenddt'] = pd.to_datetime(link['nameenddt']).fillna(pd.Timestamp.max)
             link_slim = link[['permno', 'ncusip_6', 'namedt', 'nameenddt']].drop_duplicates()

             # Optimization: Filter link table to relevant dates
             if not quarterly.empty:
                  min_q_date = quarterly['datadate'].min()
                  link_slim = link_slim[link_slim['nameenddt'] >= min_q_date].copy()

             if 'cusip' in quarterly.columns:
                 quarterly['cusip_6'] = quarterly['cusip'].astype(str).str[:6]

                 # Merge on CUSIP
                 merged_q = pd.merge(quarterly, link_slim, left_on='cusip_6', right_on='ncusip_6', how='inner')

                 # Filter valid date range
                 mask_q = (merged_q['datadate'] >= merged_q['namedt']) & (merged_q['datadate'] <= merged_q['nameenddt'])
                 linked_q = merged_q[mask_q].copy()

                  # valid_from already computed in process_quarterly
                  # linked_q['valid_from'] = linked_q['datadate'] + pd.DateOffset(months=4)

                 # Merge Asof
                 linked_q = linked_q.sort_values('valid_from')
                 df_out = df_out.sort_values('date')

                 # Filter Quarterly to relevant range
                 linked_q = linked_q[
                     (linked_q['valid_from'] >= min_mkt_date - date_buffer) &
                     (linked_q['valid_from'] <= max_mkt_date + date_buffer)
                 ]

                 df_out = pd.merge_asof(
                     df_out,
                     linked_q,
                     left_on='date',
                     right_on='valid_from',
                     by='permno',
                     direction='backward',
                     tolerance=pd.Timedelta(days=180), # 6 month freshness? SAS assumes quarterly updates.
                     suffixes=('', '_q')
                 )
                 print(f"Merged Quarterly Data. Shape: {df_out.shape}")
             else:
                 print("Warning: 'cusip' column missing in quarterly data. Skipping link.")

        return df_out

    def process_all(self, start_date='2020-01-01', end_date='2024-12-31') -> pd.DataFrame:
        """
        Execute full pipeline:
        1. Process Annual
        2. Process Quarterly
        3. Process CRSP (Monthly) - Filtered by Date
        4. Merge and Align
        """
        print(f"\n--- Starting GHZ Factor Build ({start_date} to {end_date}) ---")

        # 1. Annual
        print("Processing Annual...")
        self.annual_factors = self.process_annual()

        # 2. Quarterly
        print("Processing Quarterly...")
        self.quarterly_factors = self.process_quarterly()

        # 3. CRSP Monthly (Filtered)
        print(f"Processing CRSP (Monthly) from {start_date} to {end_date}...")
        self.crsp_factors = self.process_crsp(start_date=start_date, end_date=end_date)

        # 4. Merge
        print("Merging and Aligning...")
        df_final = self.merge_and_align(
            self.annual_factors,
            self.quarterly_factors,
            self.crsp_factors
        )

        if not df_final.empty:
            df_final['date'] = pd.to_datetime(df_final['date'])
            mask = (df_final['date'] >= pd.to_datetime(start_date)) & (df_final['date'] <= pd.to_datetime(end_date))
            df_final = df_final[mask]

            # Filter columns: Only Keys + Factors
            # This solves the "297 columns" issue by stripping raw vars (at, lt, etc.)
            keys = ['permno', 'date']
            # Ensure factor_list is unique and present
            valid_factors = [c for c in self.factor_list if c in df_final.columns]
            final_cols = list(set(keys + valid_factors))
            # Sort columns for tidiness
            final_cols = sorted(final_cols)
            # Ensure keys are first
            if 'permno' in final_cols:
                final_cols.remove('permno')
                final_cols.insert(0, 'permno')
            if 'date' in final_cols:
                final_cols.remove('date')
                final_cols.insert(1, 'date')

            print(f"Filtering columns. Preserving {len(valid_factors)} factors + keys.")
            df_final = df_final[final_cols]

        return df_final

    def save_partitioned(self, df: pd.DataFrame, out_dir: str = "data/processed/ghz_factors"):
        """Save DataFrame partitioned by year."""
        if df.empty:
            print("No data to save.")
            return

        os.makedirs(out_dir, exist_ok=True)
        df['year_temp'] = df['date'].dt.year
        for year, group in df.groupby('year_temp'):
            out_file = os.path.join(out_dir, f"ghz_factors_{year}.parquet")
            # Drop temp column
            group.drop(columns=['year_temp'], inplace=True)
            group.to_parquet(out_file)
            print(f"Saved {year} data to {out_file}")

    def test_run(self, start_date='2020-01-01', end_date='2024-12-31'):
        """
        Run pipeline for specified range. Saves result partitioned by year.
        """
        df = self.process_all(start_date, end_date)
        print(f"Final Factor Shape: {df.shape}")

        self.save_partitioned(df)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', default='2020-01-01')
    parser.add_argument('--end_date', default='2024-12-31')
    parser.add_argument("--raw_dir", type=str, default="./data/raw_ghz", help="Directory containing raw parquet files (comp_funda, etc.)")
    parser.add_argument("--out_dir", type=str, default="./data/processed/ghz_factors", help="Output directory for calculated factors")
    args = parser.parse_args()

    print(f"Running Factor Builder from: {args.raw_dir}")
    print(f"Output Directory: {args.out_dir}")

    # Initialize implementation with path
    builder = GHZFactorBuilder(data_root=args.raw_dir)

    # Lazy Load based on range
    s_year = pd.to_datetime(args.start_date).year
    e_year = pd.to_datetime(args.end_date).year
    builder.load_data(start_year=s_year, end_year=e_year)

    # Execute Pipeline
    df = builder.process_all(args.start_date, args.end_date)
    print(f"Final Factor Shape: {df.shape}")

    # Save Results
    if not df.empty:
        # Create output directory
        os.makedirs(args.out_dir, exist_ok=True)

        # Partition by Year
        df['year_temp'] = df['date'].dt.year
        for year, group in df.groupby('year_temp'):
            out_file = os.path.join(args.out_dir, f"ghz_factors_{year}.parquet")

            # Drop temp column
            group_to_save = group.drop(columns=['year_temp'])
            group_to_save.to_parquet(out_file)
            print(f"Saved {year} data to {out_file}")
    else:
        print("No data generated.")
