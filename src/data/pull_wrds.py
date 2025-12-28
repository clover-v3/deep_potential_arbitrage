"""
WRDS Data Downloader (Modular)
Pulls required format from WRDS (Compustat, CRSP, IBES).
Requires ~/.pgpass or interactive login.

Modes:
- low_freq: Pulls Compustat Annual/Quarterly and CRSP Monthly.
- daily: Pulls CRSP Daily (High volume).
- mapping: Pulls CRSP Stocknames (if missing).
- all: Runs all of the above.
"""

import calendar
import wrds
import pandas as pd
import os
import argparse
import datetime
import sys

class WRDSDataPuller:
    def __init__(self, data_root: str, start_date: str = '2020-01-01', end_date: str = None, username: str = 'jliu3074'):
        self.data_root = data_root
        os.makedirs(self.data_root, exist_ok=True)

        if end_date is None:
            self.end_date_dt = pd.Timestamp.now()
        else:
            self.end_date_dt = pd.to_datetime(end_date)

        self.start_date_dt = pd.to_datetime(start_date)
        self.username = username
        self.db = None

        # Calculate Date Cutoffs
        self._calculate_cutoffs()

    def connect(self):
        print(f"Connecting to WRDS as {self.username}...")
        try:
             self.db = wrds.Connection(wrds_username=self.username)
             print("Connected.")
        except Exception as e:
             print(f"Connection Failed: {e}")
             sys.exit(1)

    def close(self):
        if self.db:
            self.db.close()

    def _calculate_cutoffs(self):
        end_dt = self.end_date_dt

        # Annual: Last complete year
        if end_dt.month == 12 and end_dt.day == 31:
            self.last_complete_year = end_dt.year
        else:
            self.last_complete_year = end_dt.year - 1

        # Quarterly: Last complete quarter end
        curr_q_start_month = ((end_dt.month - 1) // 3) * 3 + 1
        curr_q_start = pd.Timestamp(year=end_dt.year, month=curr_q_start_month, day=1)
        self.last_complete_q_date = curr_q_start - pd.Timedelta(days=1)

        # Monthly: Last complete month end
        if end_dt.day == end_dt.days_in_month:
             self.last_complete_m_date = end_dt
        else:
             # First day of current month -> Last day of prev month
             curr_m_start = pd.Timestamp(year=end_dt.year, month=end_dt.month, day=1)
             self.last_complete_m_date = curr_m_start - pd.Timedelta(days=1)

        print(f"Cutoffs: Annual <= {self.last_complete_year}, Quarterly <= {self.last_complete_q_date.date()}, Monthly <= {self.last_complete_m_date.date()}")

    def _process_stats(self, df):
        return f"Shape: {df.shape}"

    def pull_low_freq(self):
        """Pulls Annual, Quarterly, and Monthly Data."""
        if not self.db: self.connect()
        years = range(self.start_date_dt.year, self.end_date_dt.year + 1)

        # Batching Config for Permno/GVKEY
        PERMNO_START = 0
        PERMNO_END = 100000
        PERMNO_BATCH = 10000

        GVKEY_START = 0
        GVKEY_END = 350000 # Most strings are within '000000' to '350000'
        GVKEY_BATCH = 25000 # String comparison

        # 1. Compustat Annual
        print("\n[1/4] Pulling Compustat Annual (comp.funda)...")
        save_path_funda = os.path.join(self.data_root, 'comp_funda')
        os.makedirs(save_path_funda, exist_ok=True)

        for year in years:
            if year > self.last_complete_year: continue

            out_file = os.path.join(save_path_funda, f"funda_{year}.parquet")
            if os.path.exists(out_file):
                print(f"  Skipping {year} (Exists)")
                continue

            print(f"  > Processing Year {year}...")
            chunk_dfs = []

            for g_start in range(GVKEY_START, GVKEY_END, GVKEY_BATCH):
                g_end = g_start + GVKEY_BATCH - 1
                g_start_str = f"{g_start:06d}"
                g_end_str = f"{g_end:06d}"

                print(f"    - GVKEY {g_start_str}-{g_end_str}...", end=" ", flush=True)

                q_funda = f"""
                    SELECT
                        f.gvkey, f.datadate, f.fyear, f.cik, f.cusip, c.sic, c.naics,
                        f.sale, f.revt, f.cogs, f.xsga, f.dp, f.xrd, f.xad, f.ib, f.ebitda, f.ebit, f.nopi, f.spi, f.pi, f.txp, f.ni,
                        f.txfed, f.txfo, f.txt, f.xint, f.capx, f.oancf, f.dvt, f.ob, f.gdwlia, f.gdwlip, f.gwo,
                        f.rect, f.act, f.che, f.ppegt, f.invt, f.at, f.aco, f.intan, f.ao, f.ppent, f.gdwl, f.fatb, f.fatl,
                        f.lct, f.dlc, f.dltt, f.lt, f.dm, f.dcvt, f.cshrc, f.dcpstk, f.pstk, f.ap, f.lco, f.lo, f.drc, f.drlt, f.txdi,
                        f.ceq, f.scstkc, f.emp, f.csho, f.prcc_f
                    FROM comp.funda f
                    LEFT JOIN comp.company c ON f.gvkey = c.gvkey
                    WHERE f.datadate BETWEEN '{year}-01-01' AND '{year}-12-31'
                    AND f.indfmt='INDL' AND f.datafmt='STD' AND f.popsrc='D' AND f.consol='C'
                    AND f.gvkey BETWEEN '{g_start_str}' AND '{g_end_str}'
                """
                try:
                    df = self.db.raw_sql(q_funda)
                    if not df.empty:
                        chunk_dfs.append(df)
                        print(f"Got {len(df)} rows.")
                    else:
                        print("Empty.")
                except Exception as e:
                    print(f"Failed: {e}")

            if chunk_dfs:
                pd.concat(chunk_dfs, ignore_index=True).to_parquet(out_file)
                print(f"  >> Saved {out_file}")

        # 1.5. Compustat File Dates
        print("\n[1.5/4] Pulling Compustat File Dates (comp.co_filedate)...")
        save_path_filedate = os.path.join(self.data_root, 'comp_co_filedate')
        os.makedirs(save_path_filedate, exist_ok=True)

        for year in years:
            if year > self.last_complete_year: continue

            out_file = os.path.join(save_path_filedate, f"co_filedate_{year}.parquet")
            if os.path.exists(out_file):
                 print(f"  Skipping {year} (Exists)")
                 continue

            print(f"  > Processing Year {year} (File Dates)...")
            chunk_dfs = []

            for g_start in range(GVKEY_START, GVKEY_END, GVKEY_BATCH):
                g_end = g_start + GVKEY_BATCH - 1
                g_start_str = f"{g_start:06d}"
                g_end_str = f"{g_end:06d}"

                # print(f"    - GVKEY {g_start_str}-{g_end_str}...", end=" ", flush=True)

                q_filedate = f"""
                    SELECT gvkey, datadate, filedate, srctype
                    FROM comp.co_filedate
                    WHERE datadate BETWEEN '{year}-01-01' AND '{year}-12-31'
                    AND gvkey BETWEEN '{g_start_str}' AND '{g_end_str}'
                """
                try:
                    df = self.db.raw_sql(q_filedate)
                    if not df.empty:
                        chunk_dfs.append(df)
                        # print(f"Got {len(df)} rows.")
                except Exception as e:
                    print(f"Failed: {e}")

            if chunk_dfs:
                pd.concat(chunk_dfs, ignore_index=True).to_parquet(out_file)
                print(f"  >> Saved {out_file}")

        # 2. Compustat Quarterly
        print("\n[2/4] Pulling Compustat Quarterly (comp.fundq)...")
        save_path_fundq = os.path.join(self.data_root, 'comp_fundq')
        os.makedirs(save_path_fundq, exist_ok=True)

        for year in years:
            if year > self.last_complete_q_date.year: continue

            out_file = os.path.join(save_path_fundq, f"fundq_{year}.parquet")
            if os.path.exists(out_file):
                 print(f"  Skipping {year} (Exists)")
                 continue

            print(f"  > Processing Year {year} (Quarterly)...")
            chunk_dfs = []

            date_filter_end = f"{year}-12-31"
            if year == self.last_complete_q_date.year:
                date_filter_end = self.last_complete_q_date.strftime("%Y-%m-%d")

            for g_start in range(GVKEY_START, GVKEY_END, GVKEY_BATCH):
                g_end = g_start + GVKEY_BATCH - 1
                g_start_str = f"{g_start:06d}"
                g_end_str = f"{g_end:06d}"

                print(f"    - GVKEY {g_start_str}-{g_end_str}...", end=" ", flush=True)

                q_fundq = f"""
                    SELECT
                        f.gvkey, f.datadate, f.fyearq, f.fqtr, f.rdq, f.cusip, c.sic,
                        f.ibq, f.saleq, f.txtq, f.revtq, f.cogsq, f.xsgaq,
                        f.atq, f.actq, f.cheq, f.lctq, f.dlcq, f.ppentq,
                        f.prccq, f.cshoq, f.ceqq, f.seqq, f.pstkq, f.ltq, f.pstkrq
                    FROM comp.fundq f
                    LEFT JOIN comp.company c ON f.gvkey = c.gvkey
                    WHERE f.datadate BETWEEN '{year}-01-01' AND '{date_filter_end}'
                    AND f.indfmt='INDL' AND f.datafmt='STD' AND f.popsrc='D' AND f.consol='C'
                    AND f.gvkey BETWEEN '{g_start_str}' AND '{g_end_str}'
                """
                try:
                    df = self.db.raw_sql(q_fundq)
                    if not df.empty:
                        chunk_dfs.append(df)
                        print(f"Got {len(df)} rows.")
                    else:
                        print("Empty.")
                except Exception as e:
                    print(f"Failed: {e}")

            if chunk_dfs:
                pd.concat(chunk_dfs, ignore_index=True).to_parquet(out_file)
                print(f"  >> Saved {out_file}")

        # 3. CRSP Monthly
        print("\n[3/4] Pulling CRSP Monthly (crsp.msf)...")
        save_path_msf = os.path.join(self.data_root, 'crsp_msf')
        os.makedirs(save_path_msf, exist_ok=True)

        for year in years:
            if year > self.last_complete_m_date.year: continue

            out_file = os.path.join(save_path_msf, f"msf_{year}.parquet")
            if os.path.exists(out_file):
                  print(f"  Skipping {year} (Exists)")
                  continue

            print(f"  > Processing Year {year} (Monthly)...")
            chunk_dfs = []

            date_filter_end = f"{year}-12-31"
            if year == self.last_complete_m_date.year:
                date_filter_end = self.last_complete_m_date.strftime("%Y-%m-%d")

            for p_start in range(PERMNO_START, PERMNO_END, PERMNO_BATCH):
                p_end = p_start + PERMNO_BATCH - 1

                print(f"    - Permnos {p_start}-{p_end}...", end=" ", flush=True)

                q_msf = f"""
                    SELECT
                        permno, date, abs(prc) as prc, ret, vol, shrout
                    FROM crsp.msf
                    WHERE date BETWEEN '{year}-01-01' AND '{date_filter_end}'
                    AND permno BETWEEN {p_start} AND {p_end}
                """
                try:
                    df = self.db.raw_sql(q_msf)
                    if not df.empty:
                        chunk_dfs.append(df)
                        print(f"Got {len(df)} rows.")
                    else:
                        print("Empty.")
                except Exception as e:
                    print(f"Failed: {e}")

            if chunk_dfs:
                pd.concat(chunk_dfs, ignore_index=True).to_parquet(out_file)
                print(f"  >> Saved {out_file}")

    def pull_daily(self):
        """Pulls CRSP Daily Data."""
        if not self.db: self.connect()

        print("\n[4/4] Pulling CRSP Daily (crsp.dsf)...")
        save_path_dsf = os.path.join(self.data_root, 'crsp_dsf')
        os.makedirs(save_path_dsf, exist_ok=True)

        current_month = self.start_date_dt.replace(day=1)

        # Batching Config
        PERMNO_START = 0  # Updated to 0
        PERMNO_END = 100000 # Updated to 100000
        BATCH_SIZE = 5000

        while current_month <= self.last_complete_m_date:
            year = current_month.year
            month = current_month.month

            next_month = current_month + pd.DateOffset(months=1)
            m_end_dt = next_month - pd.Timedelta(days=1)

            out_file = os.path.join(save_path_dsf, f"dsf_{year}_{month:02d}.parquet")

            if os.path.exists(out_file):
                # print(f"  Skipping {year}-{month:02d} (Exists)")
                current_month = next_month
                continue

            print(f"  > Processing {year}-{month:02d} (Daily)...")

            chunk_dfs = []

            # Iterate by Permno Batch
            for p_start in range(PERMNO_START, PERMNO_END + 1, BATCH_SIZE):
                p_end = p_start + BATCH_SIZE - 1

                print(f"    - Permnos {p_start}-{p_end}...", end=" ", flush=True)

                q_dsf = f"""
                    SELECT
                        permno, date, abs(prc) as prc, openprc, askhi, bidlo, ret, vol, shrout, cfacpr, cfacshr, numtrd, retx
                    FROM crsp.dsf
                    WHERE date BETWEEN '{year}-{month:02d}-01' AND '{m_end_dt.strftime('%Y-%m-%d')}'
                    AND permno BETWEEN {p_start} AND {p_end}
                """
                try:
                    df = self.db.raw_sql(q_dsf)
                    if not df.empty:
                        chunk_dfs.append(df)
                        print(f"Got {len(df)} rows.")
                    else:
                        print("Empty.")
                except Exception as e:
                    print(f"Failed: {e}")

            # Combine and Save
            if chunk_dfs:
                full_df = pd.concat(chunk_dfs, ignore_index=True)
                full_df.to_parquet(out_file)
                print(f"  >> Saved {out_file} ({full_df.shape[0]} rows)")
            else:
                print(f"  >> No data for {year}-{month:02d}")

            current_month = next_month

    def pull_mapping(self):
        """Checks for crsp.stocknames, pulls if missing."""
        mapping_path = os.path.join(self.data_root, 'crsp_stocknames.parquet')

        # if os.path.exists(mapping_path):
        #     print(f"\n[Mapping] Found existing {mapping_path}. Skipping.")
        #     return

        if not self.db: self.connect()
        print("\n[Mapping] Pulling CRSP Stocknames...")

        q_names = """
            SELECT permno, permco, ncusip, namedt, nameenddt, ticker, siccd, shrcd, exchcd
            FROM crsp.stocknames
        """
        try:
            df = self.db.raw_sql(q_names)
            if not df.empty:
                df.to_parquet(mapping_path)
            print(f"Done. {self._process_stats(df)}")
        except Exception as e:
            print(f"Failed to pull mapping: {e}")

    def pull_calendar(self):
        """Checks for crsp.dsi, pulls if missing."""
        calendar_path = os.path.join(self.data_root, 'crsp_trading_days.parquet')

        # if os.path.exists(calendar_path):
        #     print(f"\n[Calendar] Found existing {calendar_path}. Skipping.")
        #     return

        if not self.db: self.connect()
        print("\n[Calendar] Pulling CRSP Calendar...")


        try:
            df = self.db.get_table(library='crsp', table='dsi', columns=['date'])
            if not df.empty:
                df.to_parquet(calendar_path)
            print(f"Done. {self._process_stats(df)}")
        except Exception as e:
            print(f"Failed to pull calendar: {e}")

def main():
    parser = argparse.ArgumentParser(description="WRDS Data Downloader (Modular)")
    parser.add_argument("--save_dir", type=str, default="./data/raw_ghz")
    parser.add_argument("--start_date", type=str, default='2020-01-01', help="YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--mode", type=str, default='all', choices=['all', 'low_freq', 'daily', 'mapping'],
                        help="Which data to pull")

    args = parser.parse_args()

    puller = WRDSDataPuller(args.save_dir, args.start_date, args.end_date)

    if args.mode in ['all', 'mapping']:
        puller.pull_mapping()
        puller.pull_calendar()

    if args.mode in ['all', 'low_freq']:
        puller.pull_low_freq()

    if args.mode in ['all', 'daily']:
        puller.pull_daily()

    puller.close()

if __name__ == "__main__":
    main()
