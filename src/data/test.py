import wrds
db = wrds.Connection(wrds_username='jliu3074')
trading_days = db.get_table(library='crsp', table='dsi', columns=['date'])
print(trading_days)
import pdb; pdb.set_trace()
vars_funda = db.describe_table(library='comp', table='funda')['name'].tolist()
# print(vars_funda)
print([target for target in ['permno','gvkey', 'datadate', 'fyear', 'cik', 'sic', 'naics','sale', 'revt',
'cogs', 'xsga', 'dp', 'xrd', 'xad', 'ib', 'ebitda', 'ebit', 'nopi', 'spi',
'pi', 'txp', 'ni','txfed', 'txfo', 'txt', 'xint', 'capx', 'oancf', 'dvt',
'ob', 'gdwlia', 'gdwlip', 'gwo','rect', 'act', 'che', 'ppegt', 'invt', 'at',
'aco', 'intan', 'ao', 'ppent', 'gdwl', 'fatb', 'fatl','lct', 'dlc', 'dltt',
'lt', 'dm', 'dcvt', 'cshrc', 'dcpstk', 'pstk', 'ap', 'lco', 'lo', 'drc',
'drlt', 'txdi', 'ceq', 'scstkc', 'emp', 'csho', 'prcc_f'] if target not in vars_funda])

vars_fundq = db.describe_table(library='comp', table='fundq')['name'].tolist()
print([target for target in ['permno','gvkey', 'datadate', 'fyearq', 'fqtr', 'rdq',
            'ibq', 'saleq', 'txtq', 'revtq', 'cogsq', 'xsgaq',
            'atq', 'actq', 'cheq', 'lctq', 'dlcq', 'ppentq',
            'prccq', 'cshoq', 'ceqq', 'seqq', 'pstkq', 'ltq', 'pstkrq'] if target not in vars_fundq])


vars_month = db.describe_table(library='crsp', table='msf')['name'].tolist()
print([target for target in ['permno', 'date', 'prc', 'ret', 'vol', 'shrout', 'cfacpr', 'cfacshr'] if target not in vars_month])


vars_link = db.describe_table(library='crsp', table='ccmxpf_linktable')['name'].tolist()
print([target for target in ['gvkey', 'lpermno', 'linktype', 'linkprim', 'linkdt', 'linkenddt'] if target not in vars_link])

# import pdb; pdb.set_trace()
start_date = '2020-01-01'
q_msf = f"""
    SELECT
        *
    FROM crsp.dsf
    WHERE date >= '{start_date}' limit 10
"""
df_msf = db.raw_sql(q_msf)
print(df_msf)
print(df_msf.columns)

# q_link = """
#     SELECT
#         gvkey, lpermno as permno, linktype, linkprim, linkdt, linkenddt
#     FROM crsp.ccmxpf_linktable
#     WHERE linktype IN ('LU', 'LC') limit 10
# """
# df_link = db.raw_sql(q_link)
# print(df_link)

print("\n--- Checking Stocknames Permissions (Fallback Link) ---")
q_names = """
    SELECT *
    FROM crsp.stocknames
    LIMIT 10
"""
q_names = """
    SELECT *
    FROM comp.co_filedate
    LIMIT 10
"""
try:
    df_names = db.raw_sql(q_names)
    print("Success! crsp.stocknames is accessible.")
    print(df_names)
    import pdb; pdb.set_trace()
except Exception as e:
    print(f"Failed to access comp.co_filedate: {e}")

# df_msf.to_parquet(os.path.join(data_root, 'crsp_msf.parquet'))
# print(f"Saved crsp_msf.parquet: {df_msf.shape}")

print("\n--- Checking Date Ranges ---")
q_date_msf = "SELECT min(date) as min_date, max(date) as max_date FROM crsp.msf"
print("Result for crsp.msf:")
print(db.raw_sql(q_date_msf))

q_date_dsf = "SELECT min(date) as min_date, max(date) as max_date FROM crsp.dsf"
print("Result for crsp.dsf:")
print(db.raw_sql(q_date_dsf))