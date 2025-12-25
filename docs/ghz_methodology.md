# GHZ (Green, Hand, Zhang 2017) Factor Methodology

This document details the implementation logic for the characteristics (factors) described in Green, Hand, & Zhang (2017), as implemented in `src/data/ghz_factors.py`.

## 1. Data Processing Logic

### 1.1 Compustat Annual (`comp.funda`)
*   **Sorting**: Data is sorted by `gvkey` (Firm ID) and `datadate` (Fiscal Year End).
*   **Missing Value Handling**:
    *   `xint` (Interest Expense): Filled with 0 if missing (`fillna(0)`).
    *   `xsga` (SG&A Expense): Filled with 0 if missing.
*   **Fiscal Market Value (`mve_f`)**:
    *   `mve_f = csho * abs(prcc_f)`
    *   Used as the scaling denominator for many valuation ratios.

#### Variable Lags
We calculate lags within each `gvkey` group using `shift(1)`.
*   `lag_at` = Total Assets at $t-1$
*   `avg_at` = $(at_t + at_{t-1}) / 2$

### 1.2 Compustat Quarterly (`comp.fundq`)
*   **Lags**: `shift(4)` is used to compare Quarter $t$ with Quarter $t-4$ (Same Quarter Last Year) for seasonality.
*   **Formulas**:
    *   **roaq**: `ibq / lag_atq` (Note: `lag_atq` is `shift(1)` here, immediate previous quarter).
    *   **sue** (Standardized Unexpected Earnings): `(ibq - ibq_{t-4}) / mveq` (Simplified Compustat-only fallback).
    *   **roavol**: Rolling 16-quarter standard deviation of `roaq`.

### 1.3 CRSP Monthly (`crsp.msf`)
GHZ 2017 includes price-based factors (Momentum, Volatility) alongside fundamentals.
*   **Log Returns**: `log_ret = ln(1 + ret)`
*   **mom12m** (12-Month Momentum):
    *   Sum of `log_ret` from $t-12$ to $t-2$.
    *   Skip month $t-1$ (Standard formation gap).
*   **mom1m** (Short-Term Reversal): Return at $t-1$.
*   **retvol** (Volatility): Proxy using 36-month standard deviation of returns.

## 2. Merging Logic (The "Valid From" Strategy)

We align low-frequency fundamental data (Annual/Quarterly) with monthly market returns using a conservative availability lag to avoid look-ahead bias.

### 2.1 Annual to Monthly
*   **Availability Assumption**: Annual data for fiscal year ending at date $D$ is assumed to be available to investors 6 months after $D$.
*   **Logic**:
    1.  Create `valid_from = datadate + 6 months`.
    2.  For a given monthly return at time $t$, we use the most recent Annual record where `valid_from <= t`.
    3.  **Implementation**: `pd.merge_asof` with `direction='backward'` and a tolerance (e.g., 12-18 months) to ensure we don't use stale data (>1.5 years old).

## 3. Why Momentum in GHZ?

Although GHZ (Green, Hand, Zhang) is famous for categorizing "Firm Characteristics", their 2017 paper ("The Characteristics that Provide Independent Information about Average Returns") serves as a comprehensive "Factor Zoo" benchmark. They evaluate **94 distinct characteristics**, including Momentum, Volatility, and Trading Frictions.

## 4. Implemented Factors List

The following **100+ factors** are currently implemented in `ghz_factors.py`.

### 4.1 Valuation
| Factor | Formula (Simplified) | Source |
| :--- | :--- | :--- |
| `bm` | `ceq / mve_f` | Annual |
| `ep` | `ib / mve_f` | Annual |
| `cashpr` | `(mve_f + dltt - at) / che` | Annual |
| `dy` | `dvt / mve_f` | Annual |
| `lev` | `lt / mve_f` | Annual |
| `sp` | `sale / mve_f` | Annual |
| `cfp` | `oancf / mve_f` | Annual |
| `rd_mve` | `xrd / mve_f` | Annual |
| `tb_1` | `((txfo + txfed) / 0.35) / ib` | Annual |
| `realestate` | `(fatb + fatl) / ppent` | Annual |
| `sin` | Dummy for Sin Industry (Tobacco, Alcohol, Gaming) | Annual |
| `secured` | `dm / dltt` | Annual |
| `conv` | `dcvt / dltt` | Annual |

### 4.2 Profitability & Efficiency
| Factor | Formula (Simplified) | Source |
| :--- | :--- | :--- |
| `roic` | `(ebit - nopi) / (ceq + lt - che)` | Annual |
| `roaq` | `ibq / lag_atq` | Quarterly |
| `roe` | `ib / lag_ceq` | Annual |
| `roa` | `ni / avg_at` | Annual |
| `cfroa` | `oancf / avg_at` | Annual |
| `operprof` | `(revt - cogs - xsga - xint) / lag_ceq` | Annual |
| `chpm` | `(ib/sale) - lag(ib/sale)` | Annual |
| `chato` | `(sale/avg_at) - lag(sale/avg_at)` | Annual |
| `currat` | `act / lct` | Annual |
| `pchcurrat` | `% chg in currat` | Annual |
| `quick` | `(act - invt) / lct` | Annual |
| `pchquick` | `% chg in quick` | Annual |
| `salecash` | `sale / che` | Annual |
| `salerec` | `sale / rect` | Annual |
| `saleinv` | `sale / invt` | Annual |
| `pchsaleinv` | `chg(sale/invt)` | Annual |
| `cashdebt` | `(ib + dp) / avg(lt)` | Annual |
| `tang` | Tangibility (weighted asset mix) | Annual |
| `ps` | Piotroski F-Score (0-9) | Annual |

### 4.3 Investment & Growth
| Factor | Formula (Simplified) | Source |
| :--- | :--- | :--- |
| `agr` | `(at / lag_at) - 1` | Annual |
| `gma` | `(revt - cogs) / lag_at` | Annual |
| `chcsho` | `(csho / lag_csho) - 1` | Annual |
| `lgr` | `(lt / lag_lt) - 1` | Annual |
| `sgr` | `(sale / lag_sale) - 1` | Annual |
| `hire` | `(emp - lag_emp) / lag_emp` | Annual |
| `invest` | `(delta(ppegt) + delta(invt)) / lag_at` | Annual |
| `cinvest` | Standardized Quarter-over-Quarter Investment | Quarterly |
| `pchcapx` | `% chg in capx` | Annual |
| `grcapx` | 2-year growth in capx | Annual |
| `grgw` | Growth in Goodwill | Annual |
| `chinv` | `delta(invt) / avg_at` | Annual |
| `pchsale_pchinvt` | Sales growth - Inventory growth | Annual |
| `pchsale_pchrect` | Sales growth - Receivables growth | Annual |
| `pchgm_pchsale` | Gross Margin growth - Sales growth | Annual |
| `pchsale_pchxsga` | Sales growth - SG&A growth | Annual |
| `rd` | Dummy: R&D increased > 5% | Annual |
| `rdbias` | R&D Growth Bias | Annual |
| `xrdint` | `xrd / avg_at` | Annual |
| `capxint` | `capx / avg_at` | Annual |
| `xadint` | `xad / avg_at` | Annual |
| `chadv` | `log(1+xad) - log(1+lag_xad)` | Annual |

### 4.4 Earnings Quality & Accruals
| Factor | Formula (Simplified) | Source |
| :--- | :--- | :--- |
| `acc` | `(ib - oancf) / avg_at` | Annual |
| `pctacc` | `acc / abs(ib)` | Annual |
| `absacc` | `abs(acc)` | Annual |
| `sue` | `(ibq - ibq_lag4) / mveq` | Quarterly |
| `rsup` | Revenue Surprise | Quarterly |
| `sacc` | Sales Accruals | Quarterly |
| `scf` | Sales Cash Flow | Quarterly |
| `spii` | Dummy: Special Items exists | Annual |
| `spi` | `spi / avg_at` | Annual |
| `cf` | `oancf / avg_at` | Annual |
| `obklg` | Output backlog / assets | Annual |
| `chobklg` | Change in backlog | Annual |
| `grltnoa` | Growth in Long Term Net Operating Assets | Annual |
| `chdrc` | Change in Deferred Revenue | Annual |
| `chtx` | Change in Tax Expense | Quarterly |

### 4.5 Momentum & Returns
| Factor | Formula (Simplified) | Source |
| :--- | :--- | :--- |
| `mom12m` | `exp(sum(log(1+ret)) t-12 to t-2) - 1` | Monthly |
| `mom6m` | `exp(sum(log(1+ret)) t-6 to t-2) - 1` | Monthly |
| `mom36m` | `exp(sum(log(1+ret)) t-36 to t-13) - 1` | Monthly |
| `mom1m` | `ret` (t-1) | Monthly |
| `chmom` | `mom6m - mom6m_lag6` | Monthly |
| `retcons` | Consistent Pos/Neg Returns (dummy) | Monthly |
| `aeavol` | Earnings Announcement Volume (Placeholder) | Quarterly |
| `ear` | Earnings Announcement Return (Placeholder) | Quarterly |
| `nincr` | Number of consecutive earnings increases | Quarterly |

### 4.6 Risk & Volatility
| Factor | Formula (Simplified) | Source |
| :--- | :--- | :--- |
| `retvol` | `std(ret, 36m)` | Monthly |
| `roavol` | `std(roaq, 16q)` | Quarterly |
| `sgrvol` | `std(rsup, 15q)` | Quarterly |
| `stdacc` | `std(sacc, 16q)` | Quarterly |
| `stdcf` | `std(scf, 16q)` | Quarterly |
| `beta` | Market Beta (Placeholder) | Monthly |
| `idiovol` | Idiosyncratic Vol (Placeholder) | Monthly |

### 4.7 Liquidity & Trading
| Factor | Formula (Simplified) | Source |
| :--- | :--- | :--- |
| `mve` | `log(close * shrout)` | Monthly |
| `turn` | `mean(vol_lag1..3) / shrout` | Monthly |
| `dolvol` | `log(vol * price)` | Monthly |
| `ipo` | Dummy: Public < 12 months | Monthly |
| `divi` | Dummy: Dividend Initiation | Annual |
| `divo` | Dummy: Dividend Omission | Annual |
| `wogw` | Dummy: Write-off Goodwill | Annual |

### 4.8 Industry Adjustments
For most Annual factors, an Industry-Adjusted version (`{factor}_ia`) is also calculated:
`X_ia = X - mean(X)` grouped by `fyear` and `SIC2` (2-digit Industry Code).

### 4.9 Advanced Daily Factors (High-Frequency Microstructure)
Implemented in `src/data/daily_factors.py` using strictly `crsp.dsf`. These capture higher-order dynamics beyond simple monthly returns.

| Factor | Formula/Logic (Simplified) | Concept |
| :--- | :--- | :--- |
| `amihud_1m` | `Mean(|Ret| / (Price * Vol))` | Liquidity (Price Impact) |
| `turnover_var` | `Std(Vol / Shrout)` | Liquidity Instability |
| `parkinson_vol` | `Mean( (ln(High/Low))^2 )` | Range-Based Volatility |
| `downside_vol` | `Sqrt(Mean(Ret^2 | Ret < 0))` | Downside Risk |
| `upside_vol` | `Sqrt(Mean(Ret^2 | Ret > 0))` | Upside Potential |
| `max_ret_1m` | `Max(Daily Ret)` | Lottery Demand |
| `skew_1m` | `Skew(Daily Ret)` | Tail Risk / Asymmetry |
| `ret_gap_vol` | `Std(Open / Prev_Close - 1)` | Overnight Risk |
| `clv_mean` | `Mean((Close-Low)/(High-Low))` | Buying Pressure (Close Location Value) |
| `dvol_cv` | `Std(DollarVol) / Mean(DollarVol)` | Liquidity Variation |
| `zero_ret_pct` | `% of Days where Ret=0` | Trading Activity / Illiquidity |

---

## Appendix A: Raw Variable Definitions

The following raw variables are pulled from WRDS (`comp.funda`, `comp.fundq`, `crsp.msf`) to construct the factors.

### A.1 Identifiers & Dates
*   **gvkey**: Firm ID (Compustat)
*   **permno**: Security ID (CRSP)
*   **cusip**: Security ID (CUSIP Committee)
*   **cik**: Central Index Key (SEC)
*   **sic**, **naics**: Standard Industrial Classification, North American Industry Classification
*   **datadate**, **date**: Report Date / Trade Date
*   **fyear**, **fyearq**: Fiscal Year
*   **fqtr**: Fiscal Quarter
*   **rdq**: Earnings Report Date

### A.2 Income Statement (Annual & Quarterly)
*   **sale**, **saleq**, **revt**, **revtq**: Sales / Revenue (Net)
*   **cogs**, **cogsq**: Cost of Goods Sold
*   **xsga**, **xsgaq**: Selling, General, and Administrative Expenses
*   **xint**: Interest and Related Expense
*   **dp**: Depreciation and Amortization
*   **xrd**: Research and Development Expense
*   **xad**: Advertising Expense
*   **ib**, **ibq**: Income Before Extraordinary Items
*   **ni**: Net Income (Loss)
*   **ebit**: Earnings Before Interest and Taxes
*   **ebitda**: Earnings Before Interest, Taxes, Depreciation and Amortization
*   **nopi**: Non-Operating Income (Expense)
*   **spi**: Special Items
*   **pi**: Pretax Income
*   **txp**: Income Taxes - Payable
*   **txt**, **txtq**: Income Taxes - Total
*   **txfed**: Income Taxes - Federal
*   **txfo**: Income Taxes - Foreign
*   **txdi**: Income Taxes - Deferred
*   **dvt**: Dividends - Total

### A.3 Balance Sheet (Annual & Quarterly)
*   **at**, **atq**: Assets - Total
*   **act**, **actq**: Current Assets - Total
*   **che**, **cheq**: Cash and Short-Term Investments
*   **rect**: Receivables - Total
*   **invt**: Inventories - Total
*   **aco**: Current Assets - Other
*   **intan**: Intangible Assets - Total
*   **gdwl**: Goodwill
*   **ao**: Assets - Other
*   **ppent**, **ppentq**: Property, Plant and Equipment - Total (Net)
*   **ppegt**: Property, Plant and Equipment - Total (Gross)
*   **fatb**: PPE - Buildings (at Cost)
*   **fatl**: PPE - Land and Improvements (at Cost)
*   **lt**, **ltq**: Liabilities - Total
*   **lct**, **lctq**: Current Liabilities - Total
*   **ap**: Accounts Payable - Trade
*   **dlc**, **dlcq**: Debt in Current Liabilities
*   **dltt**: Long-Term Debt - Total
*   **dm**: Debt - Mortgages and Other Secured
*   **dcvt**: Debt - Convertible
*   **lco**: Current Liabilities - Other
*   **lo**: Liabilities - Other
*   **drc**: Deferred Revenue - Current
*   **drlt**: Deferred Revenue - Long-Term
*   **ceq**, **ceqq**: Common/Ordinary Equity - Total
*   **seqq**: Stockholders' Equity - Parent
*   **pstk**, **pstkq**: Preferred/Preference Stock (Capital) - Total
*   **pstkrq**: Preferred/Preference Stock - Redeemable
*   **csho**, **cshoq**: Common Shares Outstanding
*   **emp**: Employees
*   **ob**: Order Backlog
*   **gdwlia**, **gdwlip**, **gwo**: Impairment of Goodwill (Pre-tax / After-tax / Write-off)

### A.4 Market Data (CRSP MSF/DSF & Compustat Values)
*   **prc**, **prccq**, **prcc_f**: Price - Close
*   **ret**: Holding Period Return
*   **retx**: Return without Dividends (Capital Gains only)
*   **vol**: Volume
*   **numtrd**: Number of Trades (Daily)
*   **shrout**: Shares Outstanding
*   **cfacpr**: Cumulative Factor to Adjust Price (Splits/Distributions)
*   **cfacshr**: Cumulative Factor to Adjust Shares
*   **dcpstk**: Convertible Debt and Preferred Stock

