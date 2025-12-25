# Daily Factors Documentation

**Module:** `src/data/daily_factors.py`
**Data Source:** CRSP Daily Stock File (DSF) only.
**Output Frequency:** Daily.

## 1. Methodology Overview

This module generates advanced statistical and microstructure features solely from daily high-frequency data (Open, High, Low, Close, Volume).

### 1.1 Timing & Availability
*   **Observation Time:** All factors for date $T$ calculated using data up to and including Market Close on date $T$.
*   **Usage:** These factors represent the state of the asset **at the end of day $T$**. In a predictive model for returns at $T+1$, these values are available and valid inputs.
*   **Look-ahead Bias:** None (assuming trading occurs at or after Close $T$). No future data ($T+1$) is used.

### 1.2 Data Handling
*   **Rolling Windows:** Standard window is **21 trading days** (approx. 1 month).
*   **Minimum Support:** Requires at least 15 valid observations within the window to produce a value (otherwise NaN).
*   **Handling Day 0:** The rolling calculation naturally results in `NaN` for the first 14 days of the *entire available dataset* for a given stock. The code assumes the input DataFrame contains sufficient history.

---

## 2. Factor Definitions

| Symbol | Factor Name | Formula / Logic | Financial Interpretation |
| :--- | :--- | :--- | :--- |
| **Liquidity** | | | |
| `amihud_1m` | 1M Amihud Illiquidity | Mean($\frac{\|R_t\|}{P_t \cdot V_t}$) | Price impact per dollar traded. Higher = Less Liquid. |
| `turnover_var` | Turnover Instability | Std($\frac{V_t}{Shares_t}$) | Volatility of trading interest. |
| `dvol_cv_1m` | Dollar Vol CV | $\frac{\text{Std}(P_t V_t)}{\text{Mean}(P_t V_t)}$ | Reliability of liquidity. High CV = liquidity dries up often. |
| `zero_ret_pct_1m` | Zero Return Days | % of days where $R_t = 0$ | Proxy for illiquidity (informed traders inactive). |
| **Volatility** | | | |
| `parkinson_vol_1m` | Parkinson Volatility | $\sqrt{\text{Mean}(\frac{1}{4\ln 2} (\ln \frac{H_t}{L_t})^2)}$ | Range-based volatility estimator (more efficient than Close-to-Close). |
| `downside_vol_1m` | Downside Volatility | $\sqrt{\text{Mean}(R_t^2 \mid R_t < 0)}$ | Risk of loss (Semi-variance). |
| `upside_vol_1m` | Upside Variability | $\sqrt{\text{Mean}(R_t^2 \mid R_t > 0)}$ | Upside potential variability (Good variance). |
| `ret_gap_vol_1m` | Overnight Gap Vol | Std($\frac{O_t}{C_{t-1}} - 1$) | Risk occurring outside trading hours. |
| **Distribution** | | | |
| `max_ret_1m` | Max Daily Return | Max($R_t$) over 21 days | Proxy for "Lottery Demand" (investors overpay for max potential). |
| `skew_1m` | Realized Skewness | Skew($R_t$) over 21 days | Asymmetry risk. Negative skew = Crash risk. |
| **Microstructure** | | | |
| `clv_mean_1m` | Close Location Value | Mean($\frac{C_t - L_t}{H_t - L_t}$) | 1 = Close at High (Accumulation), 0 = Close at Low (Distribution). |

---

## 3. Input / Output Specification

### Input
*   **Source:** Parquet files in `data/raw_ghz/crsp_dsf`.
*   **Columns Required:** `permno`, `date`, `prc` (or `abs(prc)`), `vol`, `shrout`, `ret`.
*   **Optional Columns:** `openprc` (Required for `ret_gap_vol`), `askhi`, `bidlo` (Required for `parkinson_vol`, `clv_mean`), `numtrd`, `retx`, `cfacpr` (Required for extended factors).

### Output
*   **Format:** Parquet file (`daily_factors_advanced.parquet`).
*   **Index:** RangeIndex (Columns include `permno`, `date`).
*   **Columns:**
    *   `permno`, `date` (Keys)
    *   16 Factor Columns (Float64).

---

## 4. Extended Factors (Trade & Split Based)

| Symbol | Factor Name | Formula / Logic | Financial Interpretation |
| :--- | :--- | :--- | :--- |
| `avg_trade_size_1m` | Avg Trade Size | Mean($\frac{V_t}{NumTrd_t}$) | Institutional vs Retail. Larger = Institutional. |
| `illiq_numtrd_1m` | Trade Illiquidity | Mean($\frac{\|R_t\|}{NumTrd_t}$) | Cost per trade (Kyle's Lambda proxy). |
| `payout_yield_1m` | 1M Payout Yield | Sum($R_t - Retx_t$) | Dividend/Distribution Yield realized in month. |
| `intraday_ret_1m` | Daytime Mom | Mean($\frac{C_t}{O_t} - 1$) | Intraday return (Split-Adjusted). |
| `intraday_vol_1m` | Daytime Vol | Std($\frac{C_t}{O_t} - 1$) | Volatility during market hours only (vs Gap). |
