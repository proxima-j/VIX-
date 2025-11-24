# Backtest UPRO based on VIX rules
# Run in Jupyter / Colab / Codespaces
# If running in Colab: uncomment the pip install line below
# !pip install yfinance --quiet

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['figure.figsize'] = (12,6)

# === PARAMETERS ===
START_DATE = "2010-01-01"
END_DATE = None  # None => up to today
VIX_THR = 15.0
MA_WINDOW = 5
TRADE_COST = 0.000   # proportional cost per trade (0.05% default). Adjust as needed.
RISK_FREE = 0.0       # for Sharpe calculation
RESAMPLE = None       # e.g. 'M' if you want monthly frequency; None for daily

# === DOWNLOAD DATA ===
symbols = ["^VIX", "UPRO"]
df = yf.download(symbols, start=START_DATE, end=END_DATE, progress=False)['Close']

# rename columns for clarity (yfinance returns MultiIndex if multiple symbols)
if isinstance(df.columns, pd.MultiIndex):
    df = df.droplevel(0, axis=1)

df = df.rename(columns={"^VIX":"VIX", "UPRO":"UPRO"})
df = df[["VIX","UPRO"]].dropna()

# optional resample (rarely needed)
if RESAMPLE is not None:
    df = df.resample(RESAMPLE).last().dropna()

# === SIGNALS ===
df['VIX_MA5'] = df['VIX'].rolling(MA_WINDOW).mean()

# Desired exposure based on today's VIX
df['desired_long'] = (df['VIX'] < VIX_THR) & (df['VIX'] < df['VIX_MA5'])
# We execute at next period (to avoid lookahead) -> shift desired to next bar
df['position'] = df['desired_long'].shift(1).fillna(False).astype(int)

# === RETURNS ===
df['UPRO_ret'] = df['UPRO'].pct_change().fillna(0)

# apply trading cost on changes in position
df['pos_change'] = df['position'].diff().abs()
# when pos_change == 1 (enter or exit), apply trade cost once (on that bar)
# We'll subtract cost from strategy returns on the bar where trade happens.
df['trade_cost'] = df['pos_change'] * TRADE_COST

# strategy return: when position==1 earn UPRO returns, minus trade cost when trades occur
df['strategy_ret'] = df['UPRO_ret'] * df['position'] - df['trade_cost']

# cumulative NAVs (start from 1.0)
df['NAV_UPRO'] = (1 + df['UPRO_ret']).cumprod()
df['NAV_strategy'] = (1 + df['strategy_ret']).cumprod()

# === METRICS ===
def annualized_return_from_nav(nav, periods_per_year=252):
    total_days = len(nav.dropna())
    print(f"total days: {total_days}")
    if total_days < 2:
        return np.nan
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    years = total_days / periods_per_year
    print(f"Years: {years}")
    if years <= 0:
        return np.nan
    return (1 + total_return) ** (1/years) - 1

def ann_vol(returns, periods_per_year=252):
    return returns.std() * np.sqrt(periods_per_year)

def max_drawdown(nav):
    hwm = nav.cummax()
    dd = (nav - hwm) / hwm
    mdd = dd.min()
    end = dd.idxmin()
    start = (nav[:end]).idxmax()
    return float(mdd), start, end

PER_YEAR = 252 if RESAMPLE is None else {'M':12,'W':52}.get(RESAMPLE,252)

# Strategy metrics
ann_ret_strat = annualized_return_from_nav(df['NAV_strategy'], periods_per_year=PER_YEAR)
ann_vol_strat = ann_vol(df['strategy_ret'], periods_per_year=PER_YEAR)
sharpe_strat = (ann_ret_strat - RISK_FREE) / ann_vol_strat if ann_vol_strat != 0 else np.nan
mdd_strat, mdd_start_strat, mdd_end_strat = max_drawdown(df['NAV_strategy'])

# UPRO (buy-and-hold) metrics
ann_ret_upro = annualized_return_from_nav(df['NAV_UPRO'], periods_per_year=PER_YEAR)
ann_vol_upro = ann_vol(df['UPRO_ret'], periods_per_year=PER_YEAR)
sharpe_upro = (ann_ret_upro - RISK_FREE) / ann_vol_upro if ann_vol_upro != 0 else np.nan
mdd_upro, mdd_start_upro, mdd_end_upro = max_drawdown(df['NAV_UPRO'])

# === OUTPUT SUMMARY ===
print("Data range:", df.index[0].date(), "to", df.index[-1].date())
print("\n=== Strategy (VIX rule) ===")
print(f"Final NAV: {df['NAV_strategy'].iloc[-1]:.4f}")
print(f"Annualized return: {ann_ret_strat:.2%}")
print(f"Annualized vol: {ann_vol_strat:.2%}")
print(f"Sharpe (rf={RISK_FREE}): {sharpe_strat:.2f}")
print(f"Max Drawdown: {mdd_strat:.2%}  (from {mdd_start_strat.date()} to {mdd_end_strat.date()})")

print("\n=== Buy & Hold UPRO ===")
print(f"Final NAV: {df['NAV_UPRO'].iloc[-1]:.4f}")
print(f"Annualized return: {ann_ret_upro:.2%}")
print(f"Annualized vol: {ann_vol_upro:.2%}")
print(f"Sharpe (rf={RISK_FREE}): {sharpe_upro:.2f}")
print(f"Max Drawdown: {mdd_upro:.2%}  (from {mdd_start_upro.date()} to {mdd_end_upro.date()})")

# trade stats
total_trades = int(df['pos_change'].sum())
days_in_market = int(df['position'].sum())
print("\nTrades executed (enter or exit counted):", total_trades)
print("Total days in market (sum of positions):", days_in_market)

# === PLOTS ===
fig, ax = plt.subplots(2,1, figsize=(14,10), sharex=True)

ax[0].plot(df.index, df['NAV_UPRO'], label='UPRO (Buy & Hold)', linewidth=1.2)
ax[0].plot(df.index, df['NAV_strategy'], label='Strategy (VIX rule)', linewidth=1.4)
ax[0].set_title('NAV Comparison')
ax[0].legend()
ax[0].grid(True)

ax2 = ax[1]
ax2.plot(df.index, df['VIX'], label='VIX')
ax2.plot(df.index, df['VIX_MA5'], label='VIX MA5', alpha=0.8)
ax2.fill_between(df.index, 0, df['position']*df['VIX'].max(), alpha=0.12, label='In Market (shifted)')
ax2.axhline(VIX_THR, color='k', linestyle='--', linewidth=0.8, label=f'VIX threshold {VIX_THR}')
ax2.set_title('VIX, VIX_MA5 and Position (shaded)')
ax2.set_ylabel('VIX')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Drawdown plot for strategy
nav = df['NAV_strategy']
hwm = nav.cummax()
dd = (nav - hwm)/hwm

plt.figure(figsize=(12,4))
plt.plot(dd.index, dd, label='Strategy Drawdown')
plt.fill_between(dd.index, dd, 0, where=dd<0, alpha=0.2)
plt.title('Strategy Drawdown')
plt.ylabel('Drawdown')
plt.grid(True)
plt.show()

# Optional: show head of dataframe
display_cols = ['VIX','VIX_MA5','desired_long','position','UPRO','UPRO_ret','strategy_ret','NAV_strategy']
print("\nRecent data (tail):")
print(df[display_cols].tail(10).to_string())

# === Save results to CSV (optional) ===
# df.to_csv("vix_upro_backtest_results.csv")
# print("Saved to vix_upro_backtest_results.csv")
