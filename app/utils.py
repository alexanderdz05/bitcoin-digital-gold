import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import os

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CORE_ASSETS = ["BTC", "Gold", "SPY"]
ASSET_COLORS = {
    "BTC": "#DC2626",
    "Gold": "#EAB308",
    "SPY": "#3B82F6"
}

STRESS_PERIODS = {
    "COVID Crash (Feb–Mar 2020)": ("2020-02-19", "2020-03-23"),
    "COVID Recovery (Mar–Aug 2020)": ("2020-03-24", "2020-08-31"),
    "2022 Inflation & Rate Hikes": ("2022-01-01", "2022-12-31"),
    "2018 Crypto Winter": ("2018-01-01", "2018-12-31"),
    "2022 Crypto Winter (May–Nov)": ("2022-05-01", "2022-11-30"),
}

@st.cache_data
def load_asset_prices():
    prices = pd.read_csv(os.path.join(_DATA_DIR, "asset_prices.csv"), index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    return prices, returns

@st.cache_data(ttl=86400)
def load_cpi():
    try:
        cpi = pdr.DataReader(
            "CPIAUCSL",
            "fred",
            "2014-01-01",
            pd.Timestamp.today().strftime("%Y-%m-%d")
        )
        cpi = cpi.rename(columns={"CPIAUCSL": "CPI"})
        cpi["Inflation YoY"] = cpi["CPI"].pct_change(12, fill_method=None)
        return cpi
    except Exception:
        return None

def ann_return(daily_returns):
    if len(daily_returns) < 2:
        return 0.0
    cumulative = (1 + daily_returns).prod()
    years = len(daily_returns) / 252
    return cumulative ** (1 / years) - 1 if years > 0 else 0.0

def ann_vol(daily_returns):
    return daily_returns.std() * np.sqrt(252)

def sharpe(daily_returns, rf=0.0):
    vol = ann_vol(daily_returns)
    return (ann_return(daily_returns) - rf) / vol if vol > 0 else 0.0

def cumulative_return(return_series):
    return (1 + return_series).prod() - 1

def max_dd_from_prices(price_series):
    running_max = price_series.cummax()
    return (price_series / running_max - 1).min()

def max_dd_from_returns(return_series):
    growth = (1 + return_series).cumprod()
    running_max = growth.cummax()
    return (growth / running_max - 1).min()

def recovery_days_from_trough(price_series):
    starting_value = price_series.iloc[0]
    running_max = price_series.cummax()
    drawdown = price_series / running_max - 1
    trough_date = drawdown.idxmin()

    after_trough = price_series.loc[trough_date:]
    recovered = after_trough[after_trough >= starting_value]

    if recovered.empty:
        return None

    return (recovered.index[0] - trough_date).days