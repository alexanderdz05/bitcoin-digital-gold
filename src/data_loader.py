# download data
# load csv data
# Need CPI data

import yfinance as yf

btc = yf.download("BTC-USD", start="2015-01-01")
gold = yf.download("GLD", start="2015-01-01")
spy = yf.download("SPY", start="2015-01-01")

btc.to_csv("../data/btc_data.csv")
gold.to_csv("../data/gold_data.csv")
spy.to_csv("../data/spy_data.csv")