# Bitcoin as Digital Gold: Hedge or Hype?
## Overview

This project investigates whether Bitcoin behaves as a modern store of value comparable to gold. Bitcoin is frequently described as "digital gold" because of its fixed supply, scarcity, and growing institutional adoption. This study evaluates that claim using historical market data, inflation data, institutional adoption trends, and portfolio analytics. The goal is to determine whether Bitcoin deserves a place in a serious investment portfolio because it behaves like gold, or because it offers a different set of benefits.

## Research Question

### Can Bitcoin be considered digital gold?

Specifically:

- Does Bitcoin behave like a store of value?
- Does Bitcoin protect investors from inflation?
- Is Bitcoin correlated with traditional assets?
- Does Bitcoin improve portfolio diversification?
- How does institutional adoption impact Bitcoin's investment case?

## Thesis

Bitcoin is not yet digital gold. While its fixed supply and increasing institutional adoption support the narrative, empirical evidence suggests that Bitcoin behaves more like a high-beta risk asset than a safe-haven store of value. It failed to consistently hedge inflation, experienced significantly larger drawdowns than gold during periods of market stress, and became increasingly correlated with equities following institutional adoption after 2020. Nevertheless, small Bitcoin allocations improved portfolio efficiency and risk-adjusted returns, indicating that Bitcoin's value proposition lies in diversification and growth potential rather than in providing the protections traditionally associated with gold.

## Data Sources
### Market Data
- Bitcoin (BTC-USD)
- Gold (GLD ETF)
- S&P 500 (SPY ETF)

Source:

- Yahoo Finance
### Inflation Data
- Consumer Price Index (CPI)

Source:

- Federal Reserve Economic Data (FRED)
### Institutional Adoption Research
- State Street Global Advisors
- White House Strategic Bitcoin Reserve Executive Order
- ETF and Digital Asset Industry Reports
- Regulatory Publications

## Methodology Framework
This analysis tests the "digital gold" narrative through five direct tests
1. **Gold-like behavior:** Compare BTC returns, volatility, drawdowns, and correlation with GLD.
2. **Inflation hedge behavior:** Compare BTC performance across inflation regimes.
3. **Does Bitcoin Behave Like a Risk Asset?** Perform Risk on vs Risk off analysis on BTC and compare it with GLD and SPY.
4. **Stress-period behavior:** Compare BTC, SPY, and VIX during market stress.
5. **Post-2020 institutionalization:** Split results pre- and post-2020.
6. **Portfolio role:** Evaluate whether BTC improves portfolio as a diversifier.

### Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- yfinance
- pandas-datareader


### Instructions on how to run the program
One-time setup (do this once): open a terminal run these three commands in order:
`pip install pybind11 setuptools`
`brew install libomp`
`cd "/Users/alexanderdominguez/Desktop/Deutsche Bank Intern Project/app"`

Ever time you change `monte_carlo.cpp`: From the `app/` directory, run this single command to recompile:
`python setup.py build_ext --inplace`
This will produce a file like `monte_carlo.cpython-313-darwin.so` in the same `app/` folder.
`streamlit run main.py`

Author: Alexander Dominguez Zhakav
2026 Deutsche Bank Summer Internship Case Study
Topic: Bitcoin as Digital Gold: Hedge or Hype?