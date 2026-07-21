# Bitcoin as Digital Gold: Hedge or Hype?

**Deutsche Bank Summer Internship Case Study — Alexander Dominguez Zhakav, 2026**

---

## Thesis

Bitcoin is not yet digital gold. While its fixed supply and increasing institutional adoption support the narrative, empirical evidence suggests that Bitcoin behaves more like a high-beta risk asset than a safe-haven store of value. It failed to consistently hedge inflation, experienced significantly larger drawdowns than gold during periods of market stress, and became increasingly correlated with equities following institutional adoption after 2020. Nevertheless, small Bitcoin allocations improved portfolio efficiency and risk-adjusted returns, indicating that Bitcoin's value proposition lies in diversification and growth potential rather than in the protections traditionally associated with gold.

---

## Project Structure

```
Deutsche Bank Intern Project/
├── app/
│   ├── main.py                        # Streamlit entry point and navigation
│   ├── overview.py                    # Overview page: live BTC dashboard + methodology
│   ├── utils.py                       # Shared data loaders and financial statistics
│   ├── test_1_gold_like_behavior.py   # Test 1: BTC vs Gold comparison
│   ├── test_2_inflation_hedge.py      # Test 2: Inflation regime analysis
│   ├── test_3_risk_asset_behavior.py  # Test 3: Risk-on/risk-off classification
│   ├── test_4_stress_period_behavior.py # Test 4: Market stress periods
│   ├── test_5_institutionalization.py # Test 5: Pre/post-2020 structural break
│   ├── portfolio_simulator.py         # Test 6: Monte Carlo portfolio optimizer
│   ├── monte_carlo.cpp                # C++ simulation engine (Pybind11 + OpenMP)
│   └── setup.py                       # Build script for the C++ extension
├── data/
│   └── asset_prices.csv               # Historical prices: BTC, Gold, SPY
├── notebooks/
│   └── btc_analysis.ipynb             # Exploratory analysis notebook
├── charts/                            # Static chart exports
├── research/
│   └── research.md                    # Background research notes
└── requirements.txt
```

---

## The Six Tests

| # | Test | Key Question |
|---|------|-------------|
| 1 | Gold-like behavior | Do BTC and Gold share return, volatility, and drawdown profiles? |
| 2 | Inflation hedge | Does BTC outperform during high-inflation regimes? |
| 3 | Risk asset behavior | Does BTC rally in risk-on environments and fall in risk-off? |
| 4 | Stress-period behavior | How does BTC perform vs SPY and Gold during market crises? |
| 5 | Post-2020 institutionalization | Did ETF adoption and institutional inflows change BTC's behavior? |
| 6 | Portfolio role | Does adding BTC to a portfolio improve risk-adjusted returns? |

---

## Portfolio Simulator (Test 6)

The portfolio simulator is the most technically intensive component. It runs a path-dependent Monte Carlo simulation across 3,000 randomly sampled portfolios to construct an efficient frontier and identify the optimal BTC allocation for a given risk profile.

### Simulation Design

- **GBM with Cholesky correlation** — log-returns drawn from a multivariate normal distribution calibrated to historical data; correlated via Cholesky decomposition of the covariance matrix
- **Quarterly drift rebalancing** — every 63 trading days, portfolios that have drifted more than 5% from target weights are rebalanced; a 0.1% transaction cost is applied to turnover
- **Drawdown circuit breaker** — if a portfolio falls more than 30% from its peak, BTC exposure is halved for the next 21 trading days
- **Multi-path averaging** — each weight vector is simulated over multiple independent paths and averaged, reducing noise
- **2-year horizon** — 504 trading days (2 × 252)

### Dual Engine: Python and C++

The simulator offers two execution modes that produce **bit-for-bit identical results**:

| Engine | Runtime | Implementation |
|--------|---------|----------------|
| Python | ~40s | Pure Python loop over 3,000 portfolios × 504 days |
| C++ | ~1–2s | Pybind11 extension, parallelized with OpenMP |

Identical output is guaranteed by pre-generating all random normals in Python (NumPy, seed 99) and passing the same `(n_portfolios, n_mc_paths, n_days, n_assets)` array to both engines. The C++ extension reads from this array instead of maintaining its own RNG.

### Output Metrics

For each portfolio: annualized return, annualized volatility, Sharpe ratio, maximum drawdown, and 95% Value at Risk. The optimal portfolio is selected based on the user's chosen risk profile (Max Sharpe, Min Volatility, or custom drawdown constraint).

---

## Data Sources

| Data | Source | Ticker / Series |
|------|--------|-----------------|
| Bitcoin | Yahoo Finance | BTC-USD |
| Gold | Yahoo Finance | GLD (ETF) |
| S&P 500 | Yahoo Finance | SPY (ETF) |
| CPI Inflation | FRED (Federal Reserve) | CPIAUCSL |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| App framework | Streamlit |
| Data & numerics | Pandas, NumPy |
| Charting | Plotly |
| Market data | yfinance |
| Macro data | pandas-datareader (FRED) |
| C++ interop | Pybind11 |
| C++ parallelism | OpenMP |
| Language | Python 3.13, C++17 |

---

## Setup and Installation

### One-time setup

```bash
pip install streamlit pandas numpy plotly yfinance pandas-datareader pybind11 setuptools
brew install libomp
```

### Compile the C++ extension

Run this from the `app/` directory. Re-run it any time `monte_carlo.cpp` changes.

```bash
cd "/{Parent Directory}/Deutsche Bank Intern Project/app"
python setup.py build_ext --inplace
```

This produces `monte_carlo.cpython-313-darwin.so` in the `app/` folder. The app falls back to Python-only mode if the `.so` is not present.

### Run the app

```bash
cd "/{Parent Directory}/Deutsche Bank Intern Project/app"
streamlit run main.py
```

---

## Key Findings

- Bitcoin's correlation with SPY increased significantly post-2020, weakening its diversification case
- BTC failed to consistently outperform during high-inflation regimes; its returns were driven by risk appetite, not inflation expectations
- During market stress events (COVID crash, 2022 rate hikes), BTC experienced drawdowns 2–3× larger than gold
- Despite these weaknesses, small BTC allocations (5–15%) improved Sharpe ratios on mixed portfolios due to the diversification benefit of low long-run correlation with bonds and gold
- Post-2020 institutionalization increased BTC's equity-like behavior, reducing but not eliminating its portfolio diversification value