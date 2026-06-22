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

Bitcoin is not yet digital gold. While Bitcoin shares gold's scarcity and has experienced growing institutional adoption, its historical behavior more closely resembles that of a high-volatility risk asset than a traditional store of value or inflation hedge. Therefore, Bitcoin's role in a portfolio is better justified through diversification and return potential than through its ability to provide the same protections traditionally associated with gold.

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

## Methodology
### 1. Historical Performance Analysis

Calculated:

- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown

Purpose: To compare Bitcoin's risk and return profile against Gold and the S&P 500.

### 2. Correlation Analysis

Measured correlations between:

- Bitcoin and Gold
- Bitcoin and SPY
- Gold and SPY

Purpose: To determine whether Bitcoin behaves similarly to traditional assets or provides diversification benefits.

### 3. Inflation Analysis

Merged:

- Bitcoin monthly returns
- Year-over-Year CPI inflation
- Calculated: BTC vs Inflation correlation

Purpose: To test whether Bitcoin historically behaves as an inflation hedge.

### 4. Growth of $10,000 Analysis

Simulated a $10,000 investment in:

- Bitcoin
- Gold
- SPY

Purpose: To compare long-term wealth creation across assets.

## Key Findings
### Performance Metrics
| Asset | Annual Return | Volatility | Sharpe Ratio | Max Drawdown |
|--------|--------|--------|--------|--------|
| Bitcoin | 59.2% | 66.2% | 0.89 | -83.0% |
| SPY | 13.8% | 17.7% | 0.78 | -33.7% |
| Gold | 11.3% | 16.0% | 0.71 | -24.5% |

### Correlation Results
| Asset Pair | Correlation |
|--------|--------|
BTC vs Gold	| 0.09 |
BTC vs SPY | 0.23 |
Gold vs SPY	| 0.06 |

### Inflation Relationship
|Metric | Correlation |
|--------|---------|
| BTC Monthly Return vs Inflation YoY |	-0.23 |

## Initial Findings
- Bitcoin generated significantly higher returns than Gold and the S&P 500.
- Bitcoin exhibited substantially higher volatility than traditional assets.
- Bitcoin experienced deeper drawdowns, indicating greater downside risk.
- Bitcoin showed little relationship with Gold and only a weak relationship with the S&P 500.
- Bitcoin did not demonstrate strong inflation-hedging characteristics.
- Bitcoin may provide diversification benefits due to its low correlation with traditional assets.
- Institutional adoption continues to increase through ETFs and broader regulatory acceptance.

## Written Verdict
Based on the historical data analyzed in this study, the evidence does not support the claim that Bitcoin currently functions as digital gold. Although Bitcoin shares gold's scarcity characteristics and has benefited from growing institutional adoption, its market behavior differs significantly from that of a traditional store of value. Bitcoin exhibited substantially higher volatility and deeper drawdowns than both Gold and the S&P 500, while showing little correlation with Gold and a mildly negative relationship with inflation. At the same time, Bitcoin generated significantly higher long-term returns and the strongest risk-adjusted performance among the assets analyzed. These findings suggest that Bitcoin's investment case is currently driven more by diversification benefits, asymmetric return potential, and institutional interest than by its ability to serve as a reliable inflation hedge or direct substitute for Gold. Therefore, Bitcoin may deserve consideration as a speculative portfolio allocation, but the data indicates that it has not yet earned the status of true digital gold.

## Future Development
### Phase 2 – Advanced Research:
- Bitcoin ETF flow analysis
- Inflation regime analysis
- Risk-on vs risk-off market regime testing
- Institutional adoption trend analysis
- Rolling correlation analysis through time
### Phase 3 – Interactive Dashboard
- Streamlit dashboard
- Interactive portfolio allocation tool
- Dynamic risk and return analysis
- Historical scenario testing
### Phase 4 – High Performance & Optimization
- C++ portfolio optimization engine
- Pybind11 integration with Python
- Efficient frontier calculations
- Monte Carlo simulations (maybe)
- Optimal Bitcoin allocation recommendations
### Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- yfinance
- pandas-datareader

Author: Alexander Dominguez Zhakav
2026 Deutsche Bank Summer Internship Case Study
Topic: Bitcoin as Digital Gold: Hedge or Hype?