import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Helper Functions
@st.cache_data(ttl=300)
def load_btc_data():
    btc = yf.download("BTC-USD", period="10y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    return btc.dropna()

@st.cache_data
def load_asset_prices():
    prices = pd.read_csv("data/asset_prices.csv", index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    return prices, returns

def annualized_return(daily_returns):
    return (1 + daily_returns.mean()) ** 252 - 1

def show_overview():
    # Title and Desc
    st.title("₿ Bitcoin as Digital Gold: Hedge or Hype")
    st.markdown("""
    ### Project Thesis
    Bitcoin is not yet digital gold. While its fixed supply and increasing institutional adoption support the narrative, 
    empirical evidence suggests that Bitcoin behaves more like a high-beta risk asset than a safe-haven store of value. 
    It failed to consistently hedge inflation, experienced significantly larger drawdowns than gold during periods of market stress, 
    and became increasingly correlated with equities following institutional adoption after 2020. 
    Nevertheless, small Bitcoin allocations improved portfolio efficiency and risk-adjusted returns, 
    indicating that Bitcoin's value proposition lies in diversification and growth potential rather than in providing the protections traditionally associated with gold.
    """)
    
    # Bitcoin Live Price

    st.subheader("Live Bitcoin Price Dashboard")
    btc_data = load_btc_data()
    range_option = st.selectbox(
        "Select BTC chart range",
        ["1d", "7d", "1mo", "6mo", "1y", "5y", "10y"],
        index=4
    )
    
    end_date = btc_data.index.max()
    range_days = {
        "1d": 1,
        "7d": 7,
        "1mo": 30,
        "6mo": 182,
        "1y": 365,
        "5y": 365 * 5,
        "10y": 365 * 10
    }
    start_date = end_date - pd.Timedelta(days=range_days[range_option])
    btc_chart_data = btc_data.loc[btc_data.index >= start_date]

    latest_price = btc_chart_data["Close"].iloc[-1]
    previous_price = btc_chart_data["Close"].iloc[0]
    price_change = latest_price - previous_price
    price_change_pct = price_change / previous_price

    col_price, col_change, col_high, col_low = st.columns(4)
    col_price.metric(
        "BTC Price",
        f"${latest_price:,.2f}"
    )
    col_change.metric(
        f"{range_option} Change",
        f"{price_change_pct:.2%}",
        f"${price_change:,.2f}"
    )
    col_high.metric(
        "Range High",
        f"${btc_chart_data['High'].max():,.2f}"
    )
    col_low.metric(
        "Range Low",
        f"${btc_chart_data['Low'].min():,.2f}"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=btc_chart_data.index,
            y=btc_chart_data["Close"],
            mode="lines",
            name="BTC-USD",
            line=dict(width=3)
        )
    )
    fig.update_layout(
        title=f"BTC-USD Price Chart ({range_option})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=450,
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Overview Metrics
    prices, returns = load_asset_prices()

    pre_2020_returns = returns.loc[: "2020-12-31"]
    post_2020_returns = returns.loc["2020-01-01" :]

    pre_btc_return = annualized_return(pre_2020_returns["BTC"])
    post_btc_return = annualized_return(post_2020_returns["BTC"])

    btc_return_delta = post_btc_return - pre_btc_return

    btc_spy_corr = post_2020_returns[["BTC", "SPY"]].corr().loc["BTC", "SPY"]
    btc_gold_corr = post_2020_returns[["BTC", "Gold"]].corr().loc["BTC", "Gold"]

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "BTC Annual Return",
        f"{post_btc_return:.1%}",
        f"{btc_return_delta:.1%} from pre-2020"
    )
    col2.metric(
        "BTC-SPY Correlation",
        f"{btc_spy_corr:.2f}",
        f"+{btc_spy_corr - pre_2020_returns[["BTC", "SPY"]].corr().loc["BTC", "SPY"]:.2f} since pre-2020"
    )
    col3.metric(
        "BTC-Gold Correlation",
        f"{btc_gold_corr:.2f}",
        "Still weak"
    )

    st.divider()

    st.subheader("Methodology")
    st.markdown("""
    This analysis evaluates the digital gold narrative through six tests:

    1. Gold-like behavior
    2. Inflation hedge behavior
    3. Risk asset behavior
    4. Stress-period behavior
    5. Post-2020 institutionalization
    6. Portfolio diversification
    """)