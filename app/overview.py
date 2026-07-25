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

def show_overview():
    # Title and Desc
    st.title("₿ Bitcoin as Digital Gold: Hedge or Hype")
    st.markdown("""
    Five empirical tests. One verdict. Use the navigation to explore the evidence behind each claim —
    then run the portfolio optimizer to see what allocation the data actually supports.
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

    st.subheader("Methodology")
    st.markdown("""
    This analysis evaluates the digital-gold narrative through five empirical tests, followed by a portfolio-allocation analysis that determines Bitcoin’s practical role.

    1. Gold-like behavior
    2. Inflation hedge behavior
    3. Risk asset behavior
    4. Stress-period behavior
    5. Post-2020 institutionalization \n
    Bonus: Portfolio Role
    """)