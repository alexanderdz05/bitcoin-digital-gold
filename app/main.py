import streamlit as st

from overview import show_overview
from test_1_gold_like_behavior import show_test1
from test_2_inflation_hedge import show_test2
from test_3_risk_asset_behavior import show_test3
from test_4_stress_period_behavior import show_test4
from test_5_institutionalization import show_test5
from portfolio_simulator import show_portfolio_simulator

st.set_page_config(
    page_title="Bitcoin as Digital Gold",
    page_icon="₿",
    layout="wide"
)

st.sidebar.title("Bitcoin as Digital Gold")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Test 1 - Gold-like Behavior",
        "Test 2 - Inflation Hedge",
        "Test 3 - Risk Asset Behavior",
        "Test 4 - Stress Period Behavior",
        "Test 5 - Institutionalization",
        "Portfolio Role"
    ]
)

if page == "Overview":
    show_overview()
elif page == "Test 1 - Gold-like Behavior":
    show_test1()
elif page == "Test 2 - Inflation Hedge":
    show_test2()
elif page == "Test 3 - Risk Asset Behavior":
    show_test3()
elif page == "Test 4 - Stress Period Behavior":
    show_test4()
elif page == "Test 5 - Institutionalization":
    show_test5()
elif page == "Portfolio Role":
    show_portfolio_simulator()