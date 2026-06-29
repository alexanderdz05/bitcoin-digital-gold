import streamlit as st 
from overview import show_overview
from digital_gold_tests import show_digital_gold_tests
from stress_period_analysis import show_stress_period_analysis
from portfolio_simulator import show_portfolio_simulator

# Page Config

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
        "Bitcoin's Portfolio Role",
        "Stress Period Analysis",
        "Portfolio Simulator"
    ]
)

if page == "Overview":
    show_overview()
elif page == "Bitcoin's Portfolio Role":
    show_digital_gold_tests()
elif page == "Stress Period Analysis":
    show_stress_period_analysis()
elif page == "Portfolio Simulator":
    show_portfolio_simulator()
else:
    st.write("Error: Page not found")