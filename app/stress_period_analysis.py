import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pandas_datareader import data as pdr
from datetime import date

# Data Loaders
@st.cache_data
def load_asset_prices():
    prices = pd.read_csv("data/asset_prices.csv", index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    return prices, returns

@st.cache_data(ttl=86400)
def load_cpi():
    try:
        cpi = pdr.DataReader("CPIAUCSL", "fred", "2014-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
        cpi = cpi.rename(columns={"CPIAUCSL": "CPI"})
        cpi["Inflation YoY"] = cpi["CPI"].pct_change(12, fill_method=None)
        return cpi
    except Exception:
        return None

# Math helpers 
def ann_return(daily_returns):
    if len(daily_returns) < 2:
        return 0.0
    cumulative = (1 + daily_returns).prod()
    years = len(daily_returns) / 252
    return cumulative ** (1 / years) - 1 if years > 0 else 0.0

def ann_vol(series):
    return series.std() * np.sqrt(252)

def cumulative_return(return_series):
    return (1 + return_series).prod() - 1

def max_dd_from_prices(price_series):
    running_max = price_series.cummax()
    return (price_series / running_max - 1).min()

def recovery_days_from_trough(price_series):
    starting_value = price_series.iloc[0]
    running_max = price_series.cummax()
    trough_date = (price_series / running_max - 1).idxmin()
    after_trough = price_series.loc[trough_date:]
    recovered = after_trough[after_trough >= starting_value]
    if recovered.empty:
        return None
    return (recovered.index[0] - trough_date).days

# Preset stress periods
STRESS_PERIODS = {
    "COVID Crash (Feb–Mar 2020)": ("2020-02-19", "2020-03-23"),
    "COVID Recovery (Feb–Aug 2020)": ("2020-02-19", "2020-08-31"),
    "2022 Inflation & Rate Hikes": ("2022-01-01", "2022-12-31"),
    "2018 Crypto Winter": ("2018-01-01", "2018-12-31"),
    "2022 Crypto Winter (May–Nov)": ("2022-05-01", "2022-11-30"),
}

ASSET_COLORS = {"BTC": "#F7931A", "Gold": "#EAB308", "SPY": "#3B82F6"}
CORE_ASSETS = ["BTC", "Gold", "SPY"]

def show_stress_period_analysis():
    # Title and Desc
    st.title("₿ Stress Period Analysis")
    st.markdown(
        "### Is Bitcoin an inflation hedge? "
        "How does it behave when markets are in crisis?"
    )

    st.divider()

    prices, returns = load_asset_prices()
    cpi = load_cpi()

    # Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Inflation Regime Thresholds")
    low_thresh = st.sidebar.slider(
        "Low → Moderate boundary",
        min_value=0.5, max_value=4.0, value=2.0, step=0.5,
        format="%.1f%%", key="spa_low_thresh",
        help="Monthly YoY inflation below this is classified as Low.",
    )
    high_thresh = st.sidebar.slider(
        "Moderate → High boundary",
        min_value=1.0, max_value=10.0, value=4.0, step=0.5,
        format="%.1f%%", key="spa_high_thresh",
        help="Monthly YoY inflation above this is classified as High.",
    )
    if low_thresh >= high_thresh:
        st.sidebar.warning("Low threshold must be below High threshold.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Stress Periods")
    selected_periods = st.sidebar.multiselect(
        "Select Periods",
        options=list(STRESS_PERIODS.keys()),
        default=["COVID Crash (Feb–Mar 2020)", "2022 Inflation & Rate Hikes"],
        key="spa_periods",
    )

    st.sidebar.markdown("**Custom Period**")
    custom_label = st.sidebar.text_input("Label (optional)", value="", key="spa_custom_label")
    custom_range = st.sidebar.date_input(
        "Date Range",
        value=(date(2023, 1, 1), date(2023, 6, 30)),
        min_value=prices.index.min().date(),
        max_value=prices.index.max().date(),
        key="spa_custom_range",
    )

    # Test 2: Inflation Hedge
    st.header("Test 2 — Inflation Hedge Behavior")

    if cpi is None:
        st.error(
            "Could not load CPI data from FRED. "
            "Check your internet connection or try again later."
        )
    else:
        monthly_prices = prices.resample("ME").last()
        monthly_returns = monthly_prices.pct_change().dropna()

        m_ret = monthly_returns.copy()
        m_ret.index = m_ret.index.to_period("M")

        cpi_monthly = cpi.copy()
        cpi_monthly.index = cpi_monthly.index.to_period("M")

        regime_data = m_ret.merge(
            cpi_monthly[["Inflation YoY"]],
            left_index=True, right_index=True, how="inner"
        ).dropna(subset=["Inflation YoY"])

        low_t = low_thresh / 100
        high_t = high_thresh / 100

        label_low = f"Low (<{low_thresh:.1f}%)"
        label_mod = f"Moderate ({low_thresh:.1f}–{high_thresh:.1f}%)"
        label_high = f"High (>{high_thresh:.1f}%)"
        regime_order = [label_low, label_mod, label_high]

        def classify(inf):
            if inf < low_t:
                return label_low
            elif inf < high_t:
                return label_mod
            return label_high

        regime_data["Regime"] = regime_data["Inflation YoY"].apply(classify)

        # Correlation metrics
        btc_corr = regime_data[["BTC", "Inflation YoY"]].corr().loc["BTC", "Inflation YoY"]
        gold_corr = regime_data[["Gold", "Inflation YoY"]].corr().loc["Gold", "Inflation YoY"]
        spy_corr = regime_data[["SPY", "Inflation YoY"]].corr().loc["SPY", "Inflation YoY"]

        c1, c2, c3 = st.columns(3)
        c1.metric("BTC–Inflation Corr.", f"{btc_corr:.3f}")
        c2.metric("Gold–Inflation Corr.", f"{gold_corr:.3f}")
        c3.metric("SPY–Inflation Corr.", f"{spy_corr:.3f}")
        st.caption(
            "A true inflation hedge should have a **positive** correlation with inflation. "
            "Negative means the asset tends to fall when inflation rises."
        )

        st.divider()

        # Annualized returns by regime — grouped bar chart
        st.subheader("Annualized Returns by Inflation Regime")
        st.caption(
            f"Regimes: Low (<{low_thresh:.1f}%), "
            f"Moderate ({low_thresh:.1f}–{high_thresh:.1f}%), "
            f"High (>{high_thresh:.1f}%). Adjust thresholds in the sidebar."
        )

        regime_monthly_avg = regime_data.groupby("Regime")[CORE_ASSETS].mean()
        regime_ann = ((1 + regime_monthly_avg) ** 12 - 1).reindex(regime_order)

        fig_inf_bar = go.Figure()
        for asset in CORE_ASSETS:
            fig_inf_bar.add_trace(go.Bar(
                name=asset, x=regime_ann.index, y=regime_ann[asset],
                marker_color=ASSET_COLORS[asset],
            ))
        fig_inf_bar.add_hline(y=0, line_color="black", line_width=1)
        fig_inf_bar.update_layout(
            title="Annualized Returns Across Inflation Regimes",
            yaxis_title="Annualized Return", yaxis_tickformat=".0%",
            barmode="group", height=400, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_inf_bar, use_container_width=True)

        # Regime summary table
        st.subheader("Regime Summary")
        obs_count = regime_data["Regime"].value_counts()
        summary_rows = []
        for regime in regime_order:
            rd = regime_data[regime_data["Regime"] == regime]
            n = len(rd)
            summary_rows.append({
                "Regime": regime,
                "Months": n,
                "BTC Ann. Return": f"{((1 + rd['BTC'].mean()) ** 12 - 1):.1%}" if n > 0 else "—",
                "Gold Ann. Return": f"{((1 + rd['Gold'].mean()) ** 12 - 1):.1%}" if n > 0 else "—",
                "SPY Ann. Return": f"{((1 + rd['SPY'].mean()) ** 12 - 1):.1%}" if n > 0 else "—",
                "BTC Monthly Vol.": f"{rd['BTC'].std() * np.sqrt(12):.1%}" if n > 0 else "—",
            })
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

        st.divider()

        # Scatter: BTC monthly returns vs inflation
        st.subheader("Bitcoin Monthly Returns vs YoY Inflation")
        st.caption(
            "Each point is one calendar month, colored by regime. "
            "A true inflation hedge should trend upward left-to-right."
        )

        scatter_df = regime_data[["BTC", "Inflation YoY", "Regime"]].copy()
        scatter_df.index = scatter_df.index.astype(str)

        regime_scatter_colors = {
            label_low: "#3B82F6",
            label_mod: "#F59E0B",
            label_high: "#EF4444",
        }

        fig_scatter = go.Figure()
        for regime in regime_order:
            subset = scatter_df[scatter_df["Regime"] == regime]
            fig_scatter.add_trace(go.Scatter(
                x=subset["Inflation YoY"], y=subset["BTC"],
                mode="markers", name=regime,
                marker=dict(color=regime_scatter_colors[regime], size=7, opacity=0.8),
                text=subset.index,
                hovertemplate="<b>%{text}</b><br>Inflation: %{x:.1%}<br>BTC Return: %{y:.1%}<extra></extra>",
            ))

        x_vals = scatter_df["Inflation YoY"].values
        y_vals = scatter_df["BTC"].values
        m_slope = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig_scatter.add_trace(go.Scatter(
            x=x_line, y=np.polyval(m_slope, x_line),
            mode="lines", name="Trend (OLS)",
            line=dict(color="gray", dash="dot", width=1.5),
        ))

        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
        fig_scatter.update_layout(
            title=f"BTC Monthly Return vs YoY Inflation  (corr: {btc_corr:.3f})",
            xaxis_title="Inflation YoY", yaxis_title="BTC Monthly Return",
            xaxis_tickformat=".1%", yaxis_tickformat=".0%",
            height=430, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.info(
            f"**Key Finding (Test 2):** Bitcoin's correlation with inflation is **{btc_corr:.3f}** "
            f"vs Gold's **{gold_corr:.3f}**. Bitcoin generated its strongest returns during "
            f"**{label_low}** periods and its worst during **{label_high}** periods — "
            "the opposite of an inflation hedge."
        )

    st.divider()

    # Test 4:
    st.header("Test 4 — Stress-Period Behavior")
    st.caption(
        "Compares Bitcoin, Gold, and the S&P 500 during major market crises. "
        "A safe-haven asset should hold its value or rise when other markets fall."
    )

    active_periods = {k: STRESS_PERIODS[k] for k in selected_periods}
    if custom_label.strip() and isinstance(custom_range, (list, tuple)) and len(custom_range) == 2:
        active_periods[custom_label.strip()] = (str(custom_range[0]), str(custom_range[1]))

    if not active_periods:
        st.warning("Select at least one stress period from the sidebar.")
        return

    # Compute metrics for each period
    stress_rows = []
    for period_name, (p_start, p_end) in active_periods.items():
        p_prices = prices.loc[p_start:p_end, CORE_ASSETS]
        p_returns = returns.loc[p_start:p_end, CORE_ASSETS]
        if len(p_returns) < 5:
            st.warning(f"Not enough data for '{period_name}'. Skipping.")
            continue
        for asset in CORE_ASSETS:
            rec = recovery_days_from_trough(p_prices[asset])
            stress_rows.append({
                "Period": period_name,
                "Asset": asset,
                "Cumul. Return": cumulative_return(p_returns[asset]),
                "Max Drawdown": max_dd_from_prices(p_prices[asset]),
                "Recovery Days": rec,
            })

    if not stress_rows:
        st.warning("No valid data for the selected periods.")
        return

    stress_df = pd.DataFrame(stress_rows)
    period_order = list(active_periods.keys())

    # Cumulative returns — grouped bar chart
    st.subheader("Cumulative Returns During Stress Periods")

    fig_stress_bar = go.Figure()
    for asset in CORE_ASSETS:
        asset_rows = stress_df[stress_df["Asset"] == asset].set_index("Period")
        y_vals = [
            asset_rows.loc[p, "Cumul. Return"] if p in asset_rows.index else None
            for p in period_order
        ]
        fig_stress_bar.add_trace(go.Bar(
            name=asset, x=period_order, y=y_vals,
            marker_color=ASSET_COLORS[asset],
        ))
    fig_stress_bar.add_hline(y=0, line_color="black", line_width=1)
    fig_stress_bar.update_layout(
        title="Cumulative Returns During Stress Periods",
        yaxis_title="Cumulative Return", yaxis_tickformat=".0%",
        barmode="group", height=420, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_stress_bar, use_container_width=True)

    # Tabbed detail tables
    st.subheader("Stress Period Detail")
    tab_ret, tab_dd, tab_rec = st.tabs(["Cumulative Returns", "Max Drawdowns", "Recovery Days"])

    def _pivot(metric):
        return (
            stress_df.pivot(index="Period", columns="Asset", values=metric)
            .reindex(period_order)[CORE_ASSETS]
        )

    def _color_ret(val):
        if not isinstance(val, (int, float)) or np.isnan(val):
            return ""
        alpha = min(abs(val) * 2.5, 0.8)
        color = "34,197,94" if val >= 0 else "239,68,68"
        text = "white" if alpha > 0.5 else "black"
        return f"background-color: rgba({color},{alpha:.2f}); color: {text}"

    def _color_dd(val):
        if not isinstance(val, (int, float)) or np.isnan(val):
            return ""
        alpha = min(abs(val) * 2.5, 0.8)
        return f"background-color: rgba(239,68,68,{alpha:.2f}); color: {'white' if alpha > 0.5 else 'black'}"

    with tab_ret:
        pivot_ret = _pivot("Cumul. Return")
        st.dataframe(pivot_ret.style.format("{:.1%}").map(_color_ret), use_container_width=True)

    with tab_dd:
        pivot_dd = _pivot("Max Drawdown")
        st.dataframe(pivot_dd.style.format("{:.1%}").map(_color_dd), use_container_width=True)

    with tab_rec:
        pivot_rec = _pivot("Recovery Days")

        def _fmt_rec(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "Not Recovered"
            return f"{int(val)} days"

        st.dataframe(pivot_rec.map(_fmt_rec), use_container_width=True)

    st.divider()

    # Normalized price drill-down
    st.subheader("Normalized Price Chart")
    st.caption(
        "All assets start at 100 on the first day of the selected period. "
        "Shows exactly how each asset moved through the crisis in real time."
    )

    drill_period = st.selectbox(
        "Select Period to Drill Into",
        options=period_order,
        key="spa_drill",
    )

    d_start, d_end = active_periods[drill_period]
    drill_prices = prices.loc[d_start:d_end, CORE_ASSETS]

    if len(drill_prices) < 2:
        st.warning("Not enough data for this period.")
        return

    normalized = drill_prices / drill_prices.iloc[0] * 100

    fig_norm = go.Figure()
    for asset in CORE_ASSETS:
        fig_norm.add_trace(go.Scatter(
            x=normalized.index, y=normalized[asset],
            mode="lines", name=asset,
            line=dict(color=ASSET_COLORS[asset], width=2.5),
            hovertemplate=f"<b>{asset}</b>: %{{y:.1f}}<extra></extra>",
        ))
    fig_norm.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5,
                       annotation_text="Starting value (100)")
    fig_norm.update_layout(
        title=f"Normalized Price — {drill_period}",
        xaxis_title="Date", yaxis_title="Indexed Price (100 = start of period)",
        height=430, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_norm, use_container_width=True)

    st.info(
        "**Key Finding (Test 4):** During the COVID crash, Bitcoin lost ~37% — "
        "nearly as much as equities and far more than Gold (-3%). "
        "In the 2022 rate hike cycle, Bitcoin fell 64% while Gold stayed nearly flat. "
        "This pattern is consistent with a high-beta risk asset, not a safe haven."
    )