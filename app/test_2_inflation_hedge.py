import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import load_asset_prices, load_cpi, CORE_ASSETS, ASSET_COLORS

def show_test2():
    # Title and Header
    st.title("Test 2 — Inflation Hedge Behavior")
    st.markdown("### Does Bitcoin protect purchasing power when inflation rises?")

    # Inflation Correlations
    prices, returns = load_asset_prices()
    cpi = load_cpi()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Inflation Regime Thresholds")

    low_thresh = st.sidebar.slider(
        "Low → Moderate boundary",
        min_value=0.5,
        max_value=4.0,
        value=2.0,
        step=0.5,
        format="%.1f%%",
        key="test2_low_thresh",
    )

    high_thresh = st.sidebar.slider(
        "Moderate → High boundary",
        min_value=1.0,
        max_value=10.0,
        value=4.0,
        step=0.5,
        format="%.1f%%",
        key="test2_high_thresh",
    )

    if low_thresh >= high_thresh:
        st.warning("Low threshold must be below high threshold.")
        return

    if cpi is None:
        st.error("Could not load CPI data from FRED.")
        return

    st.info(
        "**Hypothesis:** If Bitcoin is digital gold, it should perform well during high inflation "
        "and show a positive relationship with inflation."
    )

    monthly_prices = prices[CORE_ASSETS].resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()

    m_ret = monthly_returns.copy()
    m_ret.index = m_ret.index.to_period("M")

    cpi_monthly = cpi.copy()
    cpi_monthly.index = cpi_monthly.index.to_period("M")

    regime_data = m_ret.merge(
        cpi_monthly[["Inflation YoY"]],
        left_index=True,
        right_index=True,
        how="inner",
    ).dropna(subset=["Inflation YoY"])

    low_t = low_thresh / 100
    high_t = high_thresh / 100

    label_low = f"Low (<{low_thresh:.1f}%)"
    label_mod = f"Moderate ({low_thresh:.1f}–{high_thresh:.1f}%)"
    label_high = f"High (>{high_thresh:.1f}%)"
    regime_order = [label_low, label_mod, label_high]

    def classify_inflation(inflation):
        if inflation < low_t:
            return label_low
        elif inflation < high_t:
            return label_mod
        return label_high

    regime_data["Regime"] = regime_data["Inflation YoY"].apply(classify_inflation)

    btc_corr = regime_data[["BTC", "Inflation YoY"]].corr().loc["BTC", "Inflation YoY"]
    gold_corr = regime_data[["Gold", "Inflation YoY"]].corr().loc["Gold", "Inflation YoY"]
    spy_corr = regime_data[["SPY", "Inflation YoY"]].corr().loc["SPY", "Inflation YoY"]

    c1, c2, c3 = st.columns(3)
    c1.metric("BTC–Inflation Corr.", f"{btc_corr:.3f}")
    c2.metric("Gold–Inflation Corr.", f"{gold_corr:.3f}")
    c3.metric("SPY–Inflation Corr.", f"{spy_corr:.3f}")

    st.caption(
        "A true inflation hedge should have a positive correlation with inflation. "
        "Negative means the asset tends to fall when inflation rises."
    )

    st.divider()

    # Annualized Returns by Inflation Regime
    st.subheader("Annualized Returns by Inflation Regime")

    regime_monthly_avg = regime_data.groupby("Regime")[CORE_ASSETS].mean()
    regime_ann = ((1 + regime_monthly_avg) ** 12 - 1).reindex(regime_order)

    fig = go.Figure()

    for asset in CORE_ASSETS:
        fig.add_trace(
            go.Bar(
                name=asset,
                x=regime_ann.index,
                y=regime_ann[asset],
                marker_color=ASSET_COLORS[asset],
            )
        )

    fig.add_hline(y=0, line_color="black", line_width=1)

    fig.update_layout(
        title="Annualized Returns Across Inflation Regimes",
        yaxis_title="Annualized Return",
        yaxis_tickformat=".0%",
        barmode="group",
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regime Summary")

    rows = []

    for regime in regime_order:
        rd = regime_data[regime_data["Regime"] == regime]
        n = len(rd)

        rows.append(
            {
                "Regime": regime,
                "Months": n,
                "BTC Ann. Return": f"{((1 + rd['BTC'].mean()) ** 12 - 1):.1%}" if n > 0 else "—",
                "Gold Ann. Return": f"{((1 + rd['Gold'].mean()) ** 12 - 1):.1%}" if n > 0 else "—",
                "SPY Ann. Return": f"{((1 + rd['SPY'].mean()) ** 12 - 1):.1%}" if n > 0 else "—",
                "BTC Monthly Vol.": f"{rd['BTC'].std() * np.sqrt(12):.1%}" if n > 0 else "—",
            }
        )

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()

    # Bitcoin Monthly Returns vs YoY Inflation
    st.subheader("Bitcoin Monthly Returns vs YoY Inflation")

    scatter_df = regime_data[["BTC", "Inflation YoY", "Regime"]].copy()
    scatter_df.index = scatter_df.index.astype(str)

    regime_colors = {
        label_low: "#3B82F6",
        label_mod: "#F59E0B",
        label_high: "#EF4444",
    }

    fig_scatter = go.Figure()

    for regime in regime_order:
        subset = scatter_df[scatter_df["Regime"] == regime]

        fig_scatter.add_trace(
            go.Scatter(
                x=subset["Inflation YoY"],
                y=subset["BTC"],
                mode="markers",
                name=regime,
                marker=dict(color=regime_colors[regime], size=7, opacity=0.8),
                text=subset.index,
                hovertemplate="<b>%{text}</b><br>Inflation: %{x:.1%}<br>BTC Return: %{y:.1%}<extra></extra>",
            )
        )

    x_vals = scatter_df["Inflation YoY"].values
    y_vals = scatter_df["BTC"].values

    slope = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)

    fig_scatter.add_trace(
        go.Scatter(
            x=x_line,
            y=np.polyval(slope, x_line),
            mode="lines",
            name="Trend",
            line=dict(color="gray", dash="dot", width=1.5),
        )
    )

    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)

    fig_scatter.update_layout(
        title=f"BTC Monthly Return vs YoY Inflation (corr: {btc_corr:.3f})",
        xaxis_title="Inflation YoY",
        yaxis_title="BTC Monthly Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".0%",
        height=430,
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.info(
        f"**Key Finding (Test 2):** Bitcoin's correlation with inflation is **{btc_corr:.3f}** "
        f"vs Gold's **{gold_corr:.3f}**. Bitcoin performed worst during **{label_high}** periods, "
        "which does not support the inflation hedge narrative."
    )