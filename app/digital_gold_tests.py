import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# Helper functions
def load_data():
    prices = pd.read_csv("data/asset_prices.csv", index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    return prices, returns

def ann_return(daily_returns):
    if len(daily_returns) < 2:
        return 0.0
    cumulative = (1 + daily_returns).prod()
    years = len(daily_returns) / 252
    return cumulative ** (1 / years) - 1 if years > 0 else 0.0

def ann_vol(daily_returns):
    return daily_returns.std() * np.sqrt(252)

def sharpe(daily_returns, rf=0.0):
    v = ann_vol(daily_returns)
    return (ann_return(daily_returns) - rf) / v if v > 0 else 0.0

def max_dd(return_series):
    growth = (1 + return_series).cumprod()
    peak = growth.cummax()
    return (growth / peak - 1).min()

def compute_verdict(ret, prices_f, vix_threshold):
    btc_spy_corr = ret[["BTC", "SPY"]].corr().loc["BTC", "SPY"]
    btc_gold_corr = ret[["BTC", "Gold"]].corr().loc["BTC", "Gold"]

    vix_levels = prices_f["VIX"]
    risk_off = ret[vix_levels > vix_threshold]
    risk_on = ret[vix_levels <= vix_threshold]

    btc_ro_ret = ann_return(risk_off["BTC"]) if len(risk_off) > 10 else None
    gold_ro_ret = ann_return(risk_off["Gold"]) if len(risk_off) > 10 else None
    btc_ron_ret = ann_return(risk_on["BTC"]) if len(risk_on) > 10 else None

    # Classify role
    if btc_spy_corr > 0.35 and (btc_ro_ret is None or btc_ro_ret < 0):
        role = "Speculative Growth Asset"
        color = "🔴"
        summary = (
            "Bitcoin is **highly correlated with equities** and underperforms "
            "during market stress — behavior consistent with a high-beta risk asset, "
            "not a safe haven."
        )
    elif btc_spy_corr < 0.15:
        role = "Uncorrelated Diversifier"
        color = "🟢"
        summary = (
            "Bitcoin shows **low correlation with equities** in this period, "
            "offering genuine diversification. Its portfolio value is strongest here."
        )
    else:
        role = "High-Beta Diversifier"
        color = "🟡"
        summary = (
            "Bitcoin offers **partial diversification** but remains correlated with equities. "
            "Small allocations improve Sharpe ratio; large allocations amplify drawdowns."
        )

    return {
        "role": role,
        "color": color,
        "summary": summary,
        "btc_spy_corr": btc_spy_corr,
        "btc_gold_corr": btc_gold_corr,
        "btc_risk_off_ret": btc_ro_ret,
        "gold_risk_off_ret": gold_ro_ret,
        "btc_risk_on_ret": btc_ron_ret,
        "risk_off_days": len(risk_off),
        "risk_on_days": len(risk_on),
    }

def show_digital_gold_tests():
    # Title and Desc
    st.title("₿ Bitcoin's Portfolio Role")
    st.markdown("### If Bitcoin is not yet digital gold, what role does it actually play in a portfolio?")
    
    prices, returns = load_data()

    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Controls")

    min_date = prices.index.min().date()
    max_date = prices.index.max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="dgt_date_range",
    )

    vix_threshold = st.sidebar.slider(
        "VIX Stress Threshold",
        min_value=10,
        max_value=60,
        value=20,
        step=1,
        help=("Days with VIX above this level are classified as 'Risk-Off' / market stress. "
        "Adjust to see how Bitcoin's role changes under different stress definitions."),
        key="dgt_vix",
    )

    # Filter data
    if len(date_range) == 2:
        start, end = str(date_range[0]), str(date_range[1])
    else:
        start, end = str(min_date), str(max_date)
    
    prices_f = prices.loc[start:end]
    ret_f = returns.loc[start:end]

    if len(ret_f) < 60:
        st.warning("Please select a wider date range (at least 3 monthts).")
        return
    
    # Live verdict
    v = compute_verdict(ret_f, prices_f, vix_threshold)
    st.info(
        f"{v['color']} **Portfolio Role: {v['role']}**\n\n"
        f"{v['summary']}\n\n"
        f"BTC-SPY Correlation: **{v['btc_spy_corr']:.3f}** &nbsp;|nbsp; "
        f"BTC-Gold Correlation: **{v['btc_gold_corr']:.3f}** &nbsp;|&nbsp; "
        f"Stress Days (VIX>{vix_threshold}): **{v['risk_off_days']}** of {len(ret_f)} trading days"
    )

    st.divider()

    # Section 1 - Rolling 1-year correlation
    st.subheader("Rolling 1-Year Correlation")
    st.caption("How Bitcoin's relationship with equities and gold has shifted over time." \
    "Higher BTC-SPY correlation means Bitcoin behaves more like a risk asset.")

    roll = 252
    roll_btc_spy = ret_f["BTC"].rolling(roll).corr(ret_f["SPY"])
    roll_btc_gold = ret_f["BTC"].rolling(roll).corr(ret_f["Gold"])

    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=roll_btc_spy.index, y=roll_btc_spy,
        name="BTC vs SPY", line=dict(color="#EF4444", width=2.5),
    ))
    fig_corr.add_trace(go.Scatter(
        x=roll_btc_gold.index, y=roll_btc_gold,
        name="BTC vs Gold", line=dict(color="#F59E0B", width=2.5),
    ))
    fig_corr.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig_corr.add_hline(
        y=0.3, line_dash="dot", line_color="#EF4444", opacity=0.5,
        annotation_text="Low Correlation (0.30)", annotation_position="top left",
    )
    fig_corr.add_hline(
        y=0.7, line_dash="dot", line_color="#EF4444", opacity=0.5,
        annotation_text="High Correlation (0.70)", annotation_position="top left",
    )
    # Shade post-2020 era if in view
    post_2020_start = "2020-01-01"
    if pd.Timestamp(post_2020_start) <= prices_f.index.max():
        shade_start = max(pd.Timestamp(post_2020_start), prices_f.index.min())
        fig_corr.add_vrect(
            x0=str(shade_start.date()), x1=end,
            fillcolor="rgba(99,102,241,0.07)", line_width=0,
            annotation_text="Post-2020 Institutional Era",
            annotation_position="top left",
        )
    
    fig_corr.update_layout(
        title="252-Day Rolling Correlation: Bitcoin vs SPY and Gold",
        xaxis_title="Date", yaxis_title="Pearson Correlation",
        height=420, template="plotly_white", hovermode="x unified",
        yaxis=dict(range=[-0.55, 0.85]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # Section 2 - Allocation sensitvity table (7 levels)
    st.subheader("Allocation Sensitivity Table")
    st.caption("Impact of adding Bitcoin to a 60/40 (SPY/Gold) portfolio across 7 allocation levels. " \
    "BTC weight is taken proportionally from both SPY and Gold")

    ALLOCATIONS = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]

    rows = []
    for btc_w in ALLOCATIONS:
        spy_w = 0.60 * (1 - btc_w)
        gold_w = 0.40 * (1 - btc_w)
        port_ret = (
            ret_f["BTC"] * btc_w
            + ret_f["SPY"] * spy_w
            + ret_f["Gold"] * gold_w
        )
        rows.append({
            "BTC": f"{btc_w:.0%}",
            "SPY": f"{spy_w:.0%}",
            "Gold": f"{gold_w:.0%}",
            "Ann. Return": ann_return(port_ret),
            "Volatility": ann_vol(port_ret),
            "Sharpe Ratio": sharpe(port_ret),
            "Max Drawdown": max_dd(port_ret),
        })

    sens_df = pd.DataFrame(rows)
    best_idx = sens_df["Sharpe Ratio"].idxmax()

    display_df = sens_df.copy()
    for col in ["Ann. Return", "Volatility", "Max Drawdown"]:
        display_df[col] = display_df[col].map("{:.2%}".format)
    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].map("{:.3f}".format)

    def _highlight_best(row):
        if row.name == best_idx:
            return ["background-color: rgba(34,197,94,0.25); font-weight: bold"] * len(row)
        return [""] * len(row)
    
    st.dataframe(
        display_df.style.apply(_highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"✅ Highest Sharpe Ratio at **{sens_df.loc[best_idx, 'BTC']}** BTC allocation "
        f"({float(sens_df.loc[best_idx, 'Sharpe Ratio']) if isinstance(sens_df.loc[best_idx, 'Sharpe Ratio'], float) else sens_df['Sharpe Ratio'].max():.3f})"
    )

    st.divider()

    # Section 3 - VIX regime breakdown
    st.subheader(f"VIX Regime Breakdown (Threshold: {vix_threshold})")
    st.caption(
        "Gold historically rises during market stress. Does Bitcoin? "
        "Move the VIX slider to change the stress definition."   
    )

    vix_levels = prices_f["VIX"]
    ron_ret = ret_f[vix_levels <= vix_threshold]
    roff_ret = ret_f[vix_levels > vix_threshold]

    def _regime_table(regime_ret):
        if len(regime_ret) < 10:
            return None
        return pd.DataFrame({
            "Asset": ["BTC", "Gold", "SPY"],
            "Ann. Return": [
                f"{ann_return(regime_ret['BTC']):.1%}",
                f"{ann_return(regime_ret['Gold']):.1%}",
                f"{ann_return(regime_ret['SPY']):.1%}",
            ],
            "Volatility": [
                f"{ann_vol(regime_ret['BTC']):.1%}",
                f"{ann_vol(regime_ret['Gold']):.1%}",
                f"{ann_vol(regime_ret['SPY']):.1%}",
            ],
            "Sharpe": [
                f"{sharpe(regime_ret['BTC']):.2f}",
                f"{sharpe(regime_ret['Gold']):.2f}",
                f"{sharpe(regime_ret['SPY']):.2f}",
            ],
        })

    col_on, col_off = st.columns(2)

    with col_on:
        st.markdown(f"**Risk-On (VIX ≤ {vix_threshold})** - {len(ron_ret)} days")
        tbl = _regime_table(ron_ret)
        if tbl is not None:
            st.dataframe(tbl, hide_index=True, use_container_width=True)
        else:
            st.info("Insufficient data.")
    
    with col_off:
        st.markdown(f"**Risk-Off (VIX > {vix_threshold})** - {len(roff_ret)} days")
        tbl = _regime_table(roff_ret)
        if tbl is not None:
            st.dataframe(tbl, hide_index=True, use_container_width=True)
        else:
            st.info("No days above this VIX threshold in the selected period.")
    
    # VIX regime bar chart
    if len(ron_ret) > 10 and len(roff_ret) > 10:
        fig_vix = go.Figure()
        regimes=[f"Risk-On (VIX≤{vix_threshold})", f"Risk-Off (VIX>{vix_threshold})"]
        assets = ["BTC", "Gold", "SPY"]
        colors = {"BTC": "#F7931A", "Gold": "#EAB308", "SPY": "#3B82F6"}
        
        for asset in assets:
            fig_vix.add_trace(go.Bar(
                name=asset,
                x=regimes,
                y=[
                    ann_return(ron_ret[asset]),
                    ann_return(roff_ret[asset]),
                ],
                marker_color=colors[asset],
            ))

        fig_vix.add_hline(y=0, line_color="black", line_width=1)
        fig_vix.update_layout(
            title="Annualized Returns by VIX Regime",
            yaxis_title="Annualized Return",
            yaxis_tickformat=".0%",
            barmode="group",
            height=380,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_vix, use_container_width=True)

    st.divider()

    # Section 4 - Quarterly color-coded table
    st.subheader("Quarterly Returns - Color-Coded Detail")
    st.caption("Green = positive quarter, Red = negative. Shade intensity reflects magnitude")

    q_prices = prices_f[["BTC", "Gold", "SPY"]].resample("QE").last()
    q_ret = q_prices.pct_change().dropna()
    q_ret.index = q_ret.index.to_period("Q").astype(str)

    def _color_cell(val):
        if not isinstance(val, (int, float)) or np.isnan(val):
            return ""
        if val >= 0:
            alpha = min(val * 2.5, 0.8)
            return f"background-color: rgba(34,197,94, {alpha:.2f}); color: {'white' if alpha > 0.5 else 'black'}"
        else:
            alpha = min(abs(val) * 2.5, 0.8)
            return f"background-color: rgba(239,68,68, {alpha:.2f}); color: {'white' if alpha >0.5 else 'black'}"
        
    st.dataframe(
        q_ret.style.format("{:.1%}").map(_color_cell),
        use_container_width=True,
        height=min(35 * len(q_ret) + 38, 600),
    )

    st.divider()

    #Section 5 - Downloads
    st.subheader("Export")
    col_csv, col_report = st.columns(2)
    
    with col_csv:
        export_df = ret_f[["BTC", "Gold", "SPY"]].copy()
        export_df.index = export_df.index.strftime("%Y-%m-%d")
        export_df.index.name = "Date"
        st.download_button(
            label="⬇ Download Returns CSV",
            data=export_df.to_csv(),
            file_name=f"btc_returns_{start}_{end}.csv",
            mime="text/csv",
        )
    
    with col_report:
        report = _build_report(v, ret_f, prices_f, sens_df, best_idx, start, end, vix_threshold)
        st.download_button(
            label="⬇ Download Summary Report",
            data=report,
            file_name=f"bitcoin_portfolio_role_{start}.txt",
            mime="text/plain",
        )

# Build Report Function
def _build_report(v, ret_f, prices_f, sens_df, best_idx, start, end, vix_threshold):
        btc_r = ann_return(ret_f["BTC"])
        gold_r = ann_return(ret_f["Gold"])
        spy_r = ann_return(ret_f["SPY"])
        btc_v = ann_vol(ret_f["BTC"])
        gold_v = ann_vol(ret_f["Gold"])
        spy_v = ann_vol(ret_f["SPY"])
        btc_s = sharpe(ret_f["BTC"])
        gold_s = sharpe(ret_f["Gold"])
        spy_s = sharpe(ret_f["SPY"])
        btc_d = max_dd(ret_f["BTC"])
        gold_d = max_dd(ret_f["Gold"])
        spy_d = max_dd(ret_f["SPY"])

        best_alloc = sens_df.loc[best_idx, "BTC"]
        best_sharpe_val = sens_df["Sharpe Ratio"].max()

        lines = [
            "BITCOIN PORTFOLIO ROLE — ANALYSIS REPORT",
            "=" * 50,
            f"Period:              {start} to {end}",
            f"VIX Stress Threshold: {vix_threshold}",
            f"Trading Days:        {len(ret_f)}",
            "",
            "VERDICT",
            "-" * 30,
            f"Portfolio Role:      {v['role']}",
            f"BTC–SPY Correlation: {v['btc_spy_corr']:.4f}",
            f"BTC–Gold Correlation:{v['btc_gold_corr']:.4f}",
            f"Stress Days (VIX>{vix_threshold}): {v['risk_off_days']}",
            "",
            "INDIVIDUAL ASSET METRICS",
            "-" * 30,
            f"{'Metric':<22} {'BTC':>10} {'Gold':>10} {'SPY':>10}",
            f"{'Ann. Return':<22} {btc_r:>10.2%} {gold_r:>10.2%} {spy_r:>10.2%}",
            f"{'Volatility':<22} {btc_v:>10.2%} {gold_v:>10.2%} {spy_v:>10.2%}",
            f"{'Sharpe Ratio':<22} {btc_s:>10.3f} {gold_s:>10.3f} {spy_s:>10.3f}",
            f"{'Max Drawdown':<22} {btc_d:>10.2%} {gold_d:>10.2%} {spy_d:>10.2%}",
            "",
            "PORTFOLIO ALLOCATION SENSITIVITY",
            "-" * 30,
            f"{'BTC Alloc':<12} {'Ann. Return':>13} {'Volatility':>12} {'Sharpe':>10} {'Max DD':>10}",
        ]

        ALLOCATIONS = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
        for btc_w in ALLOCATIONS:
            spy_w = 0.60 * (1 - btc_w)
            gold_w = 0.40 * (1 - btc_w)
            pr = ret_f["BTC"] * btc_w + ret_f["SPY"] * spy_w + ret_f["Gold"] * gold_w
            marker = " ← Optimal Sharpe" if abs(btc_w - float(best_alloc.strip("%")) / 100) < 0.001 else""
            lines.append(
                f"{btc_w:<12.0%} {ann_return(pr):>13.2%} {ann_vol(pr):>12.2%} "
                f"{sharpe(pr):>10.3f} {max_dd(pr):>10.2%}{marker}"
            )
        
        if v["btc_risk_off_ret"] is not None:
            lines += [
                "",
                "VIX REGIME ANALYSIS",
                "-" * 30,
                f"Risk-On Days (VIX≤{vix_threshold}):  {v['risk_on_days']}",
                f"Risk-Off Days (VIX>{vix_threshold}): {v['risk_off_days']}",
                f"BTC Ann. Return (Risk-Off):  {v['btc_risk_off_ret']:.2%}",
                f"Gold Ann. Return (Risk-Off): {v['gold_risk_off_ret']:.2%}",
            ]
        
        lines += [
            "",
            "CONCLUSION",
            "-" * 30,
            "Bitcoin is not yet digtial gold. In the selected period, Bitcoin demonstrates",
            f"characteristics of a {v['role'].lower()} rather than a safe-haven",
            "store of value. Its strongest case for portfolio inclusion lies in",
            "diversification and assymetric return potential, not inflation protection.",
            f"Optimal BTC allocation by Sharpe ratio: {best_alloc} ({best_sharpe_val:.3f} Sharpe).",
            "",
            "-" * 50,
            "Bitcoin as Digital gold: Hedge or Hype?",
            "Deutsche Bank Intern Case Study",
            "Alexander Dominguez Zhakav - 2026"
        ]

        return "\n".join(lines)