import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import time
import math

try:
    import monte_carlo as _mc_cpp
    _CPP = True
except ImportError:
    _CPP = False

TRADING_DAYS = 252          # annualisation constant — always 252
SIM_DAYS     = 504          # simulation horizon: 2 years (2 × 252)
RF_ANNUAL    = 0.04

# Path-dependent simulation parameters (shared by Python and C++)
N_MC_PATHS  = 2       # paths per portfolio weight for averaging
REBAL_FREQ  = 63      # quarterly rebalance check (trading days)
DRIFT_LIMIT = 0.05    # rebalance if any weight drifts > 5%
TXCOST      = 0.001   # 0.1% one-way transaction cost on rebalance
DD_TRIGGER  = -0.30   # drawdown level that trips BTC circuit breaker
DD_HALFLIFE = 21      # days BTC remains reduced after circuit breaker trips
BTC_COL_DEFAULT = 0   # BTC is index 0 in the Mode 1 universe

ASSET_COLORS = {
    "BTC": "#F7931A", 
    "Gold": "#EAB308",
    "GLD": "#EAB308",
    "SPY": "#3B82F6",
    "QQQ": "#8B5CF6",
    "TLT": "#10B981"
}

# Section 1 - Data loading
@st.cache_data
def load_csv_universe() -> pd.DataFrame:
    """Mode 1 uses the pre-fetched CSV so the app never blocks on a network call."""
    prices = pd.read_csv("data/asset_prices.csv", index_col=0, parse_dates=True)
    return prices[["BTC", "Gold", "SPY", "QQQ", "TLT"]]

@st.cache_data
def load_csv_full() -> pd.DataFrame:
    """Full CSV including VIX - used for regime detection in Mode 1."""
    return pd.read_csv("data/asset_prices.csv", index_col=0, parse_dates=True)

@st.cache_data(ttl=3600)
def load_yfinance(tickers: tuple, period: str="5y") -> pd.DataFrame:
    """
    Mode 2 fetches live price history.
    Tickers is a tuple (not a list) because st.cache_data requires hashable args.
    """
    raw = yf.download(
        list(tickers),
        period=period,
        auto_adjust=True,
        progress=False
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Remove completely failed tickers
    prices = prices.dropna(axis=1, how="all")

    # Fill small missing gaps
    prices = prices.ffill().dropna()

    return prices

# Section 2 - Statistical Calibration
def calibrate(prices: pd.DataFrame):
    """
    Compute mu (expected return vector) and Sigma (covariance matrix) from historical daily LOG returns.
    WHY LOG RETURNS?
    Simple returns: r = (P_t - P_{t-1}) / P_{t-1}
    Log returns: r = ln(P_t / P_{t-1})
    Log returns are preferred because:
    1. TIME-ADDITIVE - a multi-day log return is the sum of daily log returns. 
    This makes annualization trivial: just multiply by 252.
    2. APPROXIMATELY NORMAL - required assumption for GBM.
    3. BOUNDED CORRECTLY - you cannot lose more than 100%, and log(0) = -inf 
    captures this naturally without needing special case handling.

    The covariance matrix Sigma captures how assets move TOGETHER:
        - Diagonal entries = each asset's own variance (sigma_i^2)
        - Off-diagonal = how much two assets co-move (positive = correlated, zero = independent, negative = hedge each other)
    This is crucial for diversification math. A portfolio of two perfectly correlated assets offers no risk reduction.
    Sigma quantifies that
    """
    log_ret = np.log(prices/ prices.shift(1)).dropna()
    mu = log_ret.mean().values
    cov = log_ret.cov().values
    return log_ret, mu, cov

def cholesky_factor(cov: np.ndarray) -> np.ndarray:
    """
    Factor the covariance matrix as Sigma = L @ L.T (Cholesky decomposition).
    WHY CHOLESKY?
    We need CORRELATED random returns for multiple assets simultaneously.
    If we sampled Z_i~N(0, sigma_i^2) independently, we would ignore the fact that BTC and SPY have started moving together post-2020.

    The Cholesky trick:
        1. Draw Z~N(0,1)
        2. Compute eps = L @ Z
        3. Result: E[eps @ eps.T] = L @ I @ L.T = Sigma checkmark
    
    L "colors" white notise into correlated asset returns.
    It's the multivariable generalization of multiplying a standard normal by a standard deviation to get the right variance.

    NUMERICAL NOTE:
    We add 1e-8 (tiny ridge) before decomposing. Without it, a nearly
    -singular covariance matrix (e.g., two nearly-identical ETFs) would cause
    np.linalg.cholesky to fail with a LinAlgError.
    """
    n = cov.shape[0]
    return np.linalg.cholesky(cov + 1e-8 * np.eye(n))

# Section 3 - Weight Sampling
def sample_weights(
        n_assets: int,
        n_portfolios: int,
        btc_idx: int = 0,
        btc_min: float = 0.0,
        btc_max: float = 0.35
) -> np.ndarray:
    """
    Sample random portfolio weight vectors uniformly on the probability simplex (the set of all non-negative weights that sum to 1).
    
    WHY EXPONENTIAL SAMPLING (NOT UNIFORM)?
    the naive method - sample w_i ~ Uniform(0,1) and normalize - biases the distribution toward
    equal-weight portfolios near the center of the simplex.
    Extreme allocations (e.g., 90% in one assset) are severly undersmapled.

    The CORRECT method (uniform on the simplex):
        w_i ~ Exponential(1), then w_i /= sum(w_j)
    This is the Dirichlet(1,1,...,1) distribution — the only distribution that
    weights every point on the simplex equally. It gives the Monte Carlo a fair,
    unbiased scan of ALL possible portfolios including concentrated ones.

    BTC CONSTRAINTS:
    After sampling, we clip BTC to [btc_min, btc_max] then re-normalize
    the remaining assets proportionally so weights still sum to 1.
    """
    rng = np.random.default_rng()
    raw = rng.exponential(1.0, size=(n_portfolios, n_assets))
    weights = raw / raw.sum(axis=1, keepdims=True)

    weights[:, btc_idx] = np.clip(weights[:, btc_idx], btc_min, btc_max)

    other_idx = [i for i in range(n_assets) if i != btc_idx]
    other_sum = weights[:, other_idx].sum(axis=1, keepdims=True)
    other_sum = np.where(other_sum == 0, 1.0, other_sum)
    weights[:, other_idx] *= (1.0 - weights[:, btc_idx:btc_idx+1]) / other_sum

    return weights.astype(np.float64)

# Section 4 - Monte Carlo Simulation (Python fallback)
def _gbm_paths(mu: np.ndarray, L: np.ndarray, weights: np.ndarray, n_days: int) -> np.ndarray:
    """
    Path-dependent GBM simulation — Python reference implementation (intentionally slow).

    THE GBM MODEL:
    For a single asset: dS = mu*S*dt + sigma*S*dW

    Discrete daily form:
        log(S_{t+1} / S_t)  =  mu_i  +  eps_i
    where eps_i is correlated noise from the Cholesky step.

    KEY POINT — NO ITO CORRECTION NEEDED:
    mu is estimated from log returns, so the correction is already embedded
    in the sample mean. Adding it again would double-count and bias results.

    PORTFOLIO RETURN CALCULATION (per step):
        asset_ret_i  = expm1(mu_i + eps_i)      ← arithmetic return
        port_ret     = Σ w_i * asset_ret_i       ← weighted sum
        port_val[t+1]= port_val[t] * (1 + port_ret)

    WHY THIS FUNCTION IS SLOW (and why that justifies C++):
    This simulation adds three path-dependent features that require
    per-portfolio, per-timestep state tracking. Because each portfolio's
    next step depends on its own history, we cannot batch-vectorize
    across portfolios — we must use a Python loop per portfolio per day.

    ── FEATURE 1: QUARTERLY DRIFT REBALANCING WITH TRANSACTION COSTS ────────
    Every REBAL_FREQ (63) trading days, check if any weight has drifted more
    than DRIFT_LIMIT (5%) from the target due to price changes. If so, rebalance
    and apply TXCOST (0.1%) proportional to turnover. Tracks: days_rebal,
    curr_w (which evolves differently for every portfolio).

    ── FEATURE 2: DRAWDOWN CIRCUIT BREAKER ──────────────────────────────────
    If the portfolio falls more than DD_TRIGGER (-30%) from its running peak,
    halve BTC exposure for the next DD_HALFLIFE (21) trading days, redistributing
    the excess proportionally to other assets. Tracks: peak, dd_days_left.
    This models a real institutional risk-management rule, and connects directly
    to Test 3 findings: BTC drops ~69% annualised during risk-off regimes.

    ── FEATURE 3: MULTI-PATH AVERAGING (N_MC_PATHS per weight vector) ───────
    Each weight vector is simulated N_MC_PATHS times with different random seeds.
    The portfolio VALUE PATH is averaged across runs before computing metrics.
    This is the "Monte Carlo within Monte Carlo" technique: the outer loop
    explores allocations, the inner N_MC_PATHS loop estimates each allocation's
    expected trajectory more reliably (lower noise, cleaner efficient frontier).

    TOTAL WORK: n_portfolios × N_MC_PATHS × n_days Python iterations.
    With n_portfolios=3000, N_MC_PATHS=2, n_days=504 ≈ 3M iterations at
    ~12μs each → ~35–45 seconds in Python.

    C++ eliminates the interpreter overhead and parallelises across portfolios
    with OpenMP. Expected speedup: 20–50×.

    Returns shape (n_portfolios, n_days + 1).
    """
    n_portfolios, n_assets = weights.shape
    port_values = np.ones((n_portfolios, n_days + 1))

    # Pre-convert to Python lists — avoids per-element NumPy indexing overhead
    L_py  = L.tolist()
    mu_py = mu.tolist()
    rng   = np.random.default_rng(42)

    for p in range(n_portfolios):
        target_w = weights[p].tolist()
        path_sum = [0.0] * (n_days + 1)

        for _ in range(N_MC_PATHS):
            curr_w       = list(target_w)
            val          = 1.0
            peak         = 1.0
            days_rebal   = 0
            dd_days_left = 0
            path_mc      = [0.0] * (n_days + 1)
            path_mc[0]   = 1.0

            for t in range(n_days):

                # ── GBM step: draw correlated log returns ──────────────────
                # Single batch draw (one fast NumPy call for all n_assets)
                z = rng.standard_normal(n_assets)

                port_ret     = 0.0
                asset_ret_py = []
                for i in range(n_assets):
                    # Cholesky multiply: eps_i = Σ_{j≤i} L[i][j] * z[j]
                    eps_i = 0.0
                    for j in range(i + 1):
                        eps_i += L_py[i][j] * z[j]

                    # Arithmetic return from log return (math.expm1 is numerically
                    # stable near zero: avoids catastrophic cancellation in exp(x)-1)
                    ar = math.expm1(mu_py[i] + eps_i)
                    asset_ret_py.append(ar)
                    port_ret += curr_w[i] * ar

                val          = val * (1.0 + port_ret)
                path_mc[t + 1] = val

                # ── Update drifted weights after price changes ─────────────
                total_val = 0.0
                for i in range(n_assets):
                    total_val += curr_w[i] * (1.0 + asset_ret_py[i])
                if total_val > 0.0:
                    for i in range(n_assets):
                        curr_w[i] = curr_w[i] * (1.0 + asset_ret_py[i]) / total_val

                # ── Track peak; check drawdown circuit breaker ─────────────
                if val > peak:
                    peak = val
                drawdown = val / peak - 1.0

                if drawdown < DD_TRIGGER and dd_days_left == 0:
                    dd_days_left = DD_HALFLIFE

                if dd_days_left > 0:
                    btc_excess = curr_w[BTC_COL_DEFAULT] * 0.5
                    curr_w[BTC_COL_DEFAULT] -= btc_excess
                    other_sum = sum(curr_w[i] for i in range(n_assets) if i != BTC_COL_DEFAULT)
                    if other_sum > 0.0:
                        for i in range(n_assets):
                            if i != BTC_COL_DEFAULT:
                                curr_w[i] += btc_excess * (curr_w[i] / other_sum)
                    dd_days_left -= 1

                # ── Quarterly drift rebalancing ────────────────────────────
                days_rebal += 1
                if days_rebal >= REBAL_FREQ:
                    max_drift = max(abs(curr_w[i] - target_w[i]) for i in range(n_assets))
                    if max_drift > DRIFT_LIMIT:
                        turnover = sum(abs(curr_w[i] - target_w[i]) for i in range(n_assets))
                        val         *= (1.0 - turnover * TXCOST)
                        path_mc[t + 1] = val
                        curr_w = list(target_w)
                    days_rebal = 0

            for t in range(n_days + 1):
                path_sum[t] += path_mc[t]

        for t in range(n_days + 1):
            port_values[p, t] = path_sum[t] / N_MC_PATHS

    return port_values

def __bootstrap_paths(log_ret, weights, n_days) -> np.ndarray:
    """
    Historical Bootstrap — Python/NumPy fallback.

    WHAT BOOTSTRAP DOES:
    Instead of fitting a parametric model (GBM), we resample actual observed
    returns WITH REPLACEMENT. Each simulated "day" is a random historical day.

    THE KEY IMPLEMENTATION DETAIL:
    We sample ENTIRE ROWS of the returns matrix (one row = one calendar day,
    all assets). This is what preserves cross-asset correlations automatically.
    We never need to estimate or simulate correlations separately — they are
    baked into the historical data.

    WHEN TO PREFER BOOTSTRAP OVER GBM:
    - When asset returns have fat tails (BTC has extremely fat tails)
    - When you want crash scenarios from actual history to appear
    - When you're skeptical of the normality assumption in GBM
    - As a robustness check against GBM results

    Returns shape (n_portfolios, n_days + 1).
    """
    ret_matrix = log_ret.values
    T = ret_matrix.shape[0]
    n_portfolios = weights.shape[0]
    rng = np.random.default_rng(42)
    port_values = np.ones((n_portfolios, n_days + 1))

    for t in range(n_days):
        row = ret_matrix[rng.integers(0, T)]
        day_ret = np.expm1(row)
        port_ret = weights @ day_ret
        port_values[:, t + 1] = port_values[:, t] * (1.0 + port_ret)

    return port_values

def _regime_paths(log_ret, prices_full, weights, n_days) -> np.ndarray:
    """
    Regime-Weighted Bootstrap.

    MOTIVATION FROM THE RESEARCH (Test 3):
    Bitcoin's behavior differs dramatically by VIX regime:
      - Risk-On  (VIX < 20):  BTC averages +138% annualized
      - Risk-Off (VIX >= 20): BTC averages -69% annualized
    A plain bootstrap ignores current market conditions. If we're currently
    in a Risk-Off regime, a naive bootstrap gives an over-optimistic outlook.

    THE 70/30 WEIGHTING:
    Volatility regimes PERSIST (volatility clustering). Once VIX spikes,
    it tends to stay elevated. The 70/30 split means: sample 70% from the
    current regime, 30% from the other. This captures regime persistence
    without ignoring the possibility of regime change.

    Falls back to plain bootstrap if VIX data unavailable (Mode 2).
    """
    if "VIX" not in prices_full.columns:
        return __bootstrap_paths(log_ret, weights, n_days)
    
    vix = prices_full["VIX"].reindex(log_ret.index).values
    ret_matrix = log_ret.values
    T = ret_matrix.shape[0]

    risk_on_idx = np.where(vix < 20)[0]
    risk_off_idx = np.where(vix >= 20)[0]

    current_vix = prices_full["VIX"].iloc[-1]
    in_risk_off = bool(current_vix >= 20)

    n_portfolios = weights.shape[0]
    rng = np.random.default_rng(42)
    port_values = np.ones((n_portfolios, n_days + 1))

    for t in range(n_days):
        use_risk_off = in_risk_off if rng.random() < 0.70 else (not in_risk_off)
        pool = risk_off_idx if (use_risk_off and len(risk_off_idx) > 0) else \
            risk_on_idx if (not use_risk_off and len(risk_on_idx) > 0) else \
            np.arange(T)
        row = ret_matrix[rng.choice(pool)]
        day_ret = np.expm1(row)
        port_ret = weights @ day_ret
        port_values[:, t + 1] = port_values[:, t] * (1.0 + port_ret)
    
    return port_values

# Section 5 - Portfolio Metrics
def compute_metrics(port_values: np.ndarray, rf: float = RF_ANNUAL) -> np.ndarray:
    """
    Compute annualized metrics for every simulated portfolio path.

    Input:  (n_portfolios, n_days + 1)
    Output: (n_portfolios, 5)  ->  [ann_return, ann_vol, sharpe, max_dd, var95]

    ANNUALIZED RETURN:
        R_ann = (P_final / P_start) ^ (252 / N) - 1
    Geometric annualization converts any holding period to a per-year rate.

    ANNUALIZED VOLATILITY:
        sigma_ann = std(daily_returns) * sqrt(252)
    The sqrt(252) factor converts daily std to annual assuming IID returns.
    This is the industry-standard approximation.

    SHARPE RATIO:
        Sharpe = (R_ann - R_f) / sigma_ann
    The single most important metric for comparing portfolios.
    It answers: "How much return am I earning per unit of risk taken?"
    Using R_f = 4% reflects current T-bill rates. Portfolios that cannot
    beat the risk-free rate with their Sharpe (Sharpe < 0) are not worth
    taking the volatility risk for.

    MAX DRAWDOWN:
        MDD = min_t( P(t) / max_{s<=t} P(s)  -  1 )
    Measures the worst peak-to-trough loss in the simulated path.
    This is the key metric for risk-tolerant vs risk-averse investors.
    A -50% drawdown means you watch your portfolio halve before recovering.

    VAR95 (Value at Risk at 95% confidence):
        VaR = 5th percentile of daily returns * sqrt(252)
    Annualized VaR from daily return distribution of the simulated path.
    Interpretation: 95% of trading days, your daily loss won't exceed |VaR|.
    """
    n_days = port_values.shape[1] - 1
    daily = port_values[:, 1:] / port_values[:, :-1] - 1
    ann_ret = (port_values[:, -1] / port_values[:, 0]) ** (252.0 / n_days) - 1
    ann_vol = daily.std(axis=1) * np.sqrt(252)
    sharpe   = np.where(ann_vol > 0, (ann_ret - rf) / ann_vol, 0.0)
    peak = np.maximum.accumulate(port_values, axis=1)
    max_dd = (port_values / peak - 1).min(axis=1)
    var95 = np.percentile(daily, 5, axis=1) * np.sqrt(252)

    return np.column_stack([ann_ret, ann_vol, sharpe, max_dd, var95])

# Section 6 - Optimization
def find_optimal(weights, metrics, risk_profile, max_dd_custom, btc_col) -> dict:
    """
    Select the best portfolio from all Monte Carlo simulations.

    THE EFFICIENT FRONTIER CONCEPT:
    Among our 1,000+ simulated portfolios, many are "dominated" — there exists
    another portfolio with higher return AND lower volatility. The set of
    non-dominated portfolios forms the efficient frontier.

    Our approach approximates this frontier by Monte Carlo sampling rather than
    solving the exact Markowitz quadratic program. The advantage: we can easily
    add non-linear constraints (max drawdown, BTC bounds) that are difficult in
    standard mean-variance optimization.

    RISK PROFILE LOGIC:
    - Conservative : maximize Sharpe, but only consider portfolios where the
                     simulated max drawdown never exceeded 20%.
    - Moderate     : maximize Sharpe with no drawdown filter.
    - Aggressive   : maximize raw annualized return (ignore volatility).
    - Custom       : user picks their own drawdown limit.

    If the drawdown filter removes too many portfolios (< 10 remain), we
    automatically relax it rather than returning a degenerate result.
    """
    ann_ret, ann_vol, sharpe, max_dd = metrics[:, 0], metrics[:, 1], metrics[:, 2], metrics[:, 3]
    if risk_profile == "Conservative":
        mask = max_dd > -0.20
    elif risk_profile == "Custom":
        mask = max_dd > max_dd_custom
    else:
        mask = np.ones(len(sharpe), dtype=bool)

    if mask.sum() < 10:
        mask = np.ones(len(sharpe), dtype=bool)
    
    score = np.where(mask, ann_ret if risk_profile == "Aggressive" else sharpe, -np.inf)
    best_idx = int(np.argmax(score))

    return {
        "idx": best_idx,
        "weights": weights[best_idx],
        "ann_return": float(metrics[best_idx, 0]),
        "ann_vol": float(metrics[best_idx, 1]),
        "sharpe": float(metrics[best_idx, 2]),
        "max_drawdown": float(metrics[best_idx, 3]),
        "var95": float(metrics[best_idx, 4]),
        "btc_alloc": float(weights[best_idx, btc_col]),
    }

# Section 7 - Fan Chart Data
def fan_percentiles(port_values: np.ndarray, investment: float) -> dict:
    """
    Compute percentile bands for the fan chart.

    port_values: (n_fan, n_days+1) — all rows use the SAME optimal weights.
    Each row is one independent simulation of that specific portfolio.

    WHY A FAN CHART?
    A single number like "expected return = 38%" is misleading. It hides
    the enormous range of possible outcomes, especially for BTC-heavy portfolios.

    The fan chart shows three bands:
      - 5th–95th: full range of realistic outcomes (95% of simulations land here)
      - 25th–75th: the "likely" zone (half of all outcomes)
      - 50th (median): the single most probable trajectory

    This is how institutional risk desks communicate uncertainty — not as
    a point estimate but as a distribution of possible futures.

    We run these as a DEDICATED batch of simulations all using fixed optimal
    weights. This isolates uncertainty from RETURNS (noise in the market) vs
    uncertainty in ALLOCATION (which we've already solved for).
    """
    scaled = port_values * investment
    return {
        "p5": np.percentile(scaled, 5, axis=0),
        "p25": np.percentile(scaled, 25, axis=0),
        "p50": np.percentile(scaled, 50, axis=0),
        "p75": np.percentile(scaled, 75, axis=0),
        "p95": np.percentile(scaled, 95, axis=0),
    }

# Section 8 - Pipeline Runner
def _run_pipeline(prices, prices_full, n_portfolios, forecast_method,
                  btc_idx, btc_min, btc_max, rf, risk_profile, max_dd_custom,
                  investment, n_fan=300):
    """Full end-to-end pipeline from raw prices to optimal portfolio + fan chart."""
    n_assets = prices.shape[1]
    log_ret, mu, cov = calibrate(prices)
    L = cholesky_factor(cov)
    weights = sample_weights(n_assets, n_portfolios, btc_idx, btc_min, btc_max)

    t0 = time.perf_counter()
    use_cpp = _CPP and forecast_method == "GBM (Geometric Brownian Motion)"

    if use_cpp:
        res = _mc_cpp.run_monte_carlo(
            mu, L, weights, SIM_DAYS, rf / TRADING_DAYS,
            N_MC_PATHS, float(REBAL_FREQ), DRIFT_LIMIT, TXCOST, DD_TRIGGER, DD_HALFLIFE,
        )
        port_values = res["paths"]
        metrics = res["metrics"]
    else:
        if forecast_method == "GBM (Geometric Brownian Motion)":
            port_values = _gbm_paths(mu, L, weights, SIM_DAYS)
        elif forecast_method == "Historical Bootstrap":
            port_values = __bootstrap_paths(log_ret, weights, SIM_DAYS)
        else:
            port_values = _regime_paths(log_ret, prices_full, weights, SIM_DAYS)
        metrics = compute_metrics(port_values, rf)

    elasped = time.perf_counter() - t0
    optimal = find_optimal(weights, metrics, risk_profile, max_dd_custom, btc_idx)

    # Fan chart: n_fan independent paths, all with fixed optimal weights
    fan_w = np.tile(optimal["weights"], (n_fan, 1))
    if use_cpp:
        fan_paths = _mc_cpp.run_monte_carlo(
            mu, L, fan_w, SIM_DAYS, rf / TRADING_DAYS,
            N_MC_PATHS, float(REBAL_FREQ), DRIFT_LIMIT, TXCOST, DD_TRIGGER, DD_HALFLIFE,
        )["paths"]
    elif forecast_method == "GBM (Geometric Brownian Motion)":
        fan_paths = _gbm_paths(mu, L, fan_w, SIM_DAYS)
    elif forecast_method == "Historical Bootstrap":
        fan_paths = __bootstrap_paths(log_ret, fan_w, SIM_DAYS)
    else:
        fan_paths = _regime_paths(log_ret, prices_full, fan_w, SIM_DAYS)

    return weights, metrics, optimal, fan_paths, elasped, log_ret, mu, cov

# Section 9 - Charts
def _color(name: str) -> str:
    for key, col in ASSET_COLORS.items():
        if name.upper().startswith(key):
            return col
    return "#94A3B8"

def chart_frontier(weights, metrics, opt_idx, asset_names, curr_ann_ret=None, curr_ann_vol=None):
    ann_ret = metrics[:, 0]
    ann_vol = metrics[:, 1]
    sharpe = metrics[:, 2]

    hover = [
        "<br>".join(f"{asset_names[i]}: {weights[j, i]:.1%}" for i in range(len(asset_names)))
        for j in range(len(weights))
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ann_vol, y=ann_ret, mode="markers",
        marker=dict(color=sharpe, colorscale="Viridis", size=4, opacity=0.55,
                    colorbar=dict(title="Sharpe", thickness=14, len=0.7)),
        text=hover,
        hovertemplate="<b>Allocation</b><br>%{text}<br>Return: %{y:.1%}<br>Vol: %{x:.1%}<extra></extra>",
        name="Simulated Portfolios",
    ))
    fig.add_trace(go.Scatter(
        x=[ann_vol[opt_idx]], y=[ann_ret[opt_idx]], mode="markers",
        marker=dict(symbol="star", size=20, color="#FBBF24", line=dict(color="black", width=1.5)),
        name="Optimal Portfolio",
        hovertemplate=f"<b>Optimal</b><br>Return: {ann_ret[opt_idx]:.1%}<br>Sharpe: {sharpe[opt_idx]:.3f}<extra></extra>",
    ))
    if curr_ann_ret is not None:
        fig.add_trace(go.Scatter(
            x=[curr_ann_vol], y=[curr_ann_ret], mode="markers",
            marker=dict(symbol="circle", size=16, color="#EF4444", line=dict(color="black", width=1.5)),
            name="Your Current Portfolio",
            hovertemplate=f"<b>Current</b><br>Return: {curr_ann_ret:.1%}<br>Vol: {curr_ann_vol:.1%}<extra></extra>",
        ))
    fig.update_layout(
        title="Monte Carlo Efficient Frontier",
        xaxis_title="Annualized Volatility (Risk)", yaxis_title="Annualized Return",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        height=480, template="plotly_white", hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def chart_fan(percs, investment):
    days = np.arange(len(percs["p50"]))
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([percs["p95"], percs["p5"][::-1]]),
        fill="toself", fillcolor="rgba(99,102,241,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5th – 95th pct", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([percs["p75"], percs["p25"][::-1]]),
        fill="toself", fillcolor="rgba(99,102,241,0.22)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25th – 75th pct", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=percs["p50"], mode="lines",
        line=dict(color="#6366F1", width=2.5), name="Median",
        hovertemplate="Day %{x}: $%{y:,.0f}<extra>Median</extra>",
    ))
    fig.add_hline(y=investment, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text=f"Initial ${investment:,.0f}")
    fig.update_layout(
        title="2-Year Portfolio Outcome Distribution (Monte Carlo Fan Chart)",
        xaxis_title="Trading Day", yaxis_title="Portfolio Value ($)",
        yaxis_tickformat="$,.0f",
        height=430, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def chart_donut(weights, asset_names, title, investment=None):
    colors = [_color(n) for n in asset_names]
    custom = [f"${w * investment:,.0f}" for w in weights] if investment else None
    fig = go.Figure(go.Pie(
        labels=asset_names, values=weights * 100,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+percent",
        customdata=custom,
        hovertemplate="%{label}: %{percent}" + ("<br>%{customdata}" if investment else "") + "<extra></extra>",
        hole=0.45,
    ))
    fig.update_layout(title=title, height=370,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.15))
    return fig

# Section 10 - Mode 1: Auto-Allocate
def _mode1(investment, risk_profile, forecast_method, n_portfolios, btc_min, btc_max, rf, max_dd_custom):
    prices = load_csv_universe()
    prices_full = load_csv_full()
    asset_names = list(prices.columns)
    btc_idx = asset_names.index("BTC")

    st.subheader("Asset Universe — BTC · Gold · SPY · QQQ · TLT")
    st.caption("All data from historical daily prices since 2015. Long-only, weights sum to 100%.")

    with st.spinner(f"Running {n_portfolios:,} Monte Carlo simulations..."):
        weights, metrics, optimal, fan_paths, elasped, log_ret, mu, cov = _run_pipeline(
            prices, prices_full, n_portfolios, forecast_method, btc_idx, btc_min, btc_max, rf,
            risk_profile, max_dd_custom, investment,
        )

    engine = "C++ · Pybind11" if (_CPP and forecast_method == "GBM (Geometric Brownian Motion)") else "Python · NumPy"
    st.success(f"Simulation complete — {n_portfolios:,} portfolios × {SIM_DAYS} days × {N_MC_PATHS} paths in **{elasped:.2f}s** using {engine}")

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Expected Return", f"{optimal['ann_return']:.1%}")
    c2.metric("Volatility", f"{optimal['ann_vol']:.1%}")
    c3.metric("Sharpe Ratio", f"{optimal['sharpe']:.3%}")
    c4.metric("Max Drawdown", f"{optimal['max_drawdown']:.1%}")
    c5.metric("BTC Allocation", f"{optimal['btc_alloc']:.1%}")
    st.divider()

    col_l, col_r = st.columns([6, 4])
    with col_l:
        st.plotly_chart(
            chart_frontier(weights, metrics, optimal["idx"], asset_names),
            use_container_width=True
        )
    with col_r:
        st.plotly_chart(
            chart_donut(optimal["weights"], asset_names, "Optimal Allocation", investment), 
            use_container_width=True
        )

    percs = fan_percentiles(fan_paths, investment)
    st.plotly_chart(chart_fan(percs, investment), use_container_width=True)

    st.subheader("Allocation Breakdown")
    total_years = len(prices) / TRADING_DAYS
    rows = []
    for i, name in enumerate(asset_names):
        w = float(optimal["weights"][i])
        hist_ret = (prices[name].iloc[-1] / prices[name].iloc[0]) ** (1 / total_years) - 1
        hist_vol = np.log(prices[name] / prices[name].shift(1)).dropna().std() * np.sqrt(TRADING_DAYS)
        rows.append({
            "Asset": name,
            "Weight": f"{w:.1%}",
            "$ Amount": f"${w * investment:,.0f}",
            "Hist. Ann. Return": f"{hist_ret:.1%}",
            "Hist. Volatility": f"{hist_vol:.1%}",
            "Return Contribution": f"{w * hist_ret:.2%}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    _thesis_callout(optimal, weights, metrics, btc_idx)

# Section 11 - Mode 2: Optimize My Portfolio
def _mode2(investment, risk_profile, forecast_method, n_portfolios, btc_min, btc_max, rf, max_dd_custom):
    st.subheader("Enter Your Current Portfolio")

    ticker_input = st.text_area(
        "Your holdings (yfinance tickers, comma-separated)",
        value="SPY, GLD, QQQ",
        help="E.g.: SPY, GLD, QQQ, APPL, MSFT, BTC-USD",
        key="m2_tickers",
    )
    raw_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if not raw_tickers:
        st.info("Enter at least one ticker above.")
        return
    
    st.markdown("**Curent holding value per position ($)**")
    amount_cols = st.columns(min(len(raw_tickers), 4))
    amounts: dict = {}
    for i, t in enumerate(raw_tickers):
        with amount_cols[i % len(amount_cols)]:
            amounts[t] = float(st.number_input(
                t, min_value=0, value=int(investment // len(raw_tickers)),
                step=1000, key=f"m2_amt_{t}",
            ))

    total_val = sum(amounts.values())
    if total_val <= 0:
        st.warning("Enter holding amounts to continue.")
        return
    
    cur_names = list(amounts.keys())
    cur_weights = np.array([amounts[t] / total_val for t in cur_names])

    col_tbl, col_pie = st.columns(2)
    with col_tbl:
        st.markdown("**Current Holdings**")
        st.dataframe(pd.DataFrame([
            {"Ticker": t, "Value": f"${amounts[t]:,.0f}", "Weight": f"{amounts[t]/total_val:.1%}"}
            for t in cur_names
        ]), hide_index=True, use_container_width=True)
    with col_pie:
        st.plotly_chart(chart_donut(cur_weights, cur_names, "Current Allocation"), use_container_width=True)
    
    if not st.button("▶ Run Optimization", type="primary", key="m2_run"):
        return
    
    # Fetch live data; always add BTC to the optimization universe
    fetch_set = tuple(sorted(set(raw_tickers + ["BTC-USD"])))
    try:
        with st.spinner("Fetching live data from Yahoo Finance…"):
            prices_raw = load_yfinance(fetch_set)
        prices_raw = prices_raw.rename(columns={"BTC-USD": "BTC"})
        if prices_raw.empty or len(prices_raw) < 60:
            st.error("Not enough price history was available for these tickers. Try removing newer/thinner tickers like RGTI, IONQ, GBTS, or CCJ.")
            return
    except Exception as e:
        st.error(f"Could not fetch data: {e}")
        return
    
    asset_names = list(prices_raw.columns)
    btc_idx     = asset_names.index("BTC") if "BTC" in asset_names else 0

    # Map user holdings onto the full asset universe (BTC starts at 0 if not held)
    cur_w_full = np.zeros(len(asset_names))
    for i, name in enumerate(asset_names):
        key = "BTC-USD" if name == "BTC" else name
        if key in amounts:
            cur_w_full[i] = amounts[key] / total_val

    # Current portfolio historical metrics
    log_ret_cur, _, cov_cur = calibrate(prices_raw)
    cur_port_daily = (np.expm1(log_ret_cur.values) * cur_w_full).sum(axis=1)
    n_hist         = len(cur_port_daily)
    cur_ann_ret    = float((1 + cur_port_daily).prod() ** (TRADING_DAYS / n_hist) - 1)
    cur_ann_vol    = float(np.sqrt(cur_w_full @ cov_cur @ cur_w_full) * np.sqrt(TRADING_DAYS))
    cur_sharpe     = (cur_ann_ret - rf) / cur_ann_vol if cur_ann_vol > 0 else 0.0

    with st.spinner(f"Running {n_portfolios:,} Monte Carlo simulations…"):
        weights, metrics, optimal, fan_paths, elapsed, _, _, _ = _run_pipeline(
            prices_raw, prices_raw, n_portfolios, forecast_method,
            btc_idx, btc_min, btc_max, rf, risk_profile, max_dd_custom, total_val,
        )
    
    engine = "C++ · Pybind11" if (_CPP and forecast_method == "GBM (Geometric Brownian Motion)") else "Python · NumPy"
    st.success(f"Done — {n_portfolios:,} portfolios in **{elapsed:.2f}s** using {engine}")

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Expected Return", f"{optimal['ann_return']:.1%}", f"{optimal['ann_return'] - cur_ann_ret:+.1%} vs current")
    c2.metric("Volatility",      f"{optimal['ann_vol']:.1%}",   f"{optimal['ann_vol'] - cur_ann_vol:+.1%} vs current", delta_color="inverse")
    c3.metric("Sharpe Ratio",    f"{optimal['sharpe']:.3f}",    f"{optimal['sharpe'] - cur_sharpe:+.3f} vs current")
    c4.metric("Max Drawdown",    f"{optimal['max_drawdown']:.1%}")
    c5.metric("BTC Allocation",  f"{optimal['btc_alloc']:.1%}")
    st.divider()

    st.plotly_chart(
        chart_frontier(weights, metrics, optimal["idx"], asset_names, cur_ann_ret, cur_ann_vol),
        use_container_width=True,
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.plotly_chart(chart_donut(cur_w_full, asset_names, "Your Current Portfolio", total_val), use_container_width=True)
    with col_d2:
        st.plotly_chart(chart_donut(optimal["weights"], asset_names, "Optimized Portfolio", total_val), use_container_width=True)

    st.subheader("Recommended Portfolio Changes")
    st.caption(f"Risk profile: {risk_profile} · Sharpe improvement: {optimal['sharpe'] - cur_sharpe:+.3f}")
    rows = []
    for i, name in enumerate(asset_names):
        cur_amt = cur_w_full[i] * total_val
        opt_amt = optimal["weights"][i] * total_val
        delta   = opt_amt - cur_amt
        action  = "BUY" if delta > 50 else "SELL" if delta < -50 else "HOLD"
        rows.append({"Asset": name, "Current $": f"${cur_amt:,.0f}",
                     "Optimal $": f"${opt_amt:,.0f}", "Change $": f"${delta:+,.0f}", "Action": action})
    
    def _row_color(row):
        if row["Action"] == "BUY":  return ["background-color: rgba(34,197,94,0.18)"] * len(row)
        if row["Action"] == "SELL": return ["background-color: rgba(239,68,68,0.18)"] * len(row)
        return [""] * len(row)
    
    st.dataframe(pd.DataFrame(rows).style.apply(_row_color, axis=1), hide_index=True, use_container_width=True)

    percs = fan_percentiles(fan_paths, total_val)
    st.plotly_chart(chart_fan(percs, total_val), use_container_width=True)
    _thesis_callout(optimal, weights, metrics, btc_idx)

# Section 12 - Thesis Callout
def _thesis_callout(optimal, weights, metrics, btc_idx):
    mask = weights[:, btc_idx] < 0.01
    if mask.sum() > 10:
        baseline = float(np.nanmedian(metrics[mask, 2]))
        delta = optimal["sharpe"] - baseline
        st.info(
            f"**Project Thesis:** The optimal allocation includes **{optimal['btc_alloc']:.1%} BTC**, "
            f"improving the Sharpe ratio by **{delta:+.3f}** over a 0% BTC baseline. "
            f"This confirms our finding that Bitcoin's value comes from diversification "
            f"and asymetric return potential - not inflation protection."
        )

# Section 13 - Main Entry Point
def show_portfolio_simulator():
    # Title and Header
    st.title("Portfolio Simulator")
    st.markdown("### Optimal Bitcoin allocation via Monte Carlo simulation and historical forecasting")

    if _CPP:
        st.success("⚡ C++ Monte Carlo engine loaded (Pybind11) — fast mode active")
    else:
        st.warning(
            "C++ engine not compiled - running in Python/NumPy mode. "
            "See `app/setup.py` to compile for 20-50x speedup."
        )

    mode = st.radio("Mode", ["Auto-Allocate", "Optimize my Portfolio"], horizontal=True, key="sim_mode")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Parameters")

    investment = float(st.sidebar.number_input(
        "Investment Amount ($)", min_value=1_000, max_value=10_000_000,
        value=100_000, step=1_000, format="%d", key="sim_investment",
    ))

    risk_profile = st.sidebar.radio(
        "Risk Profile", ["Conservative", "Moderate", "Aggressive", "Custom"],
        index=1, key="sim_risk",
    )
    max_dd_custom = -0.30
    if risk_profile == "Custom":
        max_dd_custom = st.sidebar.slider(
            "Max Acceptable Drawdown (%)", -60, -5, -30, step=5,
            format="%d%%", key="sim_max_dd",
        ) / 100

    forecast_method = st.sidebar.selectbox(
        "Forecast Method",
        ["GBM (Geometric Brownian Motion)", "Historical Bootstrap", "Regime-Weighted Bootstrap"],
        key="sim_forecast",
        help="GBM uses C++ when compiled. Bootstrap and Regime-Weighted use Python.",
    )

    n_portfolios = st.sidebar.slider(
        "Monte Carlo Simulations", 500, 10_000, 3_000, step=500, key="sim_n",
        help="Each portfolio runs 2 paths × 504 days. Python ≈ 40s at 3,000. C++ ≈ 1–2s.",
    )

    rf = st.sidebar.slider(
        "Risk-Free Rate (%)", 0.0, 6.0, 4.0, step=0.25, format="%.2f%%", key="sim_rf",
    ) / 100

    st.sidebar.markdown("**BTC Allocation Bounds**")
    btc_min = st.sidebar.slider("Min BTC %", 0, 20, 0, key="sim_btc_min") / 100
    btc_max = st.sidebar.slider("Max BTC %", 5, 60, 30, key="sim_btc_max") / 100
    if btc_min >= btc_max:
        st.sidebar.error("Min BTC must be less than Max BTC.")
        btc_min, btc_max = 0.0, 0.30
    
    st.divider()

    if mode == "Auto-Allocate":
        _mode1(investment, risk_profile, forecast_method, n_portfolios, btc_min, btc_max, rf, max_dd_custom)
    else:
        _mode2(investment, risk_profile, forecast_method, n_portfolios, btc_min, btc_max, rf, max_dd_custom)
  