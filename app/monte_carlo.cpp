#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <algorithm>

namespace py = pybind11;
using namespace std;

// -- Per-Portfolio Metrics
struct Metrics {
    double ann_return;
    double ann_vol;
    double sharpe;
    double max_drawdown;
    double var95;
};

Metrics compute_portfolio_metrics(const vector<double>& values, double rf_annual) {
    int n = static_cast<int>(values.size());
    int n_days = n - 1;

    vector<double> daily(n_days);
    for (int i = 0; i < n_days; ++i) {
        daily[i] = values[i + 1] / values[i] - 1.0;
    }

    double total_ret = values[n_days] / values[0] - 1.0;
    double ann_ret = pow(1.0 + total_ret, 252.0 / n_days) - 1.0;

    // Welford's online algorithm - numerically stable incremental variance
    double mean = 0.0, M2 = 0.0;
    for (int i = 0; i < n_days; ++i) {
        double delta = daily[i] - mean;
        mean += delta / (i + 1);
        M2 += delta * (daily[i] - mean);
    }
    double ann_vol = (n_days > 1) ? sqrt(M2 / (n_days - 1)) * sqrt(252.0) : 0.0;
    double sharpe = (ann_vol > 1e-10) ? (ann_ret - rf_annual) / ann_vol : 0.0;

    double peak = values[0];
    double max_dd = 0.0;
    for (double v : values) {
        if (v > peak) {
            peak = v;
        }
        double dd = v / peak - 1.0;
        if (dd < max_dd) {
            max_dd = dd;
        }
    }

    // 5th-percentile VaR via partial sort - O(n) average
    vector<double> daily_copy = daily;
    int p5_idx = max(0, static_cast<int>(0.05 * n_days) - 1);
    nth_element(daily_copy.begin(), daily_copy.begin() + p5_idx, daily_copy.end());
    double var95 = daily_copy[p5_idx] * sqrt(252.0);

    return {ann_ret, ann_vol, sharpe, max_dd, var95};
}

// -- Main Simulation
//
// normals: pre-generated standard normal array from Python, shape
//          (n_portfolios, n_mc_paths, n_days, n_assets).
//
// Using the same normals array in both Python and C++ guarantees
// bit-for-bit identical simulated paths and therefore identical output.
// C++ no longer needs its own RNG — it just reads from the provided array.
py::dict run_monte_carlo(
    py::array_t<double> mu_arr,
    py::array_t<double> chol_arr,
    py::array_t<double> weights_arr,
    py::array_t<double> normals_arr,
    int n_days,
    double rf_daily,
    double rebal_freq,
    double drift_limit,
    double txcost,
    double dd_trigger,
    int dd_halflife
) {
    auto mu      = mu_arr.unchecked<1>();
    auto chol    = chol_arr.unchecked<2>();
    auto weights = weights_arr.unchecked<2>();
    auto normals = normals_arr.unchecked<4>();   // (n_portfolios, n_mc_paths, n_days, n_assets)

    int n_assets     = static_cast<int>(mu.shape(0));
    int n_portfolios = static_cast<int>(weights.shape(0));
    int n_mc_paths   = static_cast<int>(normals.shape(1));  // derived from the array
    int BTC_COL      = 0;
    int rebal_freq_i = static_cast<int>(rebal_freq);
    double rf_annual = rf_daily * 252.0;

    py::array_t<double> paths_out({n_portfolios, n_days + 1});
    py::array_t<double> metrics_out({n_portfolios, 5});
    auto paths_buf   = paths_out.mutable_unchecked<2>();
    auto metrics_buf = metrics_out.mutable_unchecked<2>();

    // Portfolios are embarrassingly parallel - each reads its own slice of normals
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 8)
    #endif
    for (int p = 0; p < n_portfolios; ++p) {

        vector<double> target_w(n_assets);
        for (int i = 0; i < n_assets; ++i) {
            target_w[i] = weights(p, i);
        }

        vector<double> path_avg(n_days + 1, 0.0);

        // Reusable workspace - allocated once per portfolio
        vector<double> z(n_assets), eps(n_assets), asset_ret(n_assets), curr_w(n_assets);

        for (int mc = 0; mc < n_mc_paths; ++mc) {
            curr_w = target_w;
            vector<double> path(n_days + 1);
            path[0] = 1.0;
            double peak = 1.0;
            int days_rebal = 0;
            int dd_days = 0;

            for (int t = 0; t < n_days; ++t) {

                // Step 1: read pre-generated normals (same values as Python)
                for (int i = 0; i < n_assets; ++i) {
                    z[i] = normals(p, mc, t, i);
                }

                // Step 2: Cholesky correlation - eps = L @ z (lower-triangular multiply)
                for (int i = 0; i < n_assets; ++i) {
                    eps[i] = 0.0;
                    for (int j = 0; j <= i; ++j) {
                        eps[i] += chol(i, j) * z[j];
                    }
                }

                // Step 3: compute arithmetic returns and portfolio return
                double port_ret = 0.0;
                for (int i = 0; i < n_assets; ++i) {
                    asset_ret[i] = expm1(mu(i) + eps[i]);
                    port_ret += curr_w[i] * asset_ret[i];
                }
                path[t + 1] = path[t] * (1.0 + port_ret);

                // Step 4: drift weights after price changes
                double total_val = 0.0;
                for (int i = 0; i < n_assets; ++i) {
                    total_val += curr_w[i] * (1.0 + asset_ret[i]);
                }
                if (total_val > 0.0) {
                    for (int i = 0; i < n_assets; ++i) {
                        curr_w[i] = curr_w[i] * (1.0 + asset_ret[i]) / total_val;
                    }
                }

                // Step 5: drawdown circuit breaker
                if (path[t + 1] > peak) {
                    peak = path[t + 1];
                }
                double dd = path[t + 1] / peak - 1.0;
                if (dd < dd_trigger && dd_days == 0) {
                    dd_days = dd_halflife;
                }
                if (dd_days > 0) {
                    double btc_excess = curr_w[BTC_COL] * 0.5;
                    curr_w[BTC_COL] -= btc_excess;
                    double other_sum = 0.0;
                    for (int i = 0; i < n_assets; ++i) {
                        if (i != BTC_COL) {
                            other_sum += curr_w[i];
                        }
                    }
                    if (other_sum > 0.0) {
                        for (int i = 0; i < n_assets; ++i) {
                            if (i != BTC_COL) {
                                curr_w[i] += btc_excess * curr_w[i] / other_sum;
                            }
                        }
                    }
                    dd_days--;
                }

                // Step 6: quarterly drift rebalancing
                days_rebal++;
                if (days_rebal >= rebal_freq_i) {
                    double max_drift = 0.0;
                    for (int i = 0; i < n_assets; ++i) {
                        max_drift = max(max_drift, abs(curr_w[i] - target_w[i]));
                    }
                    if (max_drift > drift_limit) {
                        double turnover = 0.0;
                        for (int i = 0; i < n_assets; ++i) {
                            turnover += abs(curr_w[i] - target_w[i]);
                        }
                        path[t + 1] *= (1.0 - turnover * txcost);
                        curr_w = target_w;
                    }
                    days_rebal = 0;
                }

            } // end day loop

            // Accumulate this path into the running sum
            for (int t = 0; t <= n_days; ++t) {
                path_avg[t] += path[t];
            }

        } // end mc loop

        // Average across all mc paths and write output for this portfolio
        vector<double> final_path(n_days + 1);
        for (int t = 0; t <= n_days; ++t) {
            final_path[t] = path_avg[t] / n_mc_paths;
            paths_buf(p, t) = final_path[t];
        }

        Metrics m = compute_portfolio_metrics(final_path, rf_annual);
        metrics_buf(p, 0) = m.ann_return;
        metrics_buf(p, 1) = m.ann_vol;
        metrics_buf(p, 2) = m.sharpe;
        metrics_buf(p, 3) = m.max_drawdown;
        metrics_buf(p, 4) = m.var95;

    } // end portfolio loop

    py::dict result;
    result["paths"]   = paths_out;
    result["metrics"] = metrics_out;
    return result;
}

// -- Module
PYBIND11_MODULE(monte_carlo, m) {
    m.doc() = "Path-dependent GBM simulation: rebalancing, circuit breaker, multi-path averaging";
    m.def("run_monte_carlo", &run_monte_carlo,
        py::arg("mu"),
        py::arg("chol"),
        py::arg("weights"),
        py::arg("normals"),
        py::arg("n_days")      = 504,
        py::arg("rf_daily")    = 0.0,
        py::arg("rebal_freq")  = 63.0,
        py::arg("drift_limit") = 0.05,
        py::arg("txcost")      = 0.001,
        py::arg("dd_trigger")  = -0.30,
        py::arg("dd_halflife") = 21
    );
}
