"""Benchmark and forecast errors on the same test origins (reply, point 3b(ii)).

Table 2 of the preprint averages the conditional variance over the pre-test
wind states, Table 3 scores the forecasts on the 1,000 test targets. Here the
benchmark is evaluated at the wind states of the test origins:

    B_test(h)^2 = (1/n) sum_i Var(P_{i+h-1} | wind up to hour i-1),

with the fit of script 07 (AR order by BIC, 40-bin curve, bin variance as
scatter, 64-node Gauss-Hermite quadrature). Two fits: the published one (all
rows before the last 1,000) and a purged one (only hours known when the first
test forecast is issued).

Uncertainty of RMSE(method) - B_test, 48-h moving blocks, 1,000 resamples:
test only (fit fixed) and with refit (the benchmark is also refitted on a
block resample of the pre-test period, as in Sect. 3.6 of the preprint). The
forecasts are the archived ones; training uncertainty is not included.

``calibration`` evaluates the model's own forecasts at the same origins: the
conditional mean E[m(V) | wind history] and the plug-in m(E[V | wind
history]), for all operating states and for target hours without a logged
stop (curve and scatter fitted without stop hours, AR on all hours).

``block_check`` repeats the test-only interval of the best method with blocks
of 24, 48, 96 and 168 hours and gives the autocorrelation of the hourly
difference between squared error and conditional variance.

The random streams (seeds 20261102, 31 and 7) and the order of the draws are
those of the analysis behind the reply, so the intervals are reproduced.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

from . import common as K

NODES, NBINS, MINCOUNT, PMAX, BLOCK, N_BOOT, SEED = 64, 40, 20, 5, 48, 1000, 20261102
METHODS = ("WindPersistenceCurve", "LinearRegression", *K.NETWORKS)
GH_X, GH_W = hermgauss(NODES)
GH_W = GH_W / math.sqrt(math.pi)


def fit(train: pd.DataFrame, rated: float) -> dict:
    B07 = K.b07()
    v = train["WindSpeed_m_s"].to_numpy(float)
    order, (mu, phi, sigma2, _), _, _ = B07.choose_ar_order(pd.DatetimeIndex(train.index), v, PMAX)
    m, s2 = B07.empirical_curve(v, train["ActivePower_kW"].to_numpy(float), rated, NBINS, MINCOUNT)
    return {"order": order, "mu": mu, "phi": phi, "sigma2": sigma2, "m": m, "s2": s2}


def _nodes(f: dict, hist: np.ndarray, h: int):
    """hist[:, 0] is the wind of the last known hour, hist[:, 1] one hour earlier, ..."""
    ah, var_u = K.b07().ar_forecast_moments(f["phi"], f["sigma2"], h)
    mu_h = f["mu"] + (hist[:, :f["order"]] - f["mu"]) @ ah
    return mu_h, mu_h[:, None] + math.sqrt(2.0) * math.sqrt(max(var_u, 0.0)) * GH_X[None, :]


def conditional_variance(f: dict, hist: np.ndarray, h: int) -> np.ndarray:
    _, u = _nodes(f, hist, h)
    m_u = f["m"](u)
    e_m = m_u @ GH_W
    return np.maximum((m_u * m_u) @ GH_W - e_m * e_m, 0.0) + f["s2"](u) @ GH_W


def model_forecasts(f: dict, hist: np.ndarray, h: int):
    """Conditional-mean forecast E[m(V)] and plug-in forecast m(E[V])."""
    mu_h, u = _nodes(f, hist, h)
    return f["m"](u) @ GH_W, f["m"](mu_h)


def _history(df: pd.DataFrame, te: np.ndarray) -> np.ndarray:
    wind = df["WindSpeed_m_s"].to_numpy(float)
    return np.column_stack([wind[te - 1 - k] for k in range(PMAX)])


def _block_indices(n: int, rng) -> np.ndarray:
    starts = rng.integers(0, n - BLOCK + 1, size=math.ceil(n / BLOCK))
    return np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]


def matched_origins():
    """Returns (all methods, best method per case, conditional variance per origin, summary)."""
    B07 = K.b07()
    rng = np.random.default_rng(SEED)
    bound = K.published_bound()
    rows, per_origin = [], []
    for site, meta in K.DATASETS.items():
        rated, scale = meta["rated"], 100.0 / meta["rated"]
        df, raw = K.site_data(site), K.site_raw(site)
        train = raw.iloc[:-K.TEST]
        f_pub = fit(train, rated)
        cases = {}
        for h in K.HORIZONS:
            te = K.valid_origins(df.index, h)[-K.TEST:]
            pred = K.archived_forecasts(site, h)
            assert np.array_equal(pred["target_idx"].to_numpy(int), te + h - 1)
            hist = _history(df, te)
            f_pur = fit(raw[raw.index <= df.index[te[0] - 1]], rated)
            var_pub, var_pur = conditional_variance(f_pub, hist, h), conditional_variance(f_pur, hist, h)
            actual = pred["actual"].to_numpy(float)
            fc = {m: pred[K.forecast_column(m)].to_numpy(float) for m in METHODS}
            cases[h] = (hist, actual, fc, var_pub)
            for m, yhat in fc.items():
                rows.append({"site": site, "horizon_h": h, "method": m, "rmse_pct": K.nrmse(actual, yhat, rated),
                             "bench_pretest_published_pct": bound.loc[(site, h), "bound_pct"],
                             "bench_test_public_fit_pct": math.sqrt(var_pub.mean()) * scale,
                             "bench_test_purged_fit_pct": math.sqrt(var_pur.mean()) * scale,
                             "ar_order_purged_fit": f_pur["order"]})
            per_origin.append(pd.DataFrame({"site": site, "horizon_h": h, "target_time": df.index[te + h - 1],
                                            "cond_var_kw2": var_pub, "actual_kw": actual}))
        draws = {(h, m): {"test": [], "fit": []} for h in K.HORIZONS for m in METHODS}
        for b in range(N_BOOT):
            idx = _block_indices(K.TEST, rng)
            f_b = fit(B07.moving_block_sample(train, BLOCK, rng), rated)
            for h, (hist, actual, fc, var_pub) in cases.items():
                bt = math.sqrt(var_pub[idx].mean()) * scale
                bf = math.sqrt(conditional_variance(f_b, hist[idx], h).mean()) * scale
                for m, yhat in fc.items():
                    r = K.nrmse(actual[idx], yhat[idx], rated)
                    draws[(h, m)]["test"].append(r - bt)
                    draws[(h, m)]["fit"].append(r - bf)
            if (b + 1) % 100 == 0:
                print(f"[{site}] matched origins: resample {b + 1}/{N_BOOT}", flush=True)
        for row in rows:
            if row["site"] == site:
                for kind in ("test", "fit"):
                    lo, hi = np.percentile(draws[(row["horizon_h"], row["method"])][kind], [2.5, 97.5])
                    row[f"gap_ci95_{kind}_low_pp"], row[f"gap_ci95_{kind}_high_pp"] = float(lo), float(hi)
    res = pd.DataFrame(rows)
    res["gap_pretest_published_pp"] = res["rmse_pct"] - res["bench_pretest_published_pct"]
    res["gap_test_public_fit_pp"] = res["rmse_pct"] - res["bench_test_public_fit_pct"]
    res["gap_test_purged_fit_pp"] = res["rmse_pct"] - res["bench_test_purged_fit_pct"]
    best = res.loc[res.groupby(["site", "horizon_h"])["rmse_pct"].idxmin()].reset_index(drop=True)
    bench = res.drop_duplicates(["site", "horizon_h"])
    diff = bench["bench_test_public_fit_pct"] - bench["bench_pretest_published_pct"]
    summary = {
        "test_origins_per_case": K.TEST,
        "best_gap_test_range_pp": [float(best["gap_test_public_fit_pp"].min()), float(best["gap_test_public_fit_pp"].max())],
        "best_gap_published_range_pp": [float(best["gap_pretest_published_pp"].min()), float(best["gap_pretest_published_pp"].max())],
        "best_ci_test_excludes_zero": int(((best["gap_ci95_test_low_pp"] > 0) | (best["gap_ci95_test_high_pp"] < 0)).sum()),
        "best_ci_fit_excludes_zero": int(((best["gap_ci95_fit_low_pp"] > 0) | (best["gap_ci95_fit_high_pp"] < 0)).sum()),
        "bench_test_minus_pretest_pp": {s: [float(d.min()), float(d.max())] for s, d in diff.groupby(bench["site"])},
        "n_boot": N_BOOT, "block_hours": BLOCK, "seed": SEED,
    }
    return res, best, pd.concat(per_origin, ignore_index=True), summary


def calibration() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    rows = []
    for site, meta in K.DATASETS.items():
        rated = meta["rated"]
        df, raw = K.site_data(site), K.site_raw(site)
        train = raw.iloc[:-K.TEST]
        stop_train = K.stop_flags(site, train.index)
        fits = {"all_states": fit(train, rated), "normal": fit(train[~stop_train], rated)}
        # the AR model is fitted on all hours; only curve and scatter change for the normal target
        fits["normal"].update({k: fits["all_states"][k] for k in ("order", "mu", "phi", "sigma2")})
        for h in K.HORIZONS:
            te = K.valid_origins(df.index, h)[-K.TEST:]
            hist = _history(df, te)
            actual = df["ActivePower_kW"].to_numpy(float)[te + h - 1]
            stop_target = K.stop_flags(site, df.index[te + h - 1])
            for target, f in fits.items():
                keep = ~stop_target if target == "normal" else np.ones(len(te), bool)
                var = conditional_variance(f, hist[keep], h)
                cond_mean, plugin = model_forecasts(f, hist[keep], h)
                a = actual[keep]
                e2 = (a - cond_mean) ** 2
                n, g = len(a), []
                for _ in range(N_BOOT):
                    starts = rng.integers(0, n - BLOCK + 1, size=math.ceil(n / BLOCK))
                    idx = (starts[:, None] + np.arange(BLOCK)).ravel()[:n]
                    g.append((math.sqrt(e2[idx].mean()) - math.sqrt(var[idx].mean())) * 100 / rated)
                lo, hi = np.percentile(g, [2.5, 97.5])
                bench, rmse = math.sqrt(var.mean()) * 100 / rated, math.sqrt(e2.mean()) * 100 / rated
                rows.append({"site": site, "target": target, "horizon_h": h, "n_test": int(n),
                             "bench_test_pct": bench, "rmse_model_forecast_pct": rmse,
                             "rmse_plugin_pct": math.sqrt(((a - plugin) ** 2).mean()) * 100 / rated,
                             "gap_model_forecast_pp": rmse - bench,
                             "gap_ci95_low_pp": lo, "gap_ci95_high_pp": hi,
                             "mean_error_model_forecast_kw": float((a - cond_mean).mean())})
    return pd.DataFrame(rows)


def block_check(best: pd.DataFrame, per_origin: pd.DataFrame):
    rng = np.random.default_rng(7)
    rows, acf = [], {}
    for r in best.itertuples():
        pred = K.archived_forecasts(r.site, r.horizon_h)
        err = pred["actual"] - pred[K.forecast_column(r.method)]
        v = per_origin[(per_origin.site == r.site) & (per_origin.horizon_h == r.horizon_h)]["cond_var_kw2"].to_numpy()
        e2, n = (err ** 2).to_numpy(), len(err)
        for L in (24, 48, 96, 168):
            g = []
            for _ in range(2000):
                starts = rng.integers(0, n - L + 1, size=math.ceil(n / L))
                idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
                g.append((math.sqrt(e2[idx].mean()) - math.sqrt(v[idx].mean())) * 100 / 2050)
            g = np.array(g)
            lo, hi = np.percentile(g, [2.5, 97.5])
            rows.append({"site": r.site, "horizon_h": r.horizon_h, "method": r.method, "block_h": L,
                         "gap_point_pp": r.gap_test_public_fit_pp, "boot_mean_pp": g.mean(), "boot_sd_pp": g.std(),
                         "ci_low_pp": lo, "ci_high_pp": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
        if r.horizon_h in (1, 48):
            d = err ** 2 - v
            acf[f"{r.site}_h{r.horizon_h}"] = {str(k): float(d.autocorr(k)) for k in (1, 6, 24, 48, 96)}
    return pd.DataFrame(rows), acf
