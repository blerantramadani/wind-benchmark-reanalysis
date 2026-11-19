"""Model-conditional benchmark and the model's own conditional-mean forecast.

The functions ``contiguous_windows``, ``fit_ar``, ``choose_ar_order``,
``companion``, ``ar_forecast_moments`` and ``empirical_curve`` are taken from
scripts/07_windonly_bound.py of the preprint repository
(github.com/blerantramadani/wind-state-predictability-benchmark, commit
13c0bba, MIT licence). Comments were shortened; the computations are
unchanged, so the numerical definitions are identical to the preprint.

For each origin i and horizon h, with the fitted Gaussian AR(p) wind model,
the empirical power curve m(v) and the bin scatter s2(v):

* conditional variance  r_i = Var[m(V)] + E[s2(V)], V ~ N(mu_h, sigma_h^2),
* model forecast        E[m(V)],
* plug-in forecast      m(mu_h),

all evaluated with 64-node Gauss-Hermite quadrature. The benchmark of a set of
origins is 100 / P_r * sqrt(mean_i r_i).

Three fits per site, test year, target and horizon (protocol, Sect. 7):

* main: wind model on all valid hours up to (first test target) - h; curve and
  scatter on the same hours, without stop hours for the normal target;
* sensitivity: the same, up to the end of the training targets;
* seasonal: as main, with one AR model per meteorological season.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

from . import config as C
from .split import ar_rows, curve_rows, history, load_grid, season, split

SEASONS = ("DJF", "MAM", "JJA", "SON")
GH_X, GH_W = hermgauss(C.P["quadrature_nodes"])
GH_W = GH_W / math.sqrt(math.pi)


# --- copied from 07_windonly_bound.py (commit 13c0bba) -------------------------------------------

def contiguous_windows(index: pd.DatetimeIndex, width: int) -> np.ndarray:
    """Indices i for which rows [i-width+1, ..., i] are hourly contiguous."""
    if len(index) < width:
        return np.empty(0, dtype=int)
    gaps = np.diff(index.to_numpy()).astype("timedelta64[m]").astype(float) != 60.0
    csum = np.concatenate(([0], np.cumsum(gaps.astype(int))))
    return np.array(
        [i for i in range(width - 1, len(index)) if csum[i] - csum[i - width + 1] == 0],
        dtype=int,
    )


def fit_ar(index: pd.DatetimeIndex, v: np.ndarray, order: int, target_ids: np.ndarray | None = None):
    """OLS Gaussian AR(p) on contiguous lag/response rows."""
    mu = float(np.mean(v))
    z = v - mu
    if target_ids is None:
        target_ids = contiguous_windows(index, order + 1)
    else:
        target_ids = np.asarray(target_ids, dtype=int)
    if len(target_ids) <= order + 2:
        raise ValueError(f"too few contiguous rows for AR({order})")
    y = z[target_ids]
    x = np.column_stack([z[target_ids - lag] for lag in range(1, order + 1)])
    phi, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - np.einsum("ij,j->i", x, phi)
    sigma2 = float(np.mean(resid**2))
    n = len(resid)
    ll = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)
    bic = (order + 1) * math.log(n) - 2.0 * ll
    return mu, np.asarray(phi, dtype=float), sigma2, bic


def choose_ar_order(index: pd.DatetimeIndex, v: np.ndarray, pmax: int):
    """Select AR order by BIC on a common, gap-aware response set."""
    comparison_ids = contiguous_windows(index, pmax + 1)
    if len(comparison_ids) <= pmax + 2:
        raise ValueError("too few common contiguous rows for AR-order comparison")
    comparison_fits = {p: fit_ar(index, v, p, target_ids=comparison_ids) for p in range(1, pmax + 1)}
    bics = {p: comparison_fits[p][3] for p in comparison_fits}
    order = min(bics, key=bics.get)
    mu, phi, sigma2, _ = fit_ar(index, v, order)
    return order, (mu, phi, sigma2, bics[order]), bics, len(comparison_ids)


def companion(phi: np.ndarray) -> np.ndarray:
    p = len(phi)
    a = np.zeros((p, p), dtype=float)
    a[0, :] = phi
    if p > 1:
        a[1:, :-1] = np.eye(p - 1)
    return a


def ar_forecast_moments(phi: np.ndarray, sigma2: float, horizon: int):
    """Return (first-row A^h, innovation variance) exactly for Gaussian AR(p)."""
    a = companion(phi)
    ah = np.linalg.matrix_power(a, horizon)[0, :]
    ak = np.eye(len(phi))
    variance = 0.0
    for _ in range(horizon):
        variance += sigma2 * float((ak[0, 0]) ** 2)
        ak = ak @ a
    return ah, variance


def empirical_curve(v: np.ndarray, p: np.ndarray, rated: float, nbins: int, min_count: int):
    """Return m(v) and s2(v), both fitted only to the supplied training vectors."""
    upper = float(np.percentile(v, 99.5))
    edges = np.linspace(0.0, upper, nbins + 1)
    which = np.digitize(v, edges, right=False) - 1
    means, variances, centers = [], [], []
    for b in range(nbins):
        values = p[which == b]
        if len(values) >= min_count:
            centers.append((edges[b] + edges[b + 1]) / 2.0)
            means.append(float(np.mean(values)))
            variances.append(float(np.var(values, ddof=1)))
    if len(centers) < 3:
        raise ValueError("insufficient populated power-curve bins")
    x = np.asarray(centers)
    y = np.clip(np.asarray(means), 0.0, rated)
    s2 = np.maximum(np.asarray(variances), 0.0)

    def mean_power(q):
        q = np.asarray(q, dtype=float)
        out = np.interp(np.maximum(q, 0.0), x, y, left=0.0, right=y[-1])
        return np.where(q <= 0.0, 0.0, out)

    def residual_variance(q):
        q = np.asarray(q, dtype=float)
        out = np.interp(np.maximum(q, 0.0), x, s2, left=0.0, right=s2[-1])
        return np.where(q <= 0.0, 0.0, out)

    return mean_power, residual_variance


# --- reanalysis ---------------------------------------------------------------------------------

def wind_model(times, wind) -> dict:
    order, (mu, phi, sigma2, _), _, _ = choose_ar_order(pd.DatetimeIndex(times), wind, C.P["ar_max_order"])
    return {"order": order, "mu": mu, "phi": phi, "sigma2": sigma2}


def fit_reference(g, last_pos: int, target: str, seasonal: bool = False) -> dict:
    ar_idx = ar_rows(g, last_pos)
    cv_idx = curve_rows(g, last_pos, target)
    m, s2 = empirical_curve(g.wind[cv_idx], g.power[cv_idx], C.RATED, C.P["curve_bins"], C.P["curve_min_count"])
    fit = {"m": m, "s2": s2, "ar": wind_model(g.time[ar_idx], g.wind[ar_idx]), "ar_idx": ar_idx}
    if seasonal:
        seas = season(g.time[ar_idx])
        fit["ar_season"] = {s: wind_model(g.time[ar_idx[seas == s]], g.wind[ar_idx[seas == s]]) for s in SEASONS}
    return fit


def _nodes(ar: dict, hist: np.ndarray, h: int):
    states = hist[:, :ar["order"]] - ar["mu"]
    ah, var_u = ar_forecast_moments(ar["phi"], ar["sigma2"], h)
    mu_h = ar["mu"] + states @ ah
    return mu_h, mu_h[:, None] + math.sqrt(2.0) * math.sqrt(max(var_u, 0.0)) * GH_X[None, :]


def conditional_variance(ar: dict, m, s2, hist: np.ndarray, h: int) -> np.ndarray:
    _, u = _nodes(ar, hist, h)
    m_u = m(u)
    e_m = m_u @ GH_W
    return np.maximum((m_u * m_u) @ GH_W - e_m * e_m, 0.0) + s2(u) @ GH_W


def conditional_mean(ar: dict, m, hist: np.ndarray, h: int):
    """Return the model forecast E[m(V)] and the plug-in forecast m(E[V])."""
    mu_h, u = _nodes(ar, hist, h)
    return m(u) @ GH_W, m(mu_h)


def main():
    C.ensure_dirs()
    summary = []
    for site in C.SITES:
        g = load_grid(site)
        for year in C.TEST_YEARS:
            for target in C.TARGETS:
                for h in C.HORIZONS:
                    s = split(g, year, h, target)
                    main_fit = fit_reference(g, s.bench_cut_main, target, seasonal=True)
                    sens_fit = fit_reference(g, s.bench_cut_sens, target)
                    hist = history(g, s.test)
                    v_main = conditional_variance(main_fit["ar"], main_fit["m"], main_fit["s2"], hist, h)
                    v_sens = conditional_variance(sens_fit["ar"], sens_fit["m"], sens_fit["s2"], hist, h)
                    seas_origin = season(g.time[s.test - 1])
                    v_seas = np.empty(len(s.test))
                    for sn in SEASONS:
                        k = seas_origin == sn
                        if k.any():
                            v_seas[k] = conditional_variance(main_fit["ar_season"][sn], main_fit["m"], main_fit["s2"], hist[k], h)
                    pd.DataFrame({"origin": s.test, "target_time": g.time[s.test + h - 1],
                                  "season_target": season(g.time[s.test + h - 1]),
                                  "var_main": v_main, "var_sens": v_sens, "var_seasonal_ar": v_seas}).to_csv(
                        C.BENCHMARK / f"{site}_{year}_{target}_h{h}.csv", index=False)
                    for part, o in (("val", s.val), ("test", s.test)):
                        hh = history(g, o)
                        b_main, plug = conditional_mean(main_fit["ar"], main_fit["m"], hh, h)
                        b_sens, _ = conditional_mean(sens_fit["ar"], sens_fit["m"], hh, h)
                        pd.DataFrame({"origin": o, "Bayes_main": b_main, "Bayes_sens": b_sens, "PlugIn_main": plug}).to_csv(
                            C.FORECASTS / f"{site}_{year}_{target}_h{h}_bayes_{part}.csv", index=False)
                    summary.append({"site": site, "test_year": year, "target": target, "horizon_h": h,
                                    "n_test": len(s.test), "ar_order_main": main_fit["ar"]["order"],
                                    "ar_order_sens": sens_fit["ar"]["order"],
                                    "ar_order_seasons": ",".join(str(main_fit["ar_season"][sn]["order"]) for sn in SEASONS),
                                    "fit_last_time": str(g.time[s.bench_cut_main]),
                                    "bench_test_main_pct": math.sqrt(v_main.mean()) * C.SCALE,
                                    "bench_test_sens_pct": math.sqrt(v_sens.mean()) * C.SCALE,
                                    "bench_test_seasonal_ar_pct": math.sqrt(v_seas.mean()) * C.SCALE})
            print(f"{site}: benchmark done", flush=True)
    pd.DataFrame(summary).to_csv(C.BENCHMARK / "benchmark_summary.csv", index=False)


if __name__ == "__main__":
    main()
