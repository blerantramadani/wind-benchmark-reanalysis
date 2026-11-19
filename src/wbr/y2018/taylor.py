"""Figure 1 and the Taylor comparison (reply, points 2 and 3a).

``figure1_grid``: the parabola of Fig. 1 is built as in script 13 of the
preprint, with np.gradient derivatives of the interpolated curve on a grid.
The construction is repeated on grids of 400, 800 (as published), 1,600,
2,000 and 3,200 points; slope, curvature and the distance of the expansion
point from the nearest curve knot are recorded. The knot spacing is the bin
width of the curve.

``units``: script 10 reports the relative error of the standard deviation of
the curve-variation term; the text of the preprint calls it an error of the
variance. The same error is converted to the variance scale.

``smooth_curves``: two smooth curves are fitted to the bin means of the
preprint (40 bins up to the 99.5th percentile, at least 20 hours):

  logistic  P(v) = a + (b - a) / (1 + exp(-(v - c) / s)), weighted least
            squares with the bin counts as weights;
  spline    cubic smoothing spline (scipy UnivariateSpline, k = 3), weights
            sqrt(n) / sd of the bin (sd at least 10 kW), s = number of bins.

Outside the retained bins the curve is constant (0 for wind <= 0) and its
derivatives are zero. For every pre-test AR state and horizon the variance of
the curve term by Gauss-Hermite quadrature (64 nodes; 128 and 256 as a check)
is compared with the delta approximation [m'(mu)]^2 sigma^2 and the
second-order expression + 0.5 [m''(mu)]^2 sigma^4. Two data sets: all hours
(as the preprint) and without hours that overlap a logged stop.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

from . import common as K

GRIDS = (400, 800, 1600, 2000, 3200)


def _pretest(site: str) -> pd.DataFrame:
    return K.site_raw(site).iloc[:-K.TEST]


def figure1_grid() -> pd.DataFrame:
    B07, F13 = K.b07(), K.preprint_script("13_make_final_figure1.py")
    rows = []
    for site in ("kelmarsh", "penmanshiel"):
        train = _pretest(site)
        index = pd.DatetimeIndex(train.index)
        wind, power = train["WindSpeed_m_s"].to_numpy(float), train["ActivePower_kW"].to_numpy(float)
        order, (mu, phi, sigma2, _), _, _ = B07.choose_ar_order(index, wind, 5)
        state = F13.representative_state(B07.outer_states(index, wind, order, mu), phi, mu)
        mean_power, _ = B07.empirical_curve(wind, power, B07.RATED_KW[site], 40, 20)
        upper = float(np.percentile(wind, 99.5))
        edges = np.linspace(0.0, upper, 41)
        which = np.digitize(wind, edges) - 1
        counts = np.bincount(which[(which >= 0) & (which < 40)], minlength=40)
        knots = ((edges[:-1] + edges[1:]) / 2)[counts >= 20]
        ah48, _ = B07.ar_forecast_moments(phi, sigma2, 48)
        centre = float(mu + state @ ah48)
        for n in GRIDS:
            grid = np.linspace(0.0, upper, n)
            curve = mean_power(grid)
            d1 = np.gradient(curve, grid)
            d2 = np.gradient(d1, grid)
            i = int(np.argmin(np.abs(grid - centre)))
            rows.append({"site": site, "grid_points": n, "centre_ms": centre, "grid_point_ms": float(grid[i]),
                         "slope_kw_per_ms": float(d1[i]), "curvature_kw_per_ms2": float(d2[i]),
                         "distance_to_nearest_knot_ms": float(np.min(np.abs(knots - grid[i]))),
                         "knot_spacing_ms": float(np.median(np.diff(knots)))})
    return pd.DataFrame(rows)


def units() -> pd.DataFrame:
    t10 = pd.read_csv(K.PREPRINT / "results" / "10_local_approximation_exact.csv")
    t10["sd_error_pct"] = 100 * (t10["taylor2_curve_sd_pct"] - t10["quadrature_curve_sd_pct"]) / t10["quadrature_curve_sd_pct"]
    t10["variance_error_pct"] = 100 * (t10["taylor2_curve_sd_pct"] ** 2 / t10["quadrature_curve_sd_pct"] ** 2 - 1)
    return t10


def _bins(v, p):
    upper = float(np.percentile(v, 99.5))
    edges = np.linspace(0.0, upper, 41)
    which = np.digitize(v, edges) - 1
    x, y, n, sd = [], [], [], []
    for b in range(40):
        vals = p[which == b]
        if len(vals) >= 20:
            x.append((edges[b] + edges[b + 1]) / 2); y.append(vals.mean()); n.append(len(vals)); sd.append(vals.std(ddof=1))
    return np.array(x), np.array(y), np.array(n), np.array(sd)


def _logistic(x, y, n):
    f = lambda v, a, b, c, s: a + (b - a) / (1 + np.exp(-(v - c) / s))
    p, _ = curve_fit(f, x, y, p0=[0, y.max(), 8, 1.5], sigma=1 / np.sqrt(n), maxfev=20000)
    a, b, c, s = p

    def m(v):
        return np.where(v <= 0, 0.0, f(np.clip(v, x[0], x[-1]), *p))

    def d1(v):
        e = np.exp(-(v - c) / s)
        return np.where((v <= x[0]) | (v >= x[-1]), 0.0, (b - a) * e / (s * (1 + e) ** 2))

    def d2(v):
        e = np.exp(-(v - c) / s)
        return np.where((v <= x[0]) | (v >= x[-1]), 0.0, (b - a) * e * (e - 1) / (s ** 2 * (1 + e) ** 3))

    return m, d1, d2, {"a": a, "b": b, "c": c, "s": s,
                       "weighted_rmse_kw": float(np.sqrt(np.average((f(x, *p) - y) ** 2, weights=n)))}


def _spline(x, y, n, sd):
    sp = UnivariateSpline(x, y, w=np.sqrt(n) / np.maximum(sd, 10.0), k=3, s=len(x))
    d1s, d2s = sp.derivative(1), sp.derivative(2)

    def m(v):
        return np.where(v <= 0, 0.0, sp(np.clip(v, x[0], x[-1])))

    def d1(v):
        return np.where((v <= x[0]) | (v >= x[-1]), 0.0, d1s(np.clip(v, x[0], x[-1])))

    def d2(v):
        return np.where((v <= x[0]) | (v >= x[-1]), 0.0, d2s(np.clip(v, x[0], x[-1])))

    return m, d1, d2, {"knots": len(sp.get_knots()),
                       "weighted_rmse_kw": float(np.sqrt(np.average((sp(x) - y) ** 2, weights=n)))}


def _curve_variance(m, mu_h, sd, nodes):
    gx, gw = hermgauss(nodes)
    gw = gw / math.sqrt(math.pi)
    mu_ = m(mu_h[:, None] + math.sqrt(2) * sd * gx[None, :])
    e = mu_ @ gw
    return np.maximum((mu_ * mu_) @ gw - e * e, 0.0)


def smooth_curves():
    B07 = K.b07()
    rows, fits = [], {}
    for site in ("kelmarsh", "penmanshiel"):
        pub = pd.read_csv(K.PREPRINT / K.DATASETS[site]["file"], parse_dates=["Timestamp"]).set_index("Timestamp")
        train = pub[["WindSpeed_m_s", "ActivePower_kW"]].dropna().iloc[:-K.TEST]
        stop = K.stop_flags(site, train.index)
        idx = pd.DatetimeIndex(train.index)
        v, p = train["WindSpeed_m_s"].to_numpy(float), train["ActivePower_kW"].to_numpy(float)
        order, (mu, phi, sigma2, _), _, _ = B07.choose_ar_order(idx, v, 5)
        states = B07.outer_states(idx, v, order, mu)
        for data_label, keep in (("all_hours", np.ones(len(v), bool)), ("without_stops", ~stop)):
            x, y, n, sd = _bins(v[keep], p[keep])
            for fam, (m, d1, d2, info) in (("logistic", _logistic(x, y, n)), ("spline", _spline(x, y, n, sd))):
                fits[f"{site}_{data_label}_{fam}"] = {k: float(val) for k, val in info.items()}
                for h in K.HORIZONS:
                    ah, var_u = B07.ar_forecast_moments(phi, sigma2, h)
                    mu_h = mu + states @ ah
                    sdu = math.sqrt(var_u)
                    q = {nn: _curve_variance(m, mu_h, sdu, nn).mean() for nn in (64, 128, 256)}
                    dvar = float(np.mean(d1(mu_h) ** 2)) * var_u
                    t2var = dvar + 0.5 * float(np.mean(d2(mu_h) ** 2)) * var_u ** 2
                    rows.append({"site": site, "data": data_label, "family": fam, "horizon_h": h,
                                 "cond_wind_sd_ms": sdu, "quad_var_kw2": q[64],
                                 "quad_change_128_256_pct": 100 * (q[256] - q[128]) / q[128],
                                 "delta_var_kw2": dvar, "taylor2_var_kw2": t2var,
                                 "taylor2_sd_error_pct": 100 * (math.sqrt(t2var) / math.sqrt(q[64]) - 1),
                                 "taylor2_var_error_pct": 100 * (t2var / q[64] - 1),
                                 "delta_sd_error_pct": 100 * (math.sqrt(dvar) / math.sqrt(q[64]) - 1),
                                 "delta_var_error_pct": 100 * (dvar / q[64] - 1)})
    return pd.DataFrame(rows), fits
