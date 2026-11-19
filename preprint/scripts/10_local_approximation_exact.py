#!/usr/bin/env python3
"""Audit local Taylor approximations against the exact wind-only benchmark.

This is a diagnostic/sensitivity analysis, not a replacement for the reported
benchmark.  It uses the same pre-test window, BIC-selected gap-aware AR(p)
model, empirical power curve, and exact companion-matrix forecast moments as
07_windonly_bound.py.  Only the approximation of the curve-variation term is
changed.

The first and second derivatives are estimated from the empirical binned power
curve.  Because curvature is intrinsically smoothing-sensitive, results are
reported for every requested odd smoothing window; a claim should be made only
when its qualitative conclusion is stable across those windows.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
_spec = importlib.util.spec_from_file_location("windonly_benchmark", SCRIPT_DIR / "07_windonly_bound.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load 07_windonly_bound.py")
BENCH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BENCH)


def derivative_functions(v: np.ndarray, power: np.ndarray, rated: float, nbins: int,
                         min_count: int, smoothing: int):
    """Derivative estimates on the same binned curve used by the benchmark."""
    if smoothing < 1 or smoothing % 2 == 0:
        raise ValueError("smoothing window must be a positive odd integer")
    upper = float(np.percentile(v, 99.5))
    edges = np.linspace(0.0, upper, nbins + 1)
    which = np.digitize(v, edges, right=False) - 1
    centers, means = [], []
    for b in range(nbins):
        values = power[which == b]
        if len(values) >= min_count:
            centers.append((edges[b] + edges[b + 1]) / 2.0)
            means.append(float(np.mean(values)))
    x = np.asarray(centers)
    y = np.clip(np.asarray(means), 0.0, rated)
    if len(x) < 5:
        raise ValueError("insufficient populated bins to estimate derivatives")

    def smooth(values: np.ndarray) -> np.ndarray:
        if smoothing == 1:
            return values
        kernel = np.ones(smoothing) / smoothing
        pad = smoothing // 2
        return np.convolve(np.pad(values, (pad, pad), mode="edge"), kernel, mode="valid")

    y_s = smooth(y)
    first = smooth(np.gradient(y_s, x))
    second = smooth(np.gradient(first, x))

    def interpolate(values: np.ndarray, q):
        q = np.asarray(q, dtype=float)
        # The benchmark defines output as zero at non-positive wind speed and
        # constant above its final populated bin.  A flat extension has zero
        # derivative, hence the explicit boundary treatment here.
        out = np.interp(np.clip(q, x[0], x[-1]), x, values)
        return np.where((q <= 0.0) | (q >= x[-1]), 0.0, out)

    return lambda q: interpolate(first, q), lambda q: interpolate(second, q)


def evaluate_site(site: str, horizons, args):
    path = ROOT / BENCH.DATASETS[site]
    raw = pd.read_csv(path)
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    raw = raw.sort_values("Timestamp").set_index("Timestamp")
    train = raw[["WindSpeed_m_s", "ActivePower_kW"]].dropna().iloc[:-args.test_hours].copy()
    rated = BENCH.RATED_KW[site]
    index = pd.DatetimeIndex(train.index)
    v = train["WindSpeed_m_s"].to_numpy(float)
    power = train["ActivePower_kW"].to_numpy(float)
    order, (mu, phi, sigma2, bic), _, n_bic_comparison_rows = BENCH.choose_ar_order(index, v, args.pmax)
    states = BENCH.outer_states(index, v, order, mu)
    mean_power, _ = BENCH.empirical_curve(v, power, rated, args.nbins, args.min_bin_count)
    nodes, weights = hermgauss(args.nodes)
    weights = weights / math.sqrt(math.pi)
    rows = []

    print(f"[{site}] BIC selected AR({order}); {len(train)} pre-test rows; {len(states)} contiguous states")
    for smoothing in args.smoothing_windows:
        first, second = derivative_functions(v, power, rated, args.nbins, args.min_bin_count, smoothing)
        for horizon in horizons:
            ah, variance = BENCH.ar_forecast_moments(phi, sigma2, horizon)
            mu_h = mu + np.einsum("ij,j->i", states, ah)
            sd = math.sqrt(max(variance, 0.0))
            wind_nodes = mu_h[:, None] + math.sqrt(2.0) * sd * nodes[None, :]
            p_nodes = mean_power(wind_nodes)
            e_p = np.einsum("ij,j->i", p_nodes, weights)
            quadrature_var = float(np.mean(np.einsum("ij,j->i", (p_nodes - e_p[:, None]) ** 2, weights)))

            d1 = first(mu_h)
            d2 = second(mu_h)
            delta_var = float(np.mean(d1 ** 2)) * variance
            second_var = 0.5 * float(np.mean(d2 ** 2)) * variance ** 2
            taylor2_var = delta_var + second_var
            scale = 100.0 / rated
            quad = math.sqrt(max(quadrature_var, 0.0)) * scale
            delta = math.sqrt(max(delta_var, 0.0)) * scale
            taylor2 = math.sqrt(max(taylor2_var, 0.0)) * scale
            rows.append({
                "dataset": site, "horizon_h": horizon, "ar_order": order,
                "ar_bic": bic, "n_pretest_hours": len(train),
                "n_bic_comparison_rows": n_bic_comparison_rows,
                "n_outer_states": len(states),
                "smoothing_window_bins": smoothing, "cond_wind_sd_ms": sd,
                "quadrature_curve_sd_pct": quad, "delta_curve_sd_pct": delta,
                "taylor2_curve_sd_pct": taylor2,
                "delta_relative_error_pct": 100.0 * (delta - quad) / quad,
                "taylor2_relative_error_pct": 100.0 * (taylor2 - quad) / quad,
                "curvature_variance_share_pct": 100.0 * second_var / taylor2_var if taylor2_var else np.nan,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="kelmarsh,penmanshiel")
    parser.add_argument("--horizons", default=",".join(map(str, BENCH.DEFAULT_HORIZONS)))
    parser.add_argument("--test-hours", type=int, default=1000)
    parser.add_argument("--pmax", type=int, default=5,
                        help="maximum candidate AR order; BIC uses one common response set")
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--nbins", type=int, default=40)
    parser.add_argument("--min-bin-count", type=int, default=20)
    parser.add_argument("--smoothing-windows", default="3,5,7")
    parser.add_argument("--out", default="results/10_local_approximation_exact.csv")
    args = parser.parse_args()
    args.smoothing_windows = [int(x) for x in args.smoothing_windows.split(",")]
    horizons = [int(x) for x in args.horizons.split(",")]
    rows = []
    for site in [x.strip() for x in args.datasets.split(",")]:
        rows.extend(evaluate_site(site, horizons, args))
    out = pd.DataFrame(rows)
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination, index=False)
    print(f"saved: {destination}")


if __name__ == "__main__":
    main()
