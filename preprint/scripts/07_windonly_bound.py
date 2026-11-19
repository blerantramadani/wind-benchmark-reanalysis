#!/usr/bin/env python3
"""Wind-only predictability bound for Paper 1.

This script is intentionally separate from the legacy floor scripts.  It
estimates, using PRE-TEST data only,

    B_h^2 = E[ Var(P_{t+h} | F_t^V) ],

where F_t^V is the recent wind-speed state selected by a Gaussian AR(p) model.
The calculation uses an empirical power curve m(v), an empirical conditional
residual variance s^2(v), exact AR(p) state-space propagation, and
Gauss-Hermite quadrature.  It also reports the MSE of the plug-in predictor
m(E[V_{t+h}|F_t^V]); this is reported separately and is NOT the bound.

Run from the Paper1-Q1 root, after the current unrelated 15-turbine run ends:
  python3 scripts/07_windonly_bound.py --n-boot 1000
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss


DATASETS = {
    "kelmarsh": "data/Kelmarsh_T1_2018_hourly.csv",
    "penmanshiel": "data/Penmanshiel_T01_2018_hourly.csv",
}
RATED_KW = {"kelmarsh": 2050.0, "penmanshiel": 2050.0}
DEFAULT_HORIZONS = (1, 2, 3, 6, 12, 18, 24, 36, 48)


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


def fit_ar(index: pd.DatetimeIndex, v: np.ndarray, order: int,
           target_ids: np.ndarray | None = None):
    """OLS Gaussian AR(p) on contiguous lag/response rows.

    ``target_ids`` permits information-criterion comparisons on a common
    response set.  This is essential when timestamps contain gaps: comparing
    total log likelihoods computed from different numbers of observations
    would spuriously favour models requiring longer lag windows.
    """
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
    # einsum avoids spurious BLAS floating-point warnings observed on macOS
    # for otherwise finite, small AR design matrices.
    resid = y - np.einsum("ij,j->i", x, phi)
    sigma2 = float(np.mean(resid**2))
    n = len(resid)
    ll = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)
    bic = (order + 1) * math.log(n) - 2.0 * ll
    return mu, np.asarray(phi, dtype=float), sigma2, bic


def choose_ar_order(index: pd.DatetimeIndex, v: np.ndarray, pmax: int):
    """Select AR order by BIC on a common, gap-aware response set.

    Every candidate is scored where its full lag vector is observable for the
    largest candidate order.  The selected order is then refit using all of
    its own admissible pre-test transitions for the final forecast model.
    """
    comparison_ids = contiguous_windows(index, pmax + 1)
    if len(comparison_ids) <= pmax + 2:
        raise ValueError("too few common contiguous rows for AR-order comparison")
    comparison_fits = {
        p: fit_ar(index, v, p, target_ids=comparison_ids)
        for p in range(1, pmax + 1)
    }
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
    e1 = np.zeros(len(phi)); e1[0] = 1.0
    ah = np.linalg.matrix_power(a, horizon)[0, :]
    ak = np.eye(len(phi))
    variance = 0.0
    for _ in range(horizon):
        variance += sigma2 * float((ak[0, 0]) ** 2)
        ak = ak @ a
    return ah, variance


def empirical_curve(v: np.ndarray, p: np.ndarray, rated: float, nbins: int, min_count: int):
    """Return m(v) and s2(v), both fitted only to the supplied training vectors.

    Physical boundary convention: zero mean/variance at non-positive speed;
    constant extrapolation above the last populated bin.  This convention is
    explicit so that quadrature support outside the empirical grid is auditable.
    """
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
        clipped = np.maximum(q, 0.0)
        out = np.interp(clipped, x, y, left=0.0, right=y[-1])
        return np.where(q <= 0.0, 0.0, out)

    def residual_variance(q):
        q = np.asarray(q, dtype=float)
        clipped = np.maximum(q, 0.0)
        out = np.interp(clipped, x, s2, left=0.0, right=s2[-1])
        return np.where(q <= 0.0, 0.0, out)

    return mean_power, residual_variance


def outer_states(index: pd.DatetimeIndex, v: np.ndarray, order: int, mu: float) -> np.ndarray:
    ids = contiguous_windows(index, order)
    if len(ids) == 0:
        raise ValueError("no contiguous wind states for outer expectation")
    return np.column_stack([v[ids - lag] - mu for lag in range(order)])


def bound_for_horizon(states, mean_power, residual_variance, mu, phi, sigma2, horizon, rated, nodes):
    weights_x, weights = hermgauss(nodes)
    weights = weights / math.sqrt(math.pi)
    ah, var_u = ar_forecast_moments(phi, sigma2, horizon)
    # states contain centred wind speed; power-curve input is physical speed.
    mu_h = mu + np.einsum("ij,j->i", states, ah)
    sd_u = math.sqrt(max(var_u, 0.0))
    u = mu_h[:, None] + math.sqrt(2.0) * sd_u * weights_x[None, :]
    m_u = mean_power(u)
    e_m = np.einsum("ij,j->i", m_u, weights)
    e_m2 = np.einsum("ij,j->i", m_u * m_u, weights)
    curve_var = np.maximum(e_m2 - e_m * e_m, 0.0)
    residual = np.einsum("ij,j->i", residual_variance(u), weights)
    plugin = np.einsum("ij,j->i", (m_u - mean_power(mu_h)[:, None]) ** 2, weights)
    scale = 100.0 / rated
    return {
        "curve_sd_pct": math.sqrt(float(np.mean(curve_var))) * scale,
        "residual_sd_pct": math.sqrt(float(np.mean(residual))) * scale,
        "bound_pct": math.sqrt(float(np.mean(curve_var + residual))) * scale,
        "plugin_mse_pct": math.sqrt(float(np.mean(plugin))) * scale,
        "cond_wind_sd_ms": sd_u,
    }


def estimate(df: pd.DataFrame, rated: float, horizons, pmax: int, nodes: int, nbins: int, min_count: int):
    index = pd.DatetimeIndex(df.index)
    v = df["WindSpeed_m_s"].to_numpy(dtype=float)
    p = df["ActivePower_kW"].to_numpy(dtype=float)
    order, (mu, phi, sigma2, bic), all_bic, n_bic_comparison_rows = choose_ar_order(index, v, pmax)
    mean_power, residual_variance = empirical_curve(v, p, rated, nbins, min_count)
    states = outer_states(index, v, order, mu)
    rows = []
    for h in horizons:
        r = bound_for_horizon(states, mean_power, residual_variance, mu, phi, sigma2, h, rated, nodes)
        rows.append({
            "horizon_h": h, "ar_order": order, "ar_bic": bic,
            "n_train_hours": len(df), "n_bic_comparison_rows": n_bic_comparison_rows,
            "n_outer_states": len(states),
            **r,
        })
    return pd.DataFrame(rows), all_bic


def moving_block_sample(df: pd.DataFrame, block_length: int, rng: np.random.Generator) -> pd.DataFrame:
    """Moving-block resample while marking joins as gaps.

    A naïve block bootstrap gives every resampled row a continuous synthetic
    hourly timestamp.  That would falsely make the random joins between blocks
    look like observed AR transitions and biases a persistence-based bound.
    Here blocks are sampled without circular wrapping, and a two-hour timestamp
    gap separates every synthetic block.  AR fitting and outer states therefore
    use transitions inside blocks only; the empirical power curve still uses
    every resampled observation.
    """
    n = len(df)
    if not 2 <= block_length < n:
        raise ValueError("block_length must be between 2 and n_train_hours - 1")
    starts = rng.integers(0, n - block_length + 1, size=math.ceil(n / block_length))
    pieces, times = [], []
    cursor = pd.Timestamp("2000-01-01")
    remaining = n
    for start in starts:
        take = min(block_length, remaining)
        pieces.append(df.iloc[start:start + take])
        times.extend(pd.date_range(cursor, periods=take, freq="h"))
        # End-to-next-start difference is two hours, so sequence construction
        # cannot cross a random bootstrap join.
        cursor += pd.Timedelta(hours=take + 1)
        remaining -= take
        if remaining == 0:
            break
    out = pd.concat(pieces, axis=0).copy()
    out.index = pd.DatetimeIndex(times)
    return out


def bootstrap_intervals(df, rated, horizons, pmax, nodes, nbins, min_count, n_boot, block_length, seed):
    rng = np.random.default_rng(seed)
    values = {h: [] for h in horizons}
    for b in range(n_boot):
        sampled = moving_block_sample(df, block_length, rng)
        boot, _ = estimate(sampled, rated, horizons, pmax, nodes, nbins, min_count)
        for row in boot.itertuples(index=False):
            values[row.horizon_h].append(row.bound_pct)
        if (b + 1) % max(1, n_boot // 10) == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")
    return {
        h: (float(np.percentile(values[h], 2.5)), float(np.percentile(values[h], 97.5)))
        for h in horizons
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="kelmarsh,penmanshiel")
    ap.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    ap.add_argument("--test-hours", type=int, default=1000)
    ap.add_argument("--pmax", type=int, default=5,
                    help="maximum candidate AR order; BIC uses one common response set")
    ap.add_argument("--nodes", type=int, default=64)
    ap.add_argument("--nbins", type=int, default=40)
    ap.add_argument("--min-bin-count", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=0, help="0 skips the end-to-end bootstrap")
    ap.add_argument("--block-length", type=int, default=48)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    horizons = [int(x) for x in args.horizons.split(",")]
    all_rows = []
    for site in [x.strip() for x in args.datasets.split(",")]:
        path = root / DATASETS[site]
        raw = pd.read_csv(path)
        raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
        raw = raw.sort_values("Timestamp").set_index("Timestamp")
        raw = raw[["WindSpeed_m_s", "ActivePower_kW"]].dropna()
        if len(raw) <= args.test_hours + args.pmax + 10:
            raise ValueError(f"{site}: not enough data after reserving test window")
        train = raw.iloc[:-args.test_hours].copy()
        estimate_df, bics = estimate(
            train, RATED_KW[site], horizons, args.pmax, args.nodes, args.nbins, args.min_bin_count
        )
        estimate_df.insert(0, "dataset", site)
        estimate_df["bic_by_order"] = ";".join(f"AR{p}:{bics[p]:.1f}" for p in sorted(bics))
        if args.n_boot:
            print(f"[{site}] end-to-end moving-block bootstrap")
            ci = bootstrap_intervals(
                train, RATED_KW[site], horizons, args.pmax, args.nodes, args.nbins,
                args.min_bin_count, args.n_boot, args.block_length, args.seed,
            )
            estimate_df["bound_ci95_low_pct"] = [ci[h][0] for h in estimate_df["horizon_h"]]
            estimate_df["bound_ci95_high_pct"] = [ci[h][1] for h in estimate_df["horizon_h"]]
        all_rows.append(estimate_df)
        print(f"[{site}] BIC selected AR({estimate_df['ar_order'].iloc[0]}) using {len(train)} pre-test rows")

    out = pd.concat(all_rows, ignore_index=True)
    output = root / "results" / "07_windonly_bound.csv"
    output.parent.mkdir(exist_ok=True)
    out.to_csv(output, index=False)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
