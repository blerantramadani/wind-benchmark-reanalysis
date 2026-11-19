#!/usr/bin/env python3
"""Create the AR-order-consistent geometry figure for the final manuscript.

The figure is illustrative: for each site it shows the empirical power curve
and the conditional future-wind densities at one and forty-eight hours for a
representative observed AR state.  All quantities use the same common-support
BIC selection and empirical curve as the reported benchmark.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
spec = importlib.util.spec_from_file_location("windonly_benchmark", SCRIPT_DIR / "07_windonly_bound.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load 07_windonly_bound.py")
BENCH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BENCH)


def representative_state(states: np.ndarray, phi: np.ndarray, mu: float) -> np.ndarray:
    """Choose an observed state with a one-hour conditional mean near mu."""
    ah, _ = BENCH.ar_forecast_moments(phi, 1.0, 1)
    means = mu + np.einsum("ij,j->i", states, ah)
    return states[np.argmin(np.abs(means - mu))]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), constrained_layout=True)
    density_colours = {1: "#377eb8", 48: "#e66101"}

    for col, site in enumerate(("kelmarsh", "penmanshiel")):
        raw = pd.read_csv(ROOT / BENCH.DATASETS[site])
        raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
        train = (raw.sort_values("Timestamp").set_index("Timestamp")
                 [["WindSpeed_m_s", "ActivePower_kW"]].dropna().iloc[:-1000])
        index = pd.DatetimeIndex(train.index)
        wind = train["WindSpeed_m_s"].to_numpy(float)
        power = train["ActivePower_kW"].to_numpy(float)
        order, (mu, phi, sigma2, _), _, _ = BENCH.choose_ar_order(index, wind, pmax=5)
        states = BENCH.outer_states(index, wind, order, mu)
        state = representative_state(states, phi, mu)
        mean_power, _ = BENCH.empirical_curve(wind, power, BENCH.RATED_KW[site], 40, 20)
        upper = float(np.percentile(wind, 99.5))
        grid = np.linspace(0.0, upper, 800)
        curve = mean_power(grid)

        ax = axes[0, col]
        ax.plot(grid, curve, color="0.15", lw=2.0, label="empirical power curve", zorder=3)
        density_ax = ax.twinx()
        for horizon in (1, 48):
            ah, variance = BENCH.ar_forecast_moments(phi, sigma2, horizon)
            conditional_mean = float(mu + np.einsum("i,i->", state, ah))
            sd = math.sqrt(variance)
            density = np.exp(-0.5 * ((grid - conditional_mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
            density_ax.fill_between(grid, 0, density, color=density_colours[horizon], alpha=0.25)
            density_ax.plot(grid, density, color=density_colours[horizon], lw=1.35,
                            label=f"h = {horizon} h (sd = {sd:.2f} m/s)")
        ax.set_title(f"{site.capitalize()} — AR({order})")
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("Power (kW)")
        density_ax.set_ylabel("Conditional density")
        ax.set_xlim(0, upper)
        ax.set_ylim(0, BENCH.RATED_KW[site] * 1.05)
        density_ax.set_ylim(bottom=0)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = density_ax.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
        ax.spines[["top"]].set_visible(False)

        ax = axes[1, col]
        ax.plot(grid, curve, color="0.15", lw=2.0, label="empirical power curve")
        ah48, var48 = BENCH.ar_forecast_moments(phi, sigma2, 48)
        centre = float(mu + np.einsum("i,i->", state, ah48))
        d1 = np.gradient(curve, grid)
        d2 = np.gradient(d1, grid)
        centre_i = int(np.argmin(np.abs(grid - centre)))
        quadratic = curve[centre_i] + d1[centre_i] * (grid - centre) + 0.5 * d2[centre_i] * (grid - centre) ** 2
        ax.plot(grid, quadratic, color="#e66101", lw=1.6, ls="--",
                label="local quadratic expansion")
        sd48 = math.sqrt(var48)
        ax.axvspan(max(0, centre - 2 * sd48), min(upper, centre + 2 * sd48),
                   color=density_colours[48], alpha=0.16, label="48 h conditional range")
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("Power (kW)")
        ax.set_xlim(0, upper)
        ax.set_ylim(-0.12 * BENCH.RATED_KW[site], BENCH.RATED_KW[site] * 1.08)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig_pdf_over_curve.{ext}", dpi=300, bbox_inches="tight")
    print("Wrote figures/fig_pdf_over_curve.png and .pdf")


if __name__ == "__main__":
    main()
