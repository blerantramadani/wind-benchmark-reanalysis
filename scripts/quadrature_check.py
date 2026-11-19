"""Check that 64 Gauss-Hermite nodes are enough.

For every site, test year, target and horizon (180 cells) the main benchmark
and the model forecast E[m(V)] (and its nRMSE on the test hours with a valid
target) are evaluated on the test origins with 64, 128
and 256 nodes. The script writes one row per cell and prints the largest
change against 256 nodes, in percentage points of rated power.

Usage: python scripts/quadrature_check.py [--sites kelmarsh]
Output: outputs/checks/quadrature_<sites>.csv
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

from wbr import benchmark as B
from wbr import config as C
from wbr.split import history, load_grid, split

NODES = (64, 128, 256)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sites", default=",".join(C.SITES))
    args = ap.parse_args(argv)
    out = C.ROOT / "outputs" / "checks"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for site in args.sites.split(","):
        g = load_grid(site)
        for year in C.TEST_YEARS:
            for target in C.TARGETS:
                for h in C.HORIZONS:
                    s = split(g, year, h, target)
                    fit = B.fit_reference(g, s.bench_cut_main, target)
                    hist = history(g, s.test)
                    actual = g.power[s.test + h - 1]
                    ok = np.isfinite(actual)
                    row = {"site": site, "test_year": year, "target": target, "horizon_h": h}
                    for n in NODES:
                        x, w = hermgauss(n)
                        B.GH_X, B.GH_W = x, w / math.sqrt(math.pi)
                        v = B.conditional_variance(fit["ar"], fit["m"], fit["s2"], hist, h)
                        mean, _ = B.conditional_mean(fit["ar"], fit["m"], hist, h)
                        row[f"bench_pct_{n}"] = math.sqrt(v.mean()) * C.SCALE
                        row[f"forecast_{n}"] = mean
                        row[f"forecast_nrmse_pct_{n}"] = math.sqrt(np.mean((mean[ok] - actual[ok]) ** 2)) * C.SCALE
                    for n in NODES[:-1]:
                        row[f"bench_diff_pp_{n}"] = row[f"bench_pct_{n}"] - row["bench_pct_256"]
                        row[f"forecast_nrmse_diff_pp_{n}"] = row[f"forecast_nrmse_pct_{n}"] - row["forecast_nrmse_pct_256"]
                        row[f"forecast_maxdiff_pp_{n}"] = float(np.max(np.abs(row[f"forecast_{n}"] - row["forecast_256"]))) * C.SCALE
                    for n in NODES:
                        del row[f"forecast_{n}"]
                    rows.append(row)
                    print(site, year, target, h, f"{row['bench_diff_pp_64']:+.2e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"quadrature_{args.sites.replace(',', '-')}.csv", index=False)
    for n in NODES[:-1]:
        print(f"{n} nodes vs 256: largest |benchmark change| {df[f'bench_diff_pp_{n}'].abs().max():.2e} pp, "
              f"largest |forecast change| at one origin {df[f'forecast_maxdiff_pp_{n}'].max():.2e} pp, "
              f"largest |forecast nRMSE change| {df[f'forecast_nrmse_diff_pp_{n}'].abs().max():.2e} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
