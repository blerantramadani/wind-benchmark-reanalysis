"""Reproduction check against the published results in expected/.

Point estimates are always checked: they are rebuilt from the stored forecasts
and benchmark variances (outputs/forecasts, outputs/benchmark) and compared
with expected/point_estimates.csv. With --intervals, the merged file of a full
run (outputs/results/gaps_with_intervals_L48.csv) is also compared with
expected/gaps_with_intervals_L48.csv. Every column is compared; the check fails
if a row is missing, a text column differs (for example the methods selected in
each year) or a numeric difference exceeds the tolerance (default 1e-9 pp).

Preprocessing, splits, benchmark, model forecasts, persistence, linear
regression and the bootstrap are deterministic; on another processor or BLAS
build their last digits can differ, which may exceed 1e-9 in the intervals but
stays far below the two decimals of the reply. MLP training on a CPU does not
repeat exactly from run to run, which is why the MLP forecasts are archived.

The sequence networks of 2019 are candidates in the model selection, and the
results are compared with expected/revision/. With --no-networks they are left
out, as in the reply to RC1, and the results are compared with expected/.

With --y2018, every file in expected/y2018 is compared with the same file in
outputs/y2018 written by ``wbr y2018`` (the analyses of the preprint's 2018
configuration).

Usage: python -m wbr.check [--intervals | --y2018] [--no-networks] [--tolerance 1e-9]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config as C
from .results import load_cells, point_rows
from .split import load_grid

KEYS = ["site", "target", "period", "horizon_h", "method"]


def compare(want: pd.DataFrame, got: pd.DataFrame, label: str, tolerance: float) -> bool:
    want, got = want.astype({"period": str}), got.astype({"period": str})
    m = want.merge(got, on=KEYS, how="outer", suffixes=("_expected", "_rebuilt"), indicator=True)
    missing = int((m["_merge"] != "both").sum())
    worst, text_equal = 0.0, True
    for col in [c for c in want.columns if c not in KEYS]:
        if f"{col}_rebuilt" not in m:
            missing += 1
            continue
        a, b = m[f"{col}_expected"], m[f"{col}_rebuilt"]
        if want[col].dtype.kind in "fiu":
            a, b = a.astype(float), b.astype(float)
            if (a.isna() != b.isna()).any():
                worst = float("inf")
            both = a.notna() & b.notna()
            if both.any():
                worst = max(worst, float((a[both] - b[both]).abs().max()))
        else:
            text_equal &= bool((a.fillna("").astype(str) == b.fillna("").astype(str)).all())
    ok = missing == 0 and worst <= tolerance and text_equal
    print(f"{label}: rows {len(m)}, unmatched rows or columns {missing}, largest difference {worst:.3e}, "
          f"text columns identical: {text_equal}")
    return ok


def _same(a, b, tol) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= tol
    return a == b


def check_y2018(tolerance: float) -> bool:
    """Compare every file in expected/y2018 with the same file in outputs/y2018 (rows in the same order)."""
    ok = True
    for want_path in sorted((C.ROOT / "expected" / "y2018").iterdir()):
        got_path = C.OUT / "y2018" / want_path.name
        if not got_path.exists():
            print(f"y2018/{want_path.name}: missing in outputs")
            ok = False
            continue
        if want_path.suffix == ".json":
            same = _same(json.loads(want_path.read_text()), json.loads(got_path.read_text()), tolerance)
            print(f"y2018/{want_path.name}: {'identical within tolerance' if same else 'DIFFERENT'}")
            ok &= same
            continue
        want, got = pd.read_csv(want_path), pd.read_csv(got_path)
        same = list(want.columns) == list(got.columns) and len(want) == len(got)
        worst = 0.0
        if same:
            for col in want.columns:
                a, b = want[col], got[col]
                if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
                    if (a.isna() != b.isna()).any():
                        same = False
                    both = a.notna() & b.notna()
                    if both.any():
                        worst = max(worst, float((a[both].astype(float) - b[both].astype(float)).abs().max()))
                else:
                    same &= bool((a.fillna("").astype(str) == b.fillna("").astype(str)).all())
        same &= worst <= tolerance
        print(f"y2018/{want_path.name}: rows {len(want)}, largest difference {worst:.3e}, {'ok' if same else 'DIFFERENT'}")
        ok &= same
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--intervals", action="store_true", help="also compare the intervals of a full run")
    ap.add_argument("--y2018", action="store_true", help="compare outputs/y2018 with expected/y2018 instead")
    ap.add_argument("--no-networks", action="store_true", help="leave the 2019 sequence networks out")
    ap.add_argument("--tolerance", type=float, default=1e-9)
    args = ap.parse_args(argv)
    if args.y2018:
        ok = check_y2018(args.tolerance)
        print("REPRODUCED" if ok else "MISMATCH")
        return 0 if ok else 1
    want = C.ROOT / "expected" if args.no_networks else C.ROOT / "expected" / "revision"
    if not (want / "point_estimates.csv").exists():
        print(f"{want.relative_to(C.ROOT)}/point_estimates.csv does not exist; "
              "use --no-networks to compare with the results of the reply in expected/")
        return 1
    got = pd.concat(point_rows(s, t, load_cells(load_grid(s), t, not args.no_networks))
                    for s in C.SITES for t in C.TARGETS)
    ok = compare(pd.read_csv(want / "point_estimates.csv"), got, "point estimates", args.tolerance)
    if args.intervals:
        name = f"gaps_with_intervals_L{C.P['bootstrap']['block_hours']}.csv"
        ok &= compare(pd.read_csv(want / name), pd.read_csv(C.RESULTS / name), "intervals", args.tolerance)
    print("REPRODUCED" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
