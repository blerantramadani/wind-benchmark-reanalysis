"""Summary tables (Markdown) and headline counts from the result files.

Writes outputs/results/tables.md and outputs/results/summary.json. Every number
quoted in the author response is read from these two files (see CLAIMS.md).

With --blocks 24,48,96 it writes the block-length sensitivity instead
(outputs/results/block_sensitivity.md and .csv): for each block length, the
number of 95 % intervals above and below zero and their median width, for the
selected method (full intervals) and the model forecast (test-only intervals),
and every cell whose classification differs from the 48-h blocks.

Usage: python -m wbr.tables [--blocks 24,48,96]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd

from . import config as C
from .benchmark import SEASONS
from .results import POOLED

NAMES = {"WindPersistenceCurve": "persistence", "LinearRegression": "LR", "MLP": "MLP",
         "Bayes_main": "model forecast (main fit)", "Bayes_sens": "model forecast (training fit)",
         "Selected": "selected method"}
YEARS = [str(y) for y in C.TEST_YEARS]


def _f(x):
    return f"{x:+.2f}"


def _picks(p):
    return ", ".join(f"{NAMES.get(m, m)} ×{n}" for m, n in Counter(str(p).split(",")).most_common())


def _ci(r, kind=""):
    return f"[{r[f'gap_ci95{kind}_low_pp']:+.2f}, {r[f'gap_ci95{kind}_high_pp']:+.2f}]"


def _side(low, high):
    return "above" if low > 0 else "below" if high < 0 else "includes 0"


def block_sensitivity(blocks):
    ref = C.P["bootstrap"]["block_hours"]
    keys = ["site", "target", "period", "horizon_h", "method"]
    kinds = {"Selected": "", "Bayes_main": "_testonly"}
    frames = {}
    for L in blocks:
        d = pd.read_csv(C.RESULTS / f"gaps_with_intervals_L{L}.csv")
        d = d[d.method.isin(kinds) & (d.period.isin(YEARS) | (d.period == POOLED))].copy()
        d["low"] = [r[f"gap_ci95{kinds[r.method]}_low_pp"] for _, r in d.iterrows()]
        d["high"] = [r[f"gap_ci95{kinds[r.method]}_high_pp"] for _, r in d.iterrows()]
        d["side"] = [_side(lo, hi) for lo, hi in zip(d.low, d.high)]
        frames[L] = d
    rows = []
    for L, d in frames.items():
        for (target, method), g in d.groupby(["target", "method"], sort=False):
            for scope, sub in (("pooled 2019-2023", g[g.period == POOLED]), ("single years", g[g.period != POOLED])):
                rows.append({"block_hours": L, "target": target, "method": method, "scope": scope, "cells": len(sub),
                             "above_zero": int((sub.side == "above").sum()), "below_zero": int((sub.side == "below").sum()),
                             "median_width_pp": float((sub.high - sub.low).median())})
    table = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate([*C.TARGETS, *kinds, "pooled 2019-2023", "single years"])}
    table = table.sort_values(["target", "method", "scope", "block_hours"], key=lambda c: c.map(order) if c.name != "block_hours" else c,
                              kind="stable").reset_index(drop=True)
    table.to_csv(C.RESULTS / "block_sensitivity.csv", index=False)
    out = ["# Block-length sensitivity of the bootstrap intervals", "",
           "Selected method: full intervals (benchmark refitted in each resample). Model forecast (main fit): "
           "test-only intervals, as in the calibration check.", "",
           "| Target | Method | Scope | Cells | " + " | ".join(f"L = {L} h: above / below 0, median width (pp)" for L in blocks) + " |",
           "|---|---|---|---:|" + "---|" * len(blocks)]
    for (target, method, scope), g in table.groupby(["target", "method", "scope"], sort=False):
        g = g.set_index("block_hours")
        cells = " | ".join(f"{g.loc[L, 'above_zero']} / {g.loc[L, 'below_zero']}, {g.loc[L, 'median_width_pp']:.2f}" for L in blocks)
        out.append(f"| {target} | {NAMES.get(method, method)} | {scope} | {g.cells.iloc[0]} | {cells} |")
    if ref in frames:
        out += ["", f"Cells whose classification differs from L = {ref} h:", ""]
        changed = 0
        for L in blocks:
            if L == ref:
                continue
            m = frames[ref].merge(frames[L], on=keys, suffixes=("_ref", "_L"), validate="one_to_one")
            for _, r in m[m.side_ref != m.side_L].iterrows():
                changed += 1
                out.append(f"- L = {L} h, {r.site} {r.target} {r.period} h = {r.horizon_h}, {NAMES.get(r.method, r.method)}: "
                           f"[{r.low_ref:+.2f}, {r.high_ref:+.2f}] ({r.side_ref}) → [{r.low_L:+.2f}, {r.high_L:+.2f}] ({r.side_L})")
        if not changed:
            out.append("- none")
    (C.RESULTS / "block_sensitivity.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--blocks", help="block lengths to compare, for example 24,48,96")
    args = ap.parse_args(argv)
    if args.blocks:
        block_sensitivity([int(b) for b in args.blocks.split(",")])
        return 0
    d = pd.read_csv(C.RESULTS / f"gaps_with_intervals_L{C.P['bootstrap']['block_hours']}.csv")
    bsum = pd.read_csv(C.BENCHMARK / "benchmark_summary.csv")
    out, summary = [], {}
    for target in C.TARGETS:
        t = d[d.target == target]
        pooled = t[t.period == POOLED]
        sel, bay = pooled[pooled.method == "Selected"], pooled[pooled.method == "Bayes_main"]
        out += [f"## Target: {target}", "", f"### {POOLED}, method selected in each test year", "",
                "| Site | h | Benchmark (%) | Methods selected | nRMSE (%) | Gap (pp) | 95 % interval | Model forecast gap (pp) | 95 %, test only |",
                "|---|---:|---:|---|---:|---:|---|---:|---|"]
        for site in C.SITES:
            for h in C.HORIZONS:
                s = sel[(sel.site == site) & (sel.horizon_h == h)].iloc[0]
                b = bay[(bay.site == site) & (bay.horizon_h == h)].iloc[0]
                out.append(f"| {site.capitalize()} | {h} | {s.bench_main_pct:.2f} | {_picks(s.picks)} | {s.rmse_pct:.2f} | "
                           f"{_f(s.gap_main_pp)} | {_ci(s)} | {_f(b.gap_main_pp)} | {_ci(b, '_testonly')} |")
        yr = t[t.period.isin(YEARS) & (t.method == "Selected")]
        out += ["", "### Single test years, selected method: gap (pp)", "",
                "| Site | h | " + " | ".join(YEARS) + " |", "|---|---:|" + "---:|" * len(YEARS)]
        for site in C.SITES:
            for h in C.HORIZONS:
                vals = [yr[(yr.site == site) & (yr.horizon_h == h) & (yr.period == y)].gap_main_pp.iloc[0] for y in YEARS]
                out.append(f"| {site.capitalize()} | {h} | " + " | ".join(_f(v) for v in vals) + " |")
        st = t[t.period.str.startswith("season") & (t.method == "Bayes_main")]
        out += ["", "### By season, all years: model forecast gap (pp), annual AR / seasonal AR", "",
                "| Site | h | " + " | ".join(SEASONS) + " |", "|---|---:|" + "---|" * len(SEASONS)]
        for site in C.SITES:
            for h in C.HORIZONS:
                cells = []
                for sn in SEASONS:
                    r = st[(st.site == site) & (st.horizon_h == h) & (st.period == f"season_{sn}")].iloc[0]
                    cells.append(f"{_f(r.gap_main_pp)} / {_f(r.gap_seasonal_ar_pp)}")
                out.append(f"| {site.capitalize()} | {h} | " + " | ".join(cells) + " |")
        bt = bsum[bsum.target == target]
        out += ["", "### Benchmark at the test origins (%), by test year", "",
                "| Site | h | " + " | ".join(YEARS) + " | AR order |", "|---|---:|" + "---:|" * len(YEARS) + "---|"]
        for site in C.SITES:
            for h in C.HORIZONS:
                s = bt[(bt.site == site) & (bt.horizon_h == h)].sort_values("test_year")
                out.append(f"| {site.capitalize()} | {h} | " + " | ".join(f"{v:.2f}" for v in s.bench_test_main_pct)
                           + f" | {','.join(map(str, s.ar_order_main))} |")
        out.append("")
        summary[target] = {
            "pooled_selected_gap_range_pp": [round(sel.gap_main_pp.min(), 2), round(sel.gap_main_pp.max(), 2)],
            "pooled_selected_intervals_above_zero": int((sel.gap_ci95_low_pp > 0).sum()),
            "pooled_selected_intervals_below_zero": int((sel.gap_ci95_high_pp < 0).sum()),
            "pooled_model_forecast_gap_range_pp": [round(bay.gap_main_pp.min(), 2), round(bay.gap_main_pp.max(), 2)],
            "pooled_model_forecast_testonly_above_zero": int((bay.gap_ci95_testonly_low_pp > 0).sum()),
            "pooled_model_forecast_testonly_below_zero": int((bay.gap_ci95_testonly_high_pp < 0).sum()),
            "single_year_selected_gap_range_pp": [round(yr.gap_main_pp.min(), 2), round(yr.gap_main_pp.max(), 2)],
            "single_year_selected_intervals_above_zero": int((yr.gap_ci95_low_pp > 0).sum()),
            "single_year_selected_intervals_below_zero": int((yr.gap_ci95_high_pp < 0).sum()),
            "single_year_cells": int(len(yr)),
        }
    (C.RESULTS / "tables.md").write_text("\n".join(out), encoding="utf-8")
    (C.RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
