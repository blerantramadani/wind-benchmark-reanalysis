"""Gaps to the benchmark and their uncertainty (protocol, Sect. 8-9).

Point estimates
    For every site, target, test year and horizon: nRMSE of each method, the
    benchmark on the same origins (main, sensitivity and seasonal-AR fits) and
    the gap nRMSE - benchmark, in percentage points of rated power.
    "Selected" is the method with the lowest validation error in each test
    year, chosen among persistence, linear regression, MLP and the model
    forecast with the sensitivity fit (Bayes_main is not eligible because its
    fit includes the validation year). For the normal target in 2019, at the
    horizons 1, 3, 6, 12, 24 and 48 h, the four sequence networks
    (``wbr networks``) are also candidates; they exist only in that year, so
    they have no pooled or seasonal rows of their own.
    Pooled results for 2019-2023 combine the squared errors of the method
    selected in each year. Seasonal results pool all years by the
    meteorological season of the target hour. With --no-networks the sequence
    networks are left out, which gives the results of the reply to RC1.

Intervals
    Moving-block bootstrap with blocks of consecutive calendar hours of the
    test year (default 48 h, 1,000 resamples, fixed seed). Hours without a
    valid target stay in place and receive no weight, so a block never spans
    a gap. In every resample the main benchmark is also refitted on a block
    resample of its fitting window, in which real gaps are kept and blocks are
    separated so that the AR model cannot fit across a join. For speed, this
    refit uses the cut-off of h = 48 for all horizons. Test-only intervals
    (fitted benchmark fixed) are reported as well.

Usage: python -m wbr.results --points                      (point estimates)
       python -m wbr.results --sites kelmarsh --targets normal (intervals, one job)
       python -m wbr.results --merge                       (point estimates + intervals)
       add --no-networks to leave the sequence networks out, --block 24 or 96 for the block-length sensitivity
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd

from . import config as C
from .benchmark import SEASONS, conditional_variance, empirical_curve, wind_model
from .networks import cell_is_done, cell_stem
from .split import history, load_grid, season, split

NET = C.P["networks_2019"]
NETWORKS = tuple(NET["models"])
METHODS = ("WindPersistenceCurve", "LinearRegression", "MLP", *NETWORKS, "Bayes_main", "Bayes_sens", "Selected")
SELECTABLE = ("WindPersistenceCurve", "LinearRegression", "MLP", *NETWORKS, "Bayes_sens")
POOLED = f"{C.TEST_YEARS[0]}-{C.TEST_YEARS[-1]}"


def add_networks(site, year, target, h, te, va):
    """Add the seed-mean forecasts of the sequence networks to the test and validation frames of one cell."""
    for name in NETWORKS:
        if not cell_is_done(site, name, h):
            raise FileNotFoundError(f"{cell_stem(site, name, h)} is not finished: run `wbr networks --site {site}`, "
                                    "or pass --no-networks to leave the sequence networks out")
        for part, df in (("val", va), ("test", te)):
            net = pd.read_csv(C.NETWORKS / f"{cell_stem(site, name, h)}_{part}.csv")
            if not (np.array_equal(net["origin"].to_numpy(), df["origin"].to_numpy())
                    and np.allclose(net["actual"], df["actual"], rtol=0.0, atol=1e-9)):
                raise ValueError(f"{cell_stem(site, name, h)}_{part}: origins or targets differ from the forecast file")
            df[name] = net[name].to_numpy()


def select(val):
    """The eligible method with the lowest validation error (on a tie, the first in SELECTABLE)."""
    return min((m for m in SELECTABLE if m in val), key=val.get)


def common_methods(cells, years, h):
    """Methods with forecasts in every listed year (the sequence networks exist only in 2019)."""
    return [m for m in METHODS if all(m in cells[(y, h)]["e2"] for y in years)]


def load_cells(g, target, networks=True):
    cells = {}
    for year in C.TEST_YEARS:
        for h in C.HORIZONS:
            stem = f"{g.site}_{year}_{target}_h{h}"
            te = pd.read_csv(C.FORECASTS / f"{stem}_test.csv")
            va = pd.read_csv(C.FORECASTS / f"{stem}_val.csv")
            bt = pd.read_csv(C.FORECASTS / f"{stem}_bayes_test.csv")
            bv = pd.read_csv(C.FORECASTS / f"{stem}_bayes_val.csv")
            bm = pd.read_csv(C.BENCHMARK / f"{stem}.csv")
            if not (np.array_equal(te["origin"], bt["origin"]) and np.array_equal(te["origin"], bm["origin"])
                    and np.array_equal(va["origin"], bv["origin"])):
                raise ValueError(f"{stem}: forecast and benchmark origins differ")
            for col in ("Bayes_main", "Bayes_sens"):
                te[col], va[col] = bt[col].to_numpy(), bv[col].to_numpy()
            if networks and year == NET["test_year"] and target == NET["target"] and h in NET["horizons_h"]:
                add_networks(g.site, year, target, h, te, va)
            e2 = {m: (te["actual"] - te[m]).to_numpy() ** 2 for m in METHODS if m in te}
            val = {m: math.sqrt(((va["actual"] - va[m]) ** 2).mean()) * C.SCALE for m in METHODS if m in va}
            pick = select(val)
            e2["Selected"], val["Selected"] = e2[pick], val[pick]
            o = te["origin"].to_numpy()
            cells[(year, h)] = {"origin": o, "tpos": o + h - 1, "e2": e2, "val": val, "pick": pick,
                                "var": {k: bm[k].to_numpy() for k in ("var_main", "var_sens", "var_seasonal_ar")},
                                "season": bm["season_target"].to_numpy(), "hist": history(g, o)}
    return cells


def _row(site, target, label, h, m, e2, var, n, val=np.nan, picks=""):
    r = math.sqrt(e2.mean()) * C.SCALE
    b = {k: math.sqrt(v.mean()) * C.SCALE if v is not None else np.nan for k, v in var.items()}
    return {"site": site, "target": target, "period": label, "horizon_h": h, "method": m, "n": n,
            "rmse_pct": r, "val_rmse_pct": val, "picks": picks,
            "bench_main_pct": b["var_main"], "bench_sens_pct": b["var_sens"], "bench_seasonal_ar_pct": b["var_seasonal_ar"],
            "gap_main_pp": r - b["var_main"], "gap_sens_pp": r - b["var_sens"], "gap_seasonal_ar_pp": r - b["var_seasonal_ar"]}


def point_rows(site, target, cells):
    rows = []
    for h in C.HORIZONS:
        for label, years in [(str(y), [y]) for y in C.TEST_YEARS] + [(POOLED, list(C.TEST_YEARS))]:
            cs = [cells[(y, h)] for y in years]
            var = {k: np.concatenate([c["var"][k] for c in cs]) for k in ("var_main", "var_sens", "var_seasonal_ar")}
            for m in common_methods(cells, years, h):
                rows.append(_row(site, target, label, h, m, np.concatenate([c["e2"][m] for c in cs]), var,
                                 int(sum(len(c["origin"]) for c in cs)), float(np.mean([c["val"][m] for c in cs])),
                                 ",".join(c["pick"] for c in cs) if m == "Selected" else ""))
        cs = [cells[(y, h)] for y in C.TEST_YEARS]
        seas = np.concatenate([c["season"] for c in cs])
        for sn in SEASONS:
            k = seas == sn
            var = {"var_main": np.concatenate([c["var"]["var_main"] for c in cs])[k], "var_sens": None,
                   "var_seasonal_ar": np.concatenate([c["var"]["var_seasonal_ar"] for c in cs])[k]}
            for m in common_methods(cells, C.TEST_YEARS, h):
                rows.append(_row(site, target, f"season_{sn}", h, m, np.concatenate([c["e2"][m] for c in cs])[k],
                                 var, int(k.sum())))
    return pd.DataFrame(rows)


def block_weights(positions, n_total, block, rng):
    """Weights after a moving-block resample of ordered calendar positions.

    Positions are split into runs of consecutive hours (one run for a calendar
    year; several for a season such as Jan-Feb plus Dec). A block never crosses
    from one run to the next; a run shorter than the block is taken whole.
    """
    positions = np.asarray(positions)
    n = len(positions)
    breaks = np.flatnonzero(np.diff(positions) != 1) + 1
    runs = np.split(np.arange(n), breaks)
    starts, lengths = [], []
    for r in runs:
        if len(r) >= block:
            starts.append(np.arange(r[0], r[-1] - block + 2)); lengths.append(np.full(len(r) - block + 1, block))
        else:
            starts.append(np.array([r[0]])); lengths.append(np.array([len(r)]))
    starts, lengths = np.concatenate(starts), np.concatenate(lengths)
    pick = list(rng.integers(0, len(starts), size=math.ceil(n / block)))
    while lengths[pick].sum() < n:  # only when short runs were drawn
        pick.append(rng.integers(0, len(starts)))
    idx = np.concatenate([np.arange(starts[k], starts[k] + lengths[k]) for k in pick])[:n]
    w = np.zeros(n_total)
    np.add.at(w, positions[idx], 1.0)
    return w


def resampled_fit(g, last_pos, target, block, rng):
    first = int(np.flatnonzero(g.valid)[0])
    n = last_pos - first + 1
    starts = first + rng.integers(0, n - block + 1, size=math.ceil(n / block))
    pos = (starts[:, None] + np.arange(block)[None, :]).ravel()
    fake = (np.arange(len(starts))[:, None] * (block + 2) + np.arange(block)[None, :]).ravel()
    keep_ar = g.valid[pos]
    keep_cv = keep_ar & (~g.stop[pos] if target == "normal" else True)
    ar = wind_model(pd.Timestamp("2000-01-01") + pd.to_timedelta(fake[keep_ar], unit="h"), g.wind[pos[keep_ar]])
    m, s2 = empirical_curve(g.wind[pos[keep_cv]], g.power[pos[keep_cv]], C.RATED, C.P["curve_bins"], C.P["curve_min_count"])
    return ar, m, s2


def bootstrap(site, target, g, cells, n_resamples, block, seed):
    rng = np.random.default_rng(seed)
    labels = [str(y) for y in C.TEST_YEARS] + [POOLED] + [f"season_{s}" for s in SEASONS]
    keys = [(lab, h, m) for lab in labels for h in C.HORIZONS for m in METHODS]
    draws = {k: [] for k in keys}
    draws_test = {k: [] for k in keys}
    common = {h: set(common_methods(cells, C.TEST_YEARS, h)) for h in C.HORIZONS}
    year_pos = {y: np.flatnonzero(g.time.year == y) for y in C.TEST_YEARS}
    season_pos = {(y, s): year_pos[y][season(g.time[year_pos[y]]) == s] for y in C.TEST_YEARS for s in SEASONS}
    for b in range(n_resamples):
        t0 = time.time()
        pooled = {(lab, h, m): np.zeros(4) for lab in [POOLED] + [f"season_{s}" for s in SEASONS]
                  for h in C.HORIZONS for m in METHODS}
        for y in C.TEST_YEARS:
            ar, mfun, s2fun = resampled_fit(g, split(g, y, 48, target).bench_cut_main, target, block, rng)
            wy = block_weights(year_pos[y], len(g.time), block, rng)
            ws = {s: block_weights(season_pos[(y, s)], len(g.time), block, rng)
                  for s in SEASONS if len(season_pos[(y, s)]) >= block}
            for h in C.HORIZONS:
                c = cells[(y, h)]
                var_b = conditional_variance(ar, mfun, s2fun, c["hist"], h)
                for lab, wvec in [(str(y), wy)] + [(f"season_{s}", ws[s]) for s in ws]:
                    w = wvec[c["tpos"]]
                    sw = w.sum()
                    if sw == 0:
                        continue
                    vb, vf = (w * var_b).sum(), (w * c["var"]["var_main"]).sum()
                    for m in c["e2"]:
                        eb = (w * c["e2"][m]).sum()
                        if lab == str(y):
                            draws[(lab, h, m)].append((math.sqrt(eb / sw) - math.sqrt(vb / sw)) * C.SCALE)
                            draws_test[(lab, h, m)].append((math.sqrt(eb / sw) - math.sqrt(vf / sw)) * C.SCALE)
                            if m in common[h]:
                                pooled[(POOLED, h, m)] += (eb, vb, sw, vf)
                        elif m in common[h]:
                            pooled[(lab, h, m)] += (eb, vb, sw, vf)
        for key, (eb, vb, sw, vf) in pooled.items():
            if sw > 0:
                draws[key].append((math.sqrt(eb / sw) - math.sqrt(vb / sw)) * C.SCALE)
                draws_test[key].append((math.sqrt(eb / sw) - math.sqrt(vf / sw)) * C.SCALE)
        if (b + 1) % 50 == 0 or b == 0:
            print(f"[{site} {target}] resample {b + 1}/{n_resamples} ({time.time() - t0:.1f} s each)", flush=True)
    rows = []
    for key in keys:
        if draws[key]:
            lo, hi = np.percentile(draws[key], [2.5, 97.5])
            lo_t, hi_t = np.percentile(draws_test[key], [2.5, 97.5])
            rows.append(dict(zip(("period", "horizon_h", "method"), key), site=site, target=target,
                             gap_ci95_low_pp=lo, gap_ci95_high_pp=hi,
                             gap_ci95_testonly_low_pp=lo_t, gap_ci95_testonly_high_pp=hi_t, n_resamples=len(draws[key])))
    return pd.DataFrame(rows)


def main(argv=None):
    bs = C.P["bootstrap"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", action="store_true", help="write point estimates for all sites and targets")
    ap.add_argument("--merge", action="store_true", help="merge point estimates with the interval files")
    ap.add_argument("--sites", default=",".join(C.SITES))
    ap.add_argument("--targets", default=",".join(C.TARGETS))
    ap.add_argument("--resamples", type=int, default=bs["resamples"])
    ap.add_argument("--block", type=int, default=bs["block_hours"])
    ap.add_argument("--seed", type=int, default=bs["seed"])
    ap.add_argument("--no-networks", action="store_true", help="leave the 2019 sequence networks out (results of the reply)")
    args = ap.parse_args(argv)
    nets = not args.no_networks
    C.ensure_dirs()
    if args.points:
        pts = [point_rows(site, t, load_cells(load_grid(site), t, nets)) for site in C.SITES for t in C.TARGETS]
        pd.concat(pts).to_csv(C.RESULTS / "point_estimates.csv", index=False)
        return
    if args.merge:
        pts = pd.read_csv(C.RESULTS / "point_estimates.csv")
        ci = pd.concat(pd.read_csv(p) for p in sorted(C.RESULTS.glob(f"intervals_L{args.block}_*.csv")))
        out = pts.merge(ci, on=["site", "target", "period", "horizon_h", "method"], how="left", validate="one_to_one")
        out.to_csv(C.RESULTS / f"gaps_with_intervals_L{args.block}.csv", index=False)
        print(f"rows without interval: {int(out['n_resamples'].isna().sum())}")
        return
    for site in args.sites.split(","):
        g = load_grid(site)
        for target in args.targets.split(","):
            cells = load_cells(g, target, nets)
            bootstrap(site, target, g, cells, args.resamples, args.block, args.seed).to_csv(
                C.RESULTS / f"intervals_L{args.block}_{site}_{target}.csv", index=False)


if __name__ == "__main__":
    main()
