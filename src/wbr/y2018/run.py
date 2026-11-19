"""Run the 2018 analyses and write their results to outputs/y2018/.

Usage: wbr y2018 [--retrain] [--threads 2]

Without --retrain everything runs from the preprint files, the archived
forecasts and the 2018 SCADA archives (a few minutes, most of it the
bootstrap of the matched comparison). --retrain first trains the five
networks at h = 48 (several hours, TensorFlow) and scores the new run instead
of the archived one.

When the networks retrained for the revision at the other horizons with
h > 1 are archived (archived/retrained_2018, written by
``python -m wbr.y2018.purging``), their scores and those at 48 h are written
to retrained_scores.csv.
"""
from __future__ import annotations

import argparse
import time

from . import common as K
from . import matched, purging, reproduction, states, taylor


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--retrain", action="store_true", help="retrain the networks at h = 48 before scoring them")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    out = K.OUT
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def step(name):
        print(f"== {name} ({time.time() - t0:.0f} s)", flush=True)

    step("reproduction of the published scores")
    scores, summary = reproduction.scores()
    scores.to_csv(out / "reproduction_scores.csv", index=False)
    reproduction.splits().to_csv(out / "reproduction_splits.csv", index=False)
    summary["hour_convention"] = reproduction.hour_convention()
    summary["test_window_wind"] = reproduction.test_window_wind()
    K.write_json(summary, out / "reproduction_summary.json")

    step("status logs (point 1)")
    logs, flags = states.status_logs()
    logs["notable_stops"] = states.notable_stops()
    logs["penmanshiel_missing_temperature"] = states.missing_temperature_hours()
    K.write_json(logs, out / "status_logs.json")
    for site, fl in flags.items():
        fl.to_csv(out / f"hours_and_events_{site}.csv")

    step("benchmark and errors on the same test origins (point 3b(ii))")
    res, best, per_origin, summary = matched.matched_origins()
    res.to_csv(out / "matched_all_methods.csv", index=False)
    best.to_csv(out / "matched_best_method.csv", index=False)
    per_origin.to_csv(out / "matched_variance_per_origin.csv", index=False)
    blocks, acf = matched.block_check(best, per_origin)
    blocks.to_csv(out / "matched_block_lengths.csv", index=False)
    summary["autocorrelation_of_squared_error_minus_variance"] = acf
    K.write_json(summary, out / "matched_summary.json")
    matched.calibration().to_csv(out / "calibration.csv", index=False)

    step("effect of the state filter (point 1)")
    states.filter_effect(best).to_csv(out / "filter_effect.csv", index=False)

    step("Figure 1 and the Taylor comparison (points 2 and 3a)")
    taylor.figure1_grid().to_csv(out / "figure1_grid.csv", index=False)
    taylor.units().to_csv(out / "taylor_units.csv", index=False)
    smooth, fits = taylor.smooth_curves()
    smooth.to_csv(out / "taylor_smooth_curves.csv", index=False)
    K.write_json(fits, out / "taylor_smooth_fits.json")

    step("purged splits (point 4)")
    purged, summary = purging.purged_splits()
    purged.to_csv(out / "purged_splits.csv", index=False)
    K.write_json(summary, out / "purged_splits_summary.json")
    if args.retrain:
        step("retraining the networks at h = 48")
        purging.retrain(threads=args.threads)
    purging.retrained_scores("outputs" if args.retrain else "archived").to_csv(out / "retrained_h48_scores.csv", index=False)
    needed = [K.RETRAINED_REVISION / f"preds_{site}_h{h}_{v}.csv"
              for site in K.DATASETS for h in purging.H_REVISION for v in purging.VARIANTS]
    if all(p.exists() for p in needed):
        purging.revision_scores("archived").to_csv(out / "retrained_scores.csv", index=False)
    else:
        print(f"   archived/retrained_2018 is incomplete ({sum(p.exists() for p in needed)} of {len(needed)} files); "
              "retrained_scores.csv not written", flush=True)
    step("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
