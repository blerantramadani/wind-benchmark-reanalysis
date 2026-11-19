"""Command-line entry point: ``wbr <step> [options]``.

Steps, in order:

  download     fetch the 19 SCADA archives from Zenodo and check md5 sums
  preprocess   hourly series 2016-2023 and stop flags
  benchmark    benchmark variances and model forecasts at every origin
  forecasts    persistence, linear regression and MLP (several hours; can be split)
  networks     LSTM, GRU, DLinear and PatchTST for 2019 (many hours; one site per call)
  results      point estimates (--points), intervals (per site/target), --merge;
               --no-networks leaves the 2019 sequence networks out (results of the reply)
  tables       Markdown tables and headline counts; --blocks 24,48,96 for the
               block-length sensitivity
  check        rebuild the point estimates from the stored forecasts and compare
               them with expected/point_estimates.csv
  y2018        analyses of the preprint's 2018 configuration (points 1, 2, 3a,
               3b(ii) and 4 of the reply)
  all          every step except y2018, in one process; ``all --no-networks``
               skips the sequence networks and gives the results of the reply
"""
from __future__ import annotations

import importlib
import sys

STEPS = ("download", "preprocess", "benchmark", "forecasts", "networks", "results", "tables", "check", "y2018")
MODULES = {"y2018": "wbr.y2018.run"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    step, rest = argv[0], argv[1:]
    if step == "all":
        from . import config as C
        nets = "--no-networks" not in rest
        flag = [] if nets else ["--no-networks"]
        steps = [("download", []), ("preprocess", []), ("benchmark", []), ("forecasts", [])]
        steps += [("networks", ["--site", site]) for site in C.SITES] if nets else []
        steps += [("results", ["--points", *flag]), ("results", flag), ("results", ["--merge"]), ("tables", [])]
        for s, extra in steps:
            print(f"== {s} {' '.join(extra)}", flush=True)
            if main([s, *extra]):
                return 1
        return 0
    if step not in STEPS:
        print(f"unknown step: {step}\n{__doc__}")
        return 2
    fn = importlib.import_module(MODULES.get(step, f"wbr.{step}")).main
    code = fn(rest) if step in ("forecasts", "networks", "results", "tables", "check", "y2018") else fn()
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
