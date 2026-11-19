"""Paths and fixed protocol settings.

All paths are relative to the repository root. The raw-data folder can be
moved with the environment variable WBR_RAW_DIR, for example to reuse
archives that were downloaded earlier.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("WBR_RAW_DIR", ROOT / "data" / "raw"))
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FORECASTS = OUT / "forecasts"
BENCHMARK = OUT / "benchmark"
NETWORKS = OUT / "networks_2019"
RESULTS = OUT / "results"

P = json.loads((ROOT / "config" / "protocol.json").read_text())
SITES = tuple(P["sites"])
YEARS = tuple(P["years"])
TEST_YEARS = tuple(P["test_years"])
HORIZONS = tuple(P["horizons_h"])
TARGETS = tuple(P["targets"])
RATED = float(P["rated_kw"])
LOOK_BACK = int(P["look_back_h"])
SCALE = 100.0 / RATED


def ensure_dirs():
    for d in (RAW, PROCESSED, OUT, FORECASTS, BENCHMARK, NETWORKS, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
