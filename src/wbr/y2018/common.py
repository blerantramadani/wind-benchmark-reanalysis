"""Shared inputs of the 2018 analyses.

These analyses examine the preprint itself. They read

* the files of the preprint repository (github.com/blerantramadani/
  wind-state-predictability-benchmark, commit 13c0bba) that they need, copied
  unchanged to ``preprint/`` (checksums in ``preprint/SHA256SUMS``);
* the per-origin forecasts behind Table 3 of the preprint, which were not part
  of that commit (``archived/preprint_forecasts``);
* the 2018 SCADA archives (turbine data and status logs), fetched by
  ``wbr download``.

Scripts 07, 10 and 13 of the preprint are loaded as they are. Script 08
imports TensorFlow when it is loaded, so the helpers used here are copied
below without change; ``tests/test_y2018.py`` compares them with
``preprint/scripts/08_final_benchmark.py``.
"""
from __future__ import annotations

import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..preprocess import TIME, _members, _read_table, archives, seconds_overlapped

PREPRINT = C.ROOT / "preprint"
ARCHIVED = C.ROOT / "archived"
PREPRINT_FORECASTS = ARCHIVED / "preprint_forecasts"
RETRAINED = ARCHIVED / "retrained_h48"
RETRAINED_REVISION = ARCHIVED / "retrained_2018"
OUT = C.OUT / "y2018"
TEST, VAL = 1000, 1000
LOOK_BACK = 48  # preprint/scripts/architectures.py
NETWORKS = ("MLP", "LSTM", "GRU", "DLinear", "PatchTST")
SITE_TAG = {"kelmarsh": "Kelmarsh_1_", "penmanshiel": "Penmanshiel_01_"}


# --- copied from preprint/scripts/08_final_benchmark.py (commit 13c0bba) -------------------------

DATASETS = {
    "kelmarsh": {"file": "data/Kelmarsh_T1_2018_hourly.csv", "rated": 2050.0},
    "penmanshiel": {"file": "data/Penmanshiel_T01_2018_hourly.csv", "rated": 2050.0},
}
HORIZONS = (1, 2, 3, 6, 12, 18, 24, 36, 48)


def nrmse(actual, pred, rated):
    return 100.0 * math.sqrt(float(np.mean((actual - pred) ** 2))) / rated


def load_data(path: Path):
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").set_index("Timestamp")
    df["Hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["Month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["WindSpeed_Rolling3h"] = df["WindSpeed_m_s"].rolling(3).mean()
    return df.dropna().copy()


def valid_origins(index: pd.DatetimeIndex, horizon: int):
    """Origins i: inputs [i-L, i) and target i+h-1; all hours contiguous."""
    gaps = np.diff(index.to_numpy()).astype("timedelta64[m]").astype(float) != 60.0
    csum = np.concatenate(([0], np.cumsum(gaps.astype(int))))
    ids = []
    for i in range(LOOK_BACK, len(index) - horizon + 1):
        if csum[i + horizon - 1] - csum[i - LOOK_BACK] == 0:
            ids.append(i)
    return np.asarray(ids, dtype=int)


def empirical_power_curve(wind_train, power_train):
    """Train-only binned curve for the wind-persistence baseline."""
    upper = float(np.percentile(wind_train, 99.5))
    edges = np.linspace(0.0, upper, 41)
    bins = np.digitize(wind_train, edges) - 1
    xs, ys = [], []
    for b in range(40):
        values = power_train[bins == b]
        if len(values) >= 20:
            xs.append((edges[b] + edges[b + 1]) / 2.0)
            ys.append(float(np.mean(values)))
    x, y = np.asarray(xs), np.asarray(ys)
    if len(x) < 3:
        raise ValueError("too few populated wind-speed bins")
    def curve(q):
        q = np.asarray(q, dtype=float)
        return np.where(q <= 0.0, 0.0, np.interp(q, x, y, left=0.0, right=y[-1]))
    return curve


# --- preprint files -------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def preprint_script(file: str):
    """Load a script of the preprint repository as a module (07, 10 or 13)."""
    spec = importlib.util.spec_from_file_location(Path(file).stem, PREPRINT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def b07():
    return preprint_script("07_windonly_bound.py")


def site_data(site: str) -> pd.DataFrame:
    """Hourly file of the preprint after load_data (rows used by script 08)."""
    return load_data(PREPRINT / DATASETS[site]["file"])


def site_raw(site: str) -> pd.DataFrame:
    """Hourly file of the preprint with wind and power present (rows used by script 07)."""
    raw = pd.read_csv(PREPRINT / DATASETS[site]["file"])
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    return raw.sort_values("Timestamp").set_index("Timestamp")[["WindSpeed_m_s", "ActivePower_kW"]].dropna()


def archived_forecasts(site: str, h: int) -> pd.DataFrame:
    return pd.read_csv(PREPRINT_FORECASTS / f"preds_{site}_h{h}_wind_only.csv")


def forecast_column(method: str) -> str:
    return f"bench_{method}" if method in ("WindPersistenceCurve", "LinearRegression") else f"{method}_avg"


def published_scores() -> pd.DataFrame:
    pub = pd.read_csv(PREPRINT / "final_benchmark" / "summary_wind_only.csv")
    return pub[(pub["mode"] == "wind_only") & (pub["n_test"] == TEST)].rename(columns={"dataset": "site"})


def published_bound() -> pd.DataFrame:
    return pd.read_csv(PREPRINT / "results" / "07_windonly_bound.csv").set_index(["dataset", "horizon_h"])


# --- 2018 SCADA archive ---------------------------------------------------------------------------

def _archive_2018(site: str):
    (archive,) = archives(site, 2018)
    turbine, status = _members(archive, site)
    return archive, turbine, status


def status_events(site: str) -> pd.DataFrame:
    """All status-log events of 2018, duplicates removed; events without an end have end = NaT."""
    archive, _, status = _archive_2018(site)
    ev = _read_table(archive, status, "Timestamp start").drop_duplicates()
    ev["start"] = pd.to_datetime(ev["Timestamp start"])
    ev["end"] = pd.to_datetime(ev["Timestamp end"], errors="coerce")
    return ev


def ten_minute(site: str, columns=("Wind speed (m/s)", "Power (kW)")) -> pd.DataFrame:
    archive, turbine, _ = _archive_2018(site)
    raw = _read_table(archive, turbine, TIME, usecols=[TIME, *columns])
    raw["t"] = pd.to_datetime(raw[TIME])
    return raw.drop(columns=[TIME]).set_index("t").sort_index()


def hours_touched(hours: pd.DatetimeIndex, events: pd.DataFrame) -> np.ndarray:
    """Seconds of each hour [t, t+1h) covered by the union of the events."""
    return seconds_overlapped(hours, events)


def stop_flags(site: str, hours: pd.DatetimeIndex) -> np.ndarray:
    ev = status_events(site)
    dur = ev[ev["end"].notna()]
    return hours_touched(hours, dur[dur["Status"] == "Stop"]) > 0


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
