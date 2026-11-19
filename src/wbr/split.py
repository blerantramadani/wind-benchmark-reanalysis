"""Chronological splits on a complete hourly grid (protocol, Sect. 6).

An origin at grid position i uses the wind of hours [i-48, i) and predicts the
power of hour i+h-1. Under the hour convention of ``preprocess`` (label t
covers [t, t+1h) and is available at t+1h), the forecast is issued when hour
i-1 closes, and a target at position j is known once hour j has closed.

For test year Y, targets are assigned by calendar year:

* training: 2016 ... Y-2, kept only if j <= (first validation target) - h;
* validation: Y-1, kept only if j <= (first test target) - h;
* test: Y.

The two inequalities remove the h-1 samples whose targets are not yet known
when the first forecast of the next block is issued ("purging"). The main
benchmark fit uses hours up to (first test target) - h; the sensitivity fit
stops where the training targets stop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass(frozen=True)
class Grid:
    site: str
    time: pd.DatetimeIndex
    wind: np.ndarray
    power: np.ndarray
    valid: np.ndarray
    stop: np.ndarray


@dataclass(frozen=True)
class Split:
    h: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    bench_cut_main: int
    bench_cut_sens: int
    train_last_target: int


def load_grid(site: str) -> Grid:
    d = pd.read_csv(C.PROCESSED / f"{site}_hourly_{C.YEARS[0]}_{C.YEARS[-1]}.csv",
                    parse_dates=["Timestamp"]).set_index("Timestamp")
    full = pd.date_range(f"{C.YEARS[0]}-01-01 00:00", f"{C.YEARS[-1]}-12-31 23:00", freq="h")
    d = d.reindex(full)
    return Grid(site, full, d["WindSpeed_m_s"].to_numpy(float), d["ActivePower_kW"].to_numpy(float),
                d["valid"].eq(True).to_numpy(), d["stop"].eq(True).to_numpy())


def candidate_origins(g: Grid, h: int, target: str) -> np.ndarray:
    """Origins with 48 valid input hours and a valid target (and no stop for the normal target)."""
    v = g.valid.astype(int)
    csum = np.concatenate(([0], np.cumsum(v)))
    i = np.arange(C.LOOK_BACK, len(v) - h + 1)
    ok = ((csum[i] - csum[i - C.LOOK_BACK]) == C.LOOK_BACK) & g.valid[i + h - 1]
    if target == "normal":
        ok &= ~g.stop[i + h - 1]
    return i[ok]


def split(g: Grid, year: int, h: int, target: str) -> Split:
    o = candidate_origins(g, h, target)
    ty = g.time[o + h - 1].year
    test = o[ty == year]
    val_all, tr_all = o[ty == year - 1], o[ty <= year - 2]
    first_test_target, first_val_target = test[0] + h - 1, val_all[0] + h - 1
    val = val_all[val_all + h - 1 <= first_test_target - h]
    train = tr_all[tr_all + h - 1 <= first_val_target - h]
    return Split(h, train, val, test, first_test_target - h, first_val_target - h, int(train[-1] + h - 1))


def curve_rows(g: Grid, last_pos: int, target: str) -> np.ndarray:
    """Hours for the power curve and scatter: valid up to last_pos, without stops for the normal target."""
    m = g.valid.copy()
    m[last_pos + 1:] = False
    if target == "normal":
        m &= ~g.stop
    return np.flatnonzero(m)


def ar_rows(g: Grid, last_pos: int) -> np.ndarray:
    """Hours for the wind model: all valid hours up to last_pos."""
    m = g.valid.copy()
    m[last_pos + 1:] = False
    return np.flatnonzero(m)


def season(times: pd.DatetimeIndex) -> np.ndarray:
    return np.select([times.month.isin([12, 1, 2]), times.month.isin([3, 4, 5]), times.month.isin([6, 7, 8])],
                     ["DJF", "MAM", "JJA"], "SON")


def history(g: Grid, origins: np.ndarray, k: int = 5) -> np.ndarray:
    """Wind at the last k hours before each origin; column 0 is the most recent hour."""
    return np.column_stack([g.wind[origins - 1 - j] for j in range(k)])
