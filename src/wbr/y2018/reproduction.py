"""Reproduction of the preprint's scores and its time conventions (reply, introduction and point 4).

* ``scores``: the 126 published scores (2 sites x 9 horizons x 7 methods) are
  recomputed from the archived per-origin forecasts and compared with
  preprint/final_benchmark/summary_wind_only.csv. The networks are not
  retrained here.
* ``splits``: the chronological splits of script 08 for every site and
  horizon, and how many training and validation targets fall after the
  first forecast of the next block is issued.
* ``hour_convention``: the hourly files are rebuilt from the ten-minute
  records with two labelling conventions (label t covers t ... t+50 min, or
  t-50 ... t min) and compared with the preprint's hourly files.

Convention: origin i uses the wind of hours [i-48, i) and predicts the power
of hour i+h-1. The label of an hour marks the start of the hour, so the
forecast is issued when hour i-1 closes, and target j is known then only if
j <= i-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as K


def scores() -> tuple[pd.DataFrame, dict]:
    rows = []
    for site, meta in K.DATASETS.items():
        df = K.site_data(site)
        for h in K.HORIZONS:
            pred = K.archived_forecasts(site, h)
            actual = pred["actual"].to_numpy(float)
            assert np.allclose(actual, df["ActivePower_kW"].to_numpy(float)[pred["target_idx"].to_numpy(int)])
            for name in ("WindPersistenceCurve", "LinearRegression"):
                rows.append({"site": site, "horizon_h": h, "model": name,
                             "nrmse_pct": K.nrmse(actual, pred[f"bench_{name}"].to_numpy(float), meta["rated"])})
            for name in K.NETWORKS:
                seeds = [K.nrmse(actual, pred[c].to_numpy(float), meta["rated"])
                         for c in pred.columns if c.startswith(f"{name}_seed")]
                rows.append({"site": site, "horizon_h": h, "model": name,
                             "nrmse_pct": K.nrmse(actual, pred[f"{name}_avg"].to_numpy(float), meta["rated"]),
                             "seed_mean_pct": float(np.mean(seeds)), "seed_std_pct": float(np.std(seeds)),
                             "n_seeds": len(seeds)})
    got = pd.DataFrame(rows)
    pub = K.published_scores()[["site", "horizon_h", "model", "nRMSE_mean", "nRMSE_seed_mean", "nRMSE_seed_std"]]
    m = got.merge(pub, on=["site", "horizon_h", "model"], how="outer", validate="one_to_one", indicator=True)
    m["diff_pp"] = m["nrmse_pct"] - m["nRMSE_mean"]
    summary = {"rows": int(len(m)), "unmatched_rows": int((m["_merge"] != "both").sum()),
               "max_abs_diff_pp": float(m["diff_pp"].abs().max()),
               "max_abs_seed_mean_diff_pp": float((m["seed_mean_pct"] - m["nRMSE_seed_mean"]).abs().max()),
               "max_abs_seed_std_diff_pp": float((m["seed_std_pct"] - m["nRMSE_seed_std"]).abs().max())}
    return m.drop(columns="_merge"), summary


def split_ids(index: pd.DatetimeIndex, h: int):
    ids = K.valid_origins(index, h)
    n_train = len(ids) - K.TEST - K.VAL
    return ids, ids[:n_train], ids[n_train:n_train + K.VAL], ids[n_train + K.VAL:]


def splits() -> pd.DataFrame:
    rows = []
    for site in K.DATASETS:
        df = K.site_data(site)
        idx = df.index
        bench_last = K.site_raw(site).index[-K.TEST - 1]
        for h in K.HORIZONS:
            ids, tr, va, te = split_ids(idx, h)
            pred = K.archived_forecasts(site, h)
            rows.append({
                "site": site, "horizon_h": h, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                "test_targets_match_archive": bool(np.array_equal(te + h - 1, pred["target_idx"].to_numpy(int))),
                "train_last_target": str(idx[tr[-1] + h - 1]), "val_first_issue_after": str(idx[va[0] - 1]),
                "val_last_target": str(idx[va[-1] + h - 1]), "test_first_issue_after": str(idx[te[0] - 1]),
                "test_first_target": str(idx[te[0] + h - 1]), "test_last_target": str(idx[te[-1] + h - 1]),
                "train_targets_after_first_val_issue": int(np.sum(tr + h - 1 > va[0] - 1)),
                "val_targets_after_first_test_issue": int(np.sum(va + h - 1 > te[0] - 1)),
                "benchmark_fit_last_row": str(bench_last),
                "benchmark_fit_hours_after_first_test_issue": int((bench_last - idx[te[0] - 1]) / pd.Timedelta(hours=1)),
            })
    return pd.DataFrame(rows)


def test_window_wind() -> dict:
    """Mean wind speed before the test window and in it (the AR forecast reverts to the pre-test mean)."""
    out = {}
    for site in K.DATASETS:
        raw = K.site_raw(site)
        out[site] = {"pretest_mean_ms": float(raw["WindSpeed_m_s"].iloc[:-K.TEST].mean()),
                     "test_mean_ms": float(raw["WindSpeed_m_s"].iloc[-K.TEST:].mean()),
                     "test_first_hour": str(raw.index[-K.TEST]), "test_last_hour": str(raw.index[-1])}
    return out


def hour_convention() -> dict:
    out = {}
    for site in K.DATASETS:
        pub = pd.read_csv(K.PREPRINT / K.DATASETS[site]["file"], parse_dates=["Timestamp"]).set_index("Timestamp")
        raw = K.ten_minute(site).rename(columns={"Wind speed (m/s)": "wind", "Power (kW)": "power"})
        res = {"raw_rows": int(len(raw))}
        for name, label in (("start_of_hour", "left"), ("end_of_hour", "right")):
            hr = raw.resample("h", label=label, closed=label).mean()
            j = pub.join(hr, how="inner")
            wind = (j["WindSpeed_m_s"] - j["wind"]).abs()
            res[name] = {"hours_compared": int(len(j)), "wind_max_abs_diff": float(wind.max()),
                         "wind_share_within_1e-6": float((wind < 1e-6).mean())}
        out[site] = res
    return out
