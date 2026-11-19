#!/usr/bin/env python3
"""Final chronological wind-only evaluation for Paper 1.

The evaluation uses the 48-hour history of wind speed only, matching the
information set of the reported predictability benchmark. All scalers are
fitted on chronological training rows only. The final 1,000 valid forecast
origins are test data; the preceding 1,000 are validation data.

Examples (run from the repository root):
  python3 scripts/08_final_benchmark.py --mode wind_only
  python3 scripts/08_final_benchmark.py --smoke --models MLP
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

# Fixed architecture definitions; preprocessing and splits are implemented here.
from architectures import ARCHITECTURES, BATCH_SIZE, EPOCHS, LOOK_BACK, TARGET_COL


DATASETS = {
    "kelmarsh": {"file": "data/Kelmarsh_T1_2018_hourly.csv", "rated": 2050.0},
    "penmanshiel": {"file": "data/Penmanshiel_T01_2018_hourly.csv", "rated": 2050.0},
}
HORIZONS = (1, 2, 3, 6, 12, 18, 24, 36, 48)
WIND_ONLY_FEATURES = ("WindSpeed_m_s",)


def nrmse(actual, pred, rated):
    return 100.0 * math.sqrt(float(np.mean((actual - pred) ** 2))) / rated


def dm_test(actual, p1, p2, horizon):
    d = (actual - p1) ** 2 - (actual - p2) ** 2
    n, dbar = len(d), float(np.mean(d))
    gamma0 = float(np.var(d, ddof=0))
    gammas = [float(np.cov(d[k:], d[:-k], ddof=0)[0, 1]) for k in range(1, horizon)]
    var_mean = (gamma0 + 2.0 * sum(gammas)) / n
    if var_mean <= 0.0:
        return 0.0, 1.0
    stat = dbar / math.sqrt(var_mean)
    stat *= math.sqrt(max((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n, 1e-12))
    return float(stat), float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1)))


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


def make_sequences(x, y, ids, horizon):
    xx = np.stack([x[i - LOOK_BACK:i] for i in ids])
    yy = y[ids + horizon - 1]
    return xx, yy, ids + horizon - 1


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


def run_unit(df, dataset, horizon, mode, seeds, epochs, test_count, val_count, models, out_dir):
    rated = DATASETS[dataset]["rated"]
    features = WIND_ONLY_FEATURES
    x_raw = df.loc[:, features].to_numpy(dtype=float)
    y_raw = df[TARGET_COL].to_numpy(dtype=float).reshape(-1, 1)
    ids = valid_origins(df.index, horizon)
    if len(ids) <= test_count + val_count + 50:
        raise ValueError(f"{dataset}, h={horizon}: insufficient valid sequences")

    # Split geometry before scaling.  The scaler may see only rows available by
    # the final training target, never validation/test rows.
    n_train = len(ids) - test_count - val_count
    train_last_target = ids[n_train - 1] + horizon - 1
    fs, ys = MinMaxScaler(), MinMaxScaler()
    fs.fit(x_raw[:train_last_target + 1])
    ys.fit(y_raw[:train_last_target + 1])
    x = fs.transform(x_raw)
    y = ys.transform(y_raw).ravel()
    x_seq, y_seq, target_idx = make_sequences(x, y, ids, horizon)
    tr = slice(0, n_train)
    va = slice(n_train, n_train + val_count)
    te = slice(n_train + val_count, None)
    xtr, xva, xte = x_seq[tr], x_seq[va], x_seq[te]
    ytr, yva, yte = y_seq[tr], y_seq[va], y_seq[te]
    actual = ys.inverse_transform(yte.reshape(-1, 1)).ravel()
    target_test = target_idx[te]
    assert len(actual) == test_count
    assert np.all(ids[te] - 1 < target_test)

    train_power = y_raw[:train_last_target + 1, 0]
    train_wind = df["WindSpeed_m_s"].to_numpy(dtype=float)[:train_last_target + 1]
    curve = empirical_power_curve(train_wind, train_power)
    origin_wind = df["WindSpeed_m_s"].to_numpy(dtype=float)[ids[te] - 1]
    xtr_final, xva_final, xte_final = xtr[:, -1, :], xva[:, -1, :], xte[:, -1, :]

    bench = {"WindPersistenceCurve": curve(origin_wind)}
    linear = LinearRegression().fit(xtr_final, ytr)
    bench["LinearRegression"] = ys.inverse_transform(linear.predict(xte_final).reshape(-1, 1)).ravel()
    scores = {name: nrmse(actual, pred, rated) for name, pred in bench.items()}
    strongest = min(scores, key=scores.get)
    rows = [
        {"dataset": dataset, "horizon_h": horizon, "mode": mode, "model": name,
         "nRMSE_mean": score, "nRMSE_std": 0.0, "is_benchmark": True,
         "n_train": n_train, "n_val": val_count, "n_test": test_count}
        for name, score in scores.items()
    ]
    dm_rows, all_predictions = [], {}
    for name in models:
        build = ARCHITECTURES[name]
        started, seed_predictions = time.time(), []
        for seed in seeds:
            np.random.seed(seed); tf.random.set_seed(seed)
            early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            if name == "MLP":
                model = build(xtr_final.shape[1], seed)
                model.fit(xtr_final, ytr, validation_data=(xva_final, yva), shuffle=False,
                          epochs=epochs, batch_size=BATCH_SIZE, callbacks=[early], verbose=0)
                pred = model.predict(xte_final, verbose=0)
            else:
                model = build((xtr.shape[1], xtr.shape[2]), seed)
                model.fit(xtr, ytr, validation_data=(xva, yva), shuffle=False,
                          epochs=epochs, batch_size=BATCH_SIZE, callbacks=[early], verbose=0)
                pred = model.predict(xte, verbose=0)
            seed_predictions.append(ys.inverse_transform(pred.reshape(-1, 1)).ravel())
            tf.keras.backend.clear_session()
        # Table 3 reports the error of the seed-averaged forecast, not the
        # arithmetic mean of the errors from separate seeds.  These quantities
        # differ because averaging predictions can reduce seed-specific noise.
        average = np.mean(seed_predictions, axis=0)
        ensemble_score = nrmse(actual, average, rated)
        seed_scores = [nrmse(actual, pred, rated) for pred in seed_predictions]
        stat, pvalue = dm_test(actual, average, bench[strongest], horizon)
        rows.append({"dataset": dataset, "horizon_h": horizon, "mode": mode, "model": name,
                     "nRMSE_mean": ensemble_score,
                     "nRMSE_seed_mean": float(np.mean(seed_scores)),
                     "nRMSE_seed_std": float(np.std(seed_scores)),
                     "nRMSE_std": float(np.std(seed_scores)),
                     "is_benchmark": False, "n_train": n_train, "n_val": val_count, "n_test": test_count})
        dm_rows.append({"dataset": dataset, "horizon_h": horizon, "mode": mode, "model": name,
                        "strongest_benchmark": strongest, "benchmark_nRMSE": scores[strongest],
                        "DM": stat, "p_value": pvalue})
        all_predictions[f"{name}_avg"] = average
        for seed, pred in zip(seeds, seed_predictions):
            all_predictions[f"{name}_seed{seed}"] = pred
        print(f"  {name:9s} ensemble={ensemble_score:5.2f} | "
              f"seed mean={np.mean(seed_scores):5.2f} ± {np.std(seed_scores):4.2f} | "
              f"vs {strongest} ({scores[strongest]:.2f}) | {time.time() - started:.0f}s")

    pred = {"target_idx": target_test, "actual": actual}
    pred.update({f"bench_{name}": value for name, value in bench.items()})
    pred.update(all_predictions)
    path = out_dir / f"preds_{dataset}_h{horizon}_{mode}.csv"
    pd.DataFrame(pred).to_csv(path, index=False)
    print(f"  saved {path.name} | train/val/test={n_train}/{val_count}/{test_count}")
    return rows, dm_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("wind_only",), required=True)
    ap.add_argument("--datasets", default="kelmarsh,penmanshiel")
    ap.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--models", default=",".join(ARCHITECTURES))
    ap.add_argument("--test-count", type=int, default=1000)
    ap.add_argument("--val-count", type=int, default=1000)
    ap.add_argument("--out-dir", default="final_benchmark",
                    help="directory for summary tables and predictions")
    ap.add_argument("--resume", action="store_true", help="skip dataset/horizon units already saved in summary")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",")]
    horizons = [int(x) for x in args.horizons.split(",")]
    models = [x.strip() for x in args.models.split(",")]
    missing = set(models) - set(ARCHITECTURES)
    if missing:
        raise ValueError(f"unknown models: {sorted(missing)}")
    seeds, epochs = list(range(args.seeds)), EPOCHS
    test_count, val_count = args.test_count, args.val_count
    if args.smoke:
        datasets, horizons, models, seeds, epochs = datasets[:1], [1], models[:1], [0], 2
        test_count, val_count = 50, 50

    summary_path = out_dir / f"summary_{args.mode}.csv"
    dm_path = out_dir / f"dm_{args.mode}.csv"
    summary = pd.read_csv(summary_path).to_dict("records") if args.resume and summary_path.exists() else []
    dm = pd.read_csv(dm_path).to_dict("records") if args.resume and dm_path.exists() else []

    def already_complete(dataset, horizon):
        prior = {
            row["model"] for row in summary
            if row["dataset"] == dataset and int(row["horizon_h"]) == horizon and row["mode"] == args.mode
        }
        return set(models).issubset(prior)

    print(f"mode={args.mode} | datasets={datasets} | horizons={horizons} | seeds={len(seeds)}")
    for dataset in datasets:
        df = load_data(root / DATASETS[dataset]["file"])
        for horizon in horizons:
            if args.resume and already_complete(dataset, horizon):
                print(f"[{dataset} | h={horizon} | {args.mode}] already complete; skipped")
                continue
            print(f"[{dataset} | h={horizon} | {args.mode}]")
            rows, dm_rows = run_unit(
                df, dataset, horizon, args.mode, seeds, epochs, test_count, val_count, models, out_dir
            )
            summary.extend(rows); dm.extend(dm_rows)
            pd.DataFrame(summary).to_csv(summary_path, index=False)
            pd.DataFrame(dm).to_csv(dm_path, index=False)


if __name__ == "__main__":
    main()
