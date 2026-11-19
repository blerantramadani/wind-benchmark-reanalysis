"""Chronological boundaries for h > 1 (reply, point 4).

``purged_splits``: for every site and horizon of the preprint

1. checks whether any test target enters training, validation or the
   benchmark fit;
2. removes from training and validation the origins whose target is not yet
   known when the first forecast of the next block is issued;
3. refits persistence through the curve and linear regression as script 08
   does, on the purged splits, and scores them on the same 1,000 targets;
4. refits the Table 2 benchmark as script 07 does, on the hours known when
   the first test forecast is issued.

``retrain`` (TensorFlow, several hours) trains the five networks at h = 48
with ten seeds on the original and the purged splits, with the settings of
script 08. Its forecasts for the published reply are archived in
``archived/retrained_h48``; ``retrained_scores`` scores them (or a new run in
outputs/y2018/retrained_h48 with ``source="outputs"``). Network training on a
CPU does not repeat exactly, so a new run differs slightly from the archive.

For the revision, the reply promised to retrain all networks under the purging
rule. ``main`` (``python -m wbr.y2018.purging``) trains them in the same way at
the other horizons with h > 1 (2, 3, 6, 12, 18, 24 and 36 h; at h = 1 purging
removes nothing), again with the original and the purged splits, so that the
effect of purging is not mixed with the run-to-run variation of training. Each
(site, horizon, split) is written to outputs/y2018/retrained_2018 with a
record of its run and skipped on restart. The forecasts are archived in
``archived/retrained_2018``; ``revision_scores`` scores all eight horizons.

Usage: python -m wbr.y2018.purging [--sites kelmarsh] [--horizons 2,3] [--threads 2]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import common as K
from .reproduction import split_ids

H_RETRAIN = 48                               # retrained for the reply (archived/retrained_h48)
H_REVISION = (2, 3, 6, 12, 18, 24, 36)       # the other horizons with h > 1 (archived/retrained_2018)
VARIANTS = ("original", "purged")


def purge(tr, va, te, h):
    return tr[tr + h - 1 <= va[0] - 1], va[va + h - 1 <= te[0] - 1]


def _baselines(df, rated, tr, te, h):
    """Persistence through the curve and linear regression, as in run_unit of script 08."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler
    x_raw, y_raw = df[["WindSpeed_m_s"]].to_numpy(float), df[["ActivePower_kW"]].to_numpy(float)
    last = tr[-1] + h - 1
    fs, ys = MinMaxScaler().fit(x_raw[:last + 1]), MinMaxScaler().fit(y_raw[:last + 1])
    x, y = fs.transform(x_raw), ys.transform(y_raw).ravel()
    actual = y_raw[te + h - 1, 0]
    curve = K.empirical_power_curve(x_raw[:last + 1, 0], y_raw[:last + 1, 0])
    lr = LinearRegression().fit(x[tr - 1], y[tr + h - 1])
    lr_pred = ys.inverse_transform(lr.predict(x[te - 1]).reshape(-1, 1)).ravel()
    return K.nrmse(actual, curve(x_raw[te - 1, 0]), rated), K.nrmse(actual, lr_pred, rated), df.index[last]


def purged_splits():
    B07 = K.b07()
    pub = K.published_scores()
    bound = K.published_bound()
    rows = []
    for site, meta in K.DATASETS.items():
        df, rated, raw = K.site_data(site), meta["rated"], K.site_raw(site)
        for h in K.HORIZONS:
            ids, tr, va, te = split_ids(df.index, h)
            test_targets = set(te + h - 1)
            first_issue = df.index[te[0] - 1]
            tr_p, va_p = purge(tr, va, te, h)
            wpc, lr, fit_last = _baselines(df, rated, tr_p, te, h)
            wpc0, lr0, _ = _baselines(df, rated, tr, te, h)
            fit_rows = raw[raw.index <= first_issue]
            est, _ = B07.estimate(fit_rows, B07.RATED_KW[site], [h], 5, 64, 40, 20)
            est = est.set_index("horizon_h")
            p = pub[(pub["site"] == site) & (pub["horizon_h"] == h)].set_index("model")["nRMSE_mean"]
            rows.append({
                "site": site, "horizon_h": h,
                "test_targets_in_train": len(test_targets & set(tr + h - 1)),
                "test_targets_in_val": len(test_targets & set(va + h - 1)),
                "train_removed": len(tr) - len(tr_p), "val_removed": len(va) - len(va_p),
                "purged_fit_last_row": str(fit_last),
                "wpc_published": p["WindPersistenceCurve"], "wpc_recomputed_original": wpc0, "wpc_purged": wpc,
                "lr_published": p["LinearRegression"], "lr_recomputed_original": lr0, "lr_purged": lr,
                "wpc_change_pp": wpc - p["WindPersistenceCurve"], "lr_change_pp": lr - p["LinearRegression"],
                "bench_fit_rows_published": int(bound.loc[(site, h), "n_train_hours"]),
                "bench_fit_rows_purged": len(fit_rows), "bench_ar_order_purged": int(est.loc[h, "ar_order"]),
                "bench_published_pct": bound.loc[(site, h), "bound_pct"],
                "bench_purged_pct": float(est.loc[h, "bound_pct"]),
                "bench_change_pp": float(est.loc[h, "bound_pct"]) - bound.loc[(site, h), "bound_pct"],
            })
    out = pd.DataFrame(rows)
    summary = {
        "test_targets_in_train_total": int(out["test_targets_in_train"].sum()),
        "test_targets_in_val_total": int(out["test_targets_in_val"].sum()),
        "removed_equals_h_minus_1": bool(((out["train_removed"] == out["horizon_h"] - 1)
                                          & (out["val_removed"] == out["horizon_h"] - 1)).all()),
        "baseline_recompute_max_diff_pp": float(max((out["wpc_recomputed_original"] - out["wpc_published"]).abs().max(),
                                                    (out["lr_recomputed_original"] - out["lr_published"]).abs().max())),
        "wpc_change_pp_range": [float(out["wpc_change_pp"].min()), float(out["wpc_change_pp"].max())],
        "lr_change_pp_range": [float(out["lr_change_pp"].min()), float(out["lr_change_pp"].max())],
        "bench_change_pp_range": [float(out["bench_change_pp"].min()), float(out["bench_change_pp"].max())],
    }
    return out, summary


def _train_unit(site, h, variant, tf):
    """Train the five networks for one site, horizon and split (ten seeds each), as script 08 does."""
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.callbacks import EarlyStopping
    from ..architectures import ARCHITECTURES, BATCH_SIZE, EPOCHS
    df, rated = K.site_data(site), K.DATASETS[site]["rated"]
    _, tr, va, te = split_ids(df.index, h)
    if variant == "purged":
        tr, va = purge(tr, va, te, h)
    x_raw, y_raw = df[["WindSpeed_m_s"]].to_numpy(float), df[["ActivePower_kW"]].to_numpy(float)
    last = tr[-1] + h - 1
    fs, ys = MinMaxScaler().fit(x_raw[:last + 1]), MinMaxScaler().fit(y_raw[:last + 1])
    x, y = fs.transform(x_raw), ys.transform(y_raw).ravel()
    seq = lambda o: np.stack([x[i - K.LOOK_BACK:i] for i in o])
    xtr, xva, xte, ytr, yva = seq(tr), seq(va), seq(te), y[tr + h - 1], y[va + h - 1]
    actual = y_raw[te + h - 1, 0]
    preds, log = {"target_idx": te + h - 1, "actual": actual}, []
    for name, build in ARCHITECTURES.items():
        started, seed_preds, epochs = time.time(), [], []
        for seed in range(10):
            np.random.seed(seed)
            tf.random.set_seed(seed)
            stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            if name == "MLP":
                model = build(1, seed)
                hist = model.fit(xtr[:, -1, :], ytr, validation_data=(xva[:, -1, :], yva), shuffle=False,
                                 epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[stop], verbose=0)
                p = model.predict(xte[:, -1, :], verbose=0)
            else:
                model = build((xtr.shape[1], xtr.shape[2]), seed)
                hist = model.fit(xtr, ytr, validation_data=(xva, yva), shuffle=False,
                                 epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[stop], verbose=0)
                p = model.predict(xte, verbose=0)
            p = ys.inverse_transform(p.reshape(-1, 1)).ravel()
            preds[f"{name}_seed{seed}"] = p
            seed_preds.append(p)
            epochs.append(len(hist.history["loss"]))
            tf.keras.backend.clear_session()
        preds[f"{name}_avg"] = np.mean(seed_preds, axis=0)
        log.append({"site": site, "horizon_h": h, "variant": variant, "model": name, "n_train": len(tr), "n_val": len(va),
                    "nrmse_ensemble_pct": K.nrmse(actual, preds[f"{name}_avg"], rated),
                    "epochs": ",".join(map(str, epochs)), "seconds": round(time.time() - started, 1)})
        print(log[-1], flush=True)
    return pd.DataFrame(preds), log


def _tensorflow(threads):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def retrain(sites=("kelmarsh", "penmanshiel"), variants=VARIANTS, threads: int = 2):
    """Train the five networks of the preprint at h = 48 (ten seeds) and write their test forecasts."""
    tf = _tensorflow(threads)
    out_dir = K.OUT / "retrained_h48"
    out_dir.mkdir(parents=True, exist_ok=True)
    h, log = H_RETRAIN, []
    for site in sites:
        for variant in variants:
            preds, rows = _train_unit(site, h, variant, tf)
            log += [{k: v for k, v in r.items() if k != "horizon_h"} for r in rows]
            preds.to_csv(out_dir / f"preds_{site}_h{h}_{variant}.csv", index=False)
            pd.DataFrame(log).to_csv(out_dir / "run_log_h48.csv", index=False)


def main(argv=None):
    """Retrain the five networks at the other horizons with h > 1, for the revision (restartable)."""
    ap = argparse.ArgumentParser(description="Retrain the 2018 networks at h = 2-36 with original and purged splits")
    ap.add_argument("--sites", default=",".join(K.DATASETS))
    ap.add_argument("--horizons", default=",".join(map(str, H_REVISION)))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    tf = _tensorflow(args.threads)
    out_dir = K.OUT / "retrained_2018"
    out_dir.mkdir(parents=True, exist_ok=True)
    for site in args.sites.split(","):
        for h in map(int, args.horizons.split(",")):
            for variant in VARIANTS:
                stem = out_dir / f"preds_{site}_h{h}_{variant}"
                if stem.with_suffix(".csv").exists() and stem.with_suffix(".json").exists():
                    continue
                preds, rows = _train_unit(site, h, variant, tf)
                tmp = stem.with_name(stem.name + f".{os.getpid()}.tmp")
                preds.to_csv(tmp, index=False)
                tmp.replace(stem.with_suffix(".csv"))
                record = {"rows": rows, "threads": args.threads, "tensorflow": tf.__version__}
                json_tmp = stem.with_name(stem.name + f".json.{os.getpid()}.tmp")
                json_tmp.write_text(json.dumps(record))
                json_tmp.replace(stem.with_suffix(".json"))
                logs = [r for p in sorted(out_dir.glob("preds_*.json")) for r in json.loads(p.read_text())["rows"]]
                log_tmp = out_dir / f"run_log.csv.{os.getpid()}.tmp"
                pd.DataFrame(logs).sort_values(["site", "horizon_h", "variant", "model"]).to_csv(log_tmp, index=False)
                log_tmp.replace(out_dir / "run_log.csv")
    return 0


def retrained_scores(source: str = "archived") -> pd.DataFrame:
    folder = K.RETRAINED if source == "archived" else K.OUT / "retrained_h48"
    pub = K.published_scores()
    pub = pub[pub["horizon_h"] == H_RETRAIN].set_index(["site", "model"])["nRMSE_mean"]
    rows = []
    for site, meta in K.DATASETS.items():
        score = {}
        for variant in ("original", "purged"):
            pred = pd.read_csv(folder / f"preds_{site}_h{H_RETRAIN}_{variant}.csv")
            for name in K.NETWORKS:
                score[(name, variant)] = K.nrmse(pred["actual"].to_numpy(float), pred[f"{name}_avg"].to_numpy(float), meta["rated"])
        for name in K.NETWORKS:
            rows.append({"site": site, "model": name, "published_pct": pub[(site, name)],
                         "original_pct": score[(name, "original")], "purged_pct": score[(name, "purged")],
                         "rerun_minus_published_pp": score[(name, "original")] - pub[(site, name)],
                         "purging_change_pp": score[(name, "purged")] - score[(name, "original")]})
    return pd.DataFrame(rows)


def revision_scores(source: str = "archived") -> pd.DataFrame:
    """Scores of the networks retrained at every horizon with h > 1 (2-36 h for the revision, 48 h from the reply)."""
    folders = {h: (K.RETRAINED_REVISION if source == "archived" else K.OUT / "retrained_2018") for h in H_REVISION}
    folders[H_RETRAIN] = K.RETRAINED if source == "archived" else K.OUT / "retrained_h48"
    pub = K.published_scores().set_index(["site", "horizon_h", "model"])["nRMSE_mean"]
    rows = []
    for site, meta in K.DATASETS.items():
        for h in sorted(folders):
            score = {}
            for variant in VARIANTS:
                pred = pd.read_csv(folders[h] / f"preds_{site}_h{h}_{variant}.csv")
                for name in K.NETWORKS:
                    score[(name, variant)] = K.nrmse(pred["actual"].to_numpy(float), pred[f"{name}_avg"].to_numpy(float), meta["rated"])
            for name in K.NETWORKS:
                rows.append({"site": site, "horizon_h": h, "model": name, "published_pct": pub[(site, h, name)],
                             "original_pct": score[(name, "original")], "purged_pct": score[(name, "purged")],
                             "rerun_minus_published_pp": score[(name, "original")] - pub[(site, h, name)],
                             "purging_change_pp": score[(name, "purged")] - score[(name, "original")]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
