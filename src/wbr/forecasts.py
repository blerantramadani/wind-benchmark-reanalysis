"""Wind-only forecasts for every test year, target and horizon (protocol, Sect. 8).

* WindPersistenceCurve: the 40-bin empirical curve fitted on the training
  hours (without stops for the normal target), applied to the last observed
  hourly wind speed;
* LinearRegression: least-squares line of power on the last hourly wind speed;
* MLP: the preprint architecture, ten seeds (0-9), Adam, MSE loss, batch 32, up
  to 50 epochs, early stopping on the validation loss (patience 5, best weights
  restored), no shuffling; the forecast is the mean of the ten seeds.

Inputs and targets are min-max scaled with the valid hours up to the last
training target. Forecasts are stored for the validation and test origins.
The curve and the scoring function follow scripts/08_final_benchmark.py of the
preprint repository (commit 13c0bba).

Usage: python -m wbr.forecasts [--sites kelmarsh] [--years 2019,2020] [--threads 1] [--no-mlp]
Each finished cell gets a small record file (<cell>_done.json) holding the
hash of config/protocol.json and whether the MLP was run. On restart a cell is
skipped only when its validation and test files and this record exist and the
hash matches; with --no-mlp a cell that already has MLP forecasts is left as it
is. Files are written under a temporary name and renamed, so an interrupted
write never looks finished.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from .split import ar_rows, curve_rows, load_grid, split


def nrmse(actual, pred, rated=C.RATED):
    return 100.0 * float(np.sqrt(np.mean((actual - pred) ** 2))) / rated


def empirical_power_curve(wind_train, power_train):
    """Train-only binned curve for the wind-persistence baseline (as in 08_final_benchmark.py)."""
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


def scalers(g, s):
    rows = ar_rows(g, s.train_last_target)
    w0, w1 = g.wind[rows].min(), g.wind[rows].max()
    p0, p1 = g.power[rows].min(), g.power[rows].max()
    return (lambda w: (w - w0) / (w1 - w0)), (lambda p: (p - p0) / (p1 - p0)), (lambda y: y * (p1 - p0) + p0)


def mlp_forecasts(x_tr, y_tr, x_va, y_va, x_eval):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from .architectures import ARCHITECTURES, BATCH_SIZE, EPOCHS
    outs, epochs = [[] for _ in x_eval], []
    for seed in range(C.P["seeds"]):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        model = ARCHITECTURES["MLP"](1, seed)
        hist = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), shuffle=False, epochs=EPOCHS,
                         batch_size=BATCH_SIZE, verbose=0,
                         callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
        epochs.append(len(hist.history["loss"]))
        for k, x in enumerate(x_eval):
            outs[k].append(model.predict(x, verbose=0).ravel())
        tf.keras.backend.clear_session()
    return [np.mean(o, axis=0) for o in outs], epochs


def run_cell(g, year, target, h, with_mlp):
    s = split(g, year, h, target)
    cv = curve_rows(g, s.train_last_target, target)
    curve = empirical_power_curve(g.wind[cv], g.power[cv])
    sx, sy, sy_inv = scalers(g, s)
    parts = {"val": s.val, "test": s.test}
    res = {}
    for name, o in parts.items():
        res[name] = pd.DataFrame({"origin": o, "target_pos": o + h - 1, "target_time": g.time[o + h - 1],
                                  "actual": g.power[o + h - 1], "wind_last": g.wind[o - 1],
                                  "WindPersistenceCurve": curve(g.wind[o - 1])})
    x = {name: sx(g.wind[o - 1]).reshape(-1, 1) for name, o in {"train": s.train, **parts}.items()}
    y_tr, y_va = sy(g.power[s.train + h - 1]), sy(g.power[s.val + h - 1])
    beta = np.polyfit(x["train"].ravel(), y_tr, 1)
    for name in parts:
        res[name]["LinearRegression"] = sy_inv(np.polyval(beta, x[name].ravel()))
    info = {"site": g.site, "test_year": year, "target": target, "horizon_h": h,
            "n_train": len(s.train), "n_val": len(s.val), "n_test": len(s.test),
            "train_last_target": str(g.time[s.train_last_target])}
    if with_mlp:
        t0 = time.time()
        (p_va, p_te), epochs = mlp_forecasts(x["train"], y_tr, x["val"], y_va, [x["val"], x["test"]])
        res["val"]["MLP"], res["test"]["MLP"] = sy_inv(p_va), sy_inv(p_te)
        info.update({"mlp_seconds": round(time.time() - t0, 1), "mlp_epochs": ",".join(map(str, epochs))})
    stem = f"{g.site}_{year}_{target}_h{h}"
    for name, df in res.items():
        _write_atomic(df, C.FORECASTS / f"{stem}_{name}.csv")
    record = C.FORECASTS / f"{stem}_done.json"
    tmp = record.with_suffix(".tmp")
    tmp.write_text(json.dumps({"protocol_sha256": protocol_hash(), "mlp": bool(with_mlp)}))
    tmp.replace(record)
    return info


def protocol_hash():
    return hashlib.sha256((C.ROOT / "config" / "protocol.json").read_bytes()).hexdigest()


def _write_atomic(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def cell_is_done(site, year, target, h, with_mlp):
    stem = C.FORECASTS / f"{site}_{year}_{target}_h{h}"
    files = [stem.with_name(stem.name + s) for s in ("_val.csv", "_test.csv", "_done.json")]
    if not all(p.exists() for p in files):
        return False
    rec = json.loads(files[2].read_text())
    if rec.get("protocol_sha256") != protocol_hash():
        return False
    return rec.get("mlp", False) or not with_mlp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sites", default=",".join(C.SITES))
    ap.add_argument("--years", default=",".join(map(str, C.TEST_YEARS)))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-mlp", action="store_true")
    args = ap.parse_args(argv)
    C.ensure_dirs()
    if args.threads:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    log = []
    tag = f"{args.sites.replace(',', '-')}_{args.years.replace(',', '-')}"
    for site in args.sites.split(","):
        g = load_grid(site)
        for year in map(int, args.years.split(",")):
            for target in C.TARGETS:
                for h in C.HORIZONS:
                    if cell_is_done(site, year, target, h, not args.no_mlp):
                        continue
                    log.append(run_cell(g, year, target, h, not args.no_mlp))
                    print(json.dumps(log[-1]), flush=True)
                    pd.DataFrame(log).to_csv(C.FORECASTS / f"run_log_{tag}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
