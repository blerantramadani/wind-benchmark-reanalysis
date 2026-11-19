"""Sequence networks of the preprint for the 2019 test year (protocol, Sect. 8).

LSTM, GRU, DLinear and PatchTST, target ``normal`` (hours without a recorded stop), horizons 1, 3, 6, 12,
24 and 48 h, both sites, ten seeds. The input is the scaled wind speed of the
48 hours before each origin; architectures and training settings are those of
the preprint (``wbr.architectures``). Each (site, model, horizon) is written
separately, so the job can be split across processes and restarted.

A cell counts as finished when its validation and test files and a record
``<cell>_done.json`` with the hash of ``config/protocol.json`` exist. The record
holds the stopping epoch of every seed, the run time, the number of threads and
the TensorFlow version. ``run_log_<site>.csv`` is rebuilt from all records of
the site after each cell, so a restart keeps the earlier rows. The record's
run time is wall time.

``--horizons 48,24`` runs only these horizons, in this order, so that a second
process can work on the same site and model from the other end. A finished
cell is skipped, and a process that finishes a cell another process has
already written discards its own forecasts, so a cell's files always come from
one run.

Usage: python -m wbr.networks --site kelmarsh --models LSTM,DLinear [--horizons 48,24] [--threads 2]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import config as C
from .forecasts import _write_atomic, protocol_hash, scalers
from .split import load_grid, split

CFG = C.P["networks_2019"]


def cell_stem(site, name, h):
    return f"{site}_{CFG['test_year']}_{CFG['target']}_h{h}_{name}"


def cell_is_done(site, name, h):
    stem = cell_stem(site, name, h)
    files = [C.NETWORKS / f"{stem}_{p}" for p in ("val.csv", "test.csv", "done.json")]
    if not all(p.exists() for p in files):
        return False
    return json.loads(files[2].read_text()).get("protocol_sha256") == protocol_hash()


def write_run_log(site):
    """Rebuild the site's run log; the temporary name carries the process id, since two processes may share a site."""
    rows = [json.loads(p.read_text()) for p in sorted(C.NETWORKS.glob(f"{site}_*_done.json"))]
    if rows:
        path = C.NETWORKS / f"run_log_{site}.csv"
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        log = pd.DataFrame(rows).drop(columns="protocol_sha256").sort_values(["model", "horizon_h"])
        log.to_csv(tmp, index=False)
        tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", required=True)
    ap.add_argument("--models", default=",".join(CFG["models"]))
    ap.add_argument("--horizons", default=",".join(map(str, CFG["horizons_h"])),
                    help="horizons to run, in this order (default: all)")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    horizons = [int(h) for h in args.horizons.split(",")]
    if not set(horizons) <= set(CFG["horizons_h"]):
        ap.error(f"--horizons must be among {CFG['horizons_h']}")
    C.ensure_dirs()
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from .architectures import ARCHITECTURES, BATCH_SIZE, EPOCHS, LOOK_BACK
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    g = load_grid(args.site)
    year, target = CFG["test_year"], CFG["target"]
    for name in args.models.split(","):
        for h in horizons:
            if cell_is_done(args.site, name, h):
                continue
            stem = cell_stem(args.site, name, h)
            s = split(g, year, h, target)
            sx, sy, sy_inv = scalers(g, s)
            x, y = sx(g.wind), sy(g.power)

            def seq(o):
                return np.stack([x[i - LOOK_BACK:i] for i in o])[..., None]

            xtr, xva, xte = seq(s.train), seq(s.val), seq(s.test)
            ytr, yva = y[s.train + h - 1], y[s.val + h - 1]
            preds = {part: {"origin": o, "target_time": g.time[o + h - 1], "actual": g.power[o + h - 1]}
                     for part, o in (("val", s.val), ("test", s.test))}
            epochs, t0 = [], time.time()
            for seed in range(C.P["seeds"]):
                np.random.seed(seed)
                tf.random.set_seed(seed)
                model = ARCHITECTURES[name]((LOOK_BACK, 1), seed)
                hist = model.fit(xtr, ytr, validation_data=(xva, yva), shuffle=False, epochs=EPOCHS,
                                 batch_size=BATCH_SIZE, verbose=0,
                                 callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
                epochs.append(len(hist.history["loss"]))
                for part, xx in (("val", xva), ("test", xte)):
                    preds[part][f"{name}_seed{seed}"] = sy_inv(model.predict(xx, verbose=0).ravel())
                tf.keras.backend.clear_session()
            if cell_is_done(args.site, name, h):
                print(f"{stem}: finished meanwhile by another process; these forecasts are discarded", flush=True)
                continue
            for part in ("val", "test"):
                df = pd.DataFrame(preds[part])
                df[name] = df[[c for c in df.columns if c.startswith(f"{name}_seed")]].mean(axis=1)
                _write_atomic(df, C.NETWORKS / f"{stem}_{part}.csv")
            record = {"site": args.site, "test_year": year, "target": target, "model": name, "horizon_h": h,
                      "n_train": len(s.train), "n_val": len(s.val), "n_test": len(s.test),
                      "seconds": round(time.time() - t0, 1), "epochs": ",".join(map(str, epochs)),
                      "threads": args.threads, "tensorflow": tf.__version__, "protocol_sha256": protocol_hash()}
            done = C.NETWORKS / f"{stem}_done.json"
            tmp = done.with_name(done.name + ".tmp")
            tmp.write_text(json.dumps(record))
            tmp.replace(done)
            write_run_log(args.site)
            print(json.dumps({k: v for k, v in record.items() if k != "protocol_sha256"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
