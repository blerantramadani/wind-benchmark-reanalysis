# Wind-benchmark reanalysis

Code, configuration and results of the reanalysis reported in the authors' reply to referee 1 (RC1) on the preprint

> Ramadani, B. and Fustic, V.: A Semi-Analytic Wind-State Predictability Benchmark for Single-Turbine Wind Power Forecasting, Wind Energ. Sci. Discuss., https://doi.org/10.5194/wes-2026-154, 2026.

The preprint compared a model-conditional benchmark with the error of wind-only forecasts on a single six-week test window in 2018. This repository repeats the comparison on five calendar test years (2019–2023) for the same two turbines, with an explicit operating-state target, purged chronological splits, the benchmark evaluated on the same test origins as the forecasts, and bootstrap intervals that include the fitting of the benchmark. The analysis choices were fixed before the new years were opened; `PROTOCOL.md` gives the protocol and every later change with its date and reason.

The repository also contains the analyses of the preprint's own 2018 configuration that the reply reports in points 1, 2, 3a, 3b(ii) and 4: operating states and the state filter, Figure 1 and the Taylor comparison, the benchmark on the test origins, and the chronological boundaries for h > 1 (`src/wbr/y2018/`).

Version 1.0.0 of this repository and the stored outputs are archived on Zenodo: https://doi.org/10.5281/zenodo.24355795.

## Contents

| Path | Content |
|---|---|
| `config/protocol.json` | every fixed analysis setting |
| `data/zenodo_files.tsv` | the 19 SCADA archives used, with md5 sums |
| `src/wbr/` | the multi-year analysis, one module per step |
| `src/wbr/y2018/` | the 2018 analyses, one module per point of the reply |
| `preprint/` | the files of the preprint repository (commit 13c0bba) that the 2018 analyses read, unchanged, with SHA-256 sums |
| `archived/` | per-origin forecasts behind Table 3 of the preprint (not in that commit) and the networks retrained at 2–48 h, with SHA-256 sums |
| `scripts/quadrature_check.py` | benchmark and model forecast with 64, 128 and 256 quadrature nodes |
| `tests/` | unit tests; `test_claims.py` and `test_claims_2018.py` check the numbers of the reply against `expected/`, `test_revision.py` the results added for the revision |
| `expected/` | results of the reply to RC1; `expected/revision/` with the 2019 networks and the block-length sensitivity; `expected/y2018/` for the 2018 analyses |
| `PROTOCOL.md` | analysis protocol and every later change, with dates |
| `DATA.md` | data sources, licences and processing rules |
| `CLAIMS.md` | each number in the reply, with the file, rows and column it comes from |
| `CHANGELOG.md` | changes to the code |

The benchmark functions are taken unchanged from the preprint repository (github.com/blerantramadani/wind-state-predictability-benchmark, commit 13c0bba), so the benchmark has the same definition as in the preprint.

## Installation

Python 3.9 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[test]"
```

`requirements.txt` pins the versions used for the published results. TensorFlow is needed only for the MLP and the sequence networks.

## Reproduction

Three levels are supported.

Numbers in the reply, a few seconds and no data needed:

```bash
pytest tests/test_claims.py tests/test_claims_2018.py
```

This reads `expected/` and checks every number quoted in the reply, row by row as listed in `CLAIMS.md`.

Point estimates from the stored forecasts, under a minute once the files are in place. Download `wind-benchmark-reanalysis-outputs-1.0.0.zip` (386 MB) from the Zenodo record of this repository (https://doi.org/10.5281/zenodo.24355795), unpack it in the repository root and check it:

```bash
unzip wind-benchmark-reanalysis-outputs-1.0.0.zip
shasum -a 256 -c outputs/SHA256SUMS      # sha256sum -c on Linux
wbr check                                # with the 2019 networks, against expected/revision/
wbr check --no-networks                  # as in the reply, against expected/
```

The archive holds the hourly series and stop events (`data/processed/`), the benchmark variances and model forecasts (`outputs/benchmark/`), the persistence, linear-regression and MLP forecasts with the model's own forecasts (`outputs/forecasts/`), the forecasts of the 2019 sequence networks with their records (`outputs/networks_2019/`) and the quadrature check (`outputs/checks/`). `wbr check` rebuilds every point estimate from these files. It prints `REPRODUCED` when all values agree to 1e-9 percentage points and the methods selected in each year are identical.

Full run from the raw data:

```bash
wbr download                 # 7.2 GB from Zenodo, md5-checked
wbr preprocess               # hourly series 2016-2023 and stop flags
wbr benchmark                # benchmark variances and model forecasts
wbr forecasts                # persistence, linear regression, MLP
wbr networks --site kelmarsh     # LSTM, GRU, DLinear and PatchTST for 2019
wbr networks --site penmanshiel
wbr results --points
wbr results                  # bootstrap intervals
wbr results --merge
wbr tables                   # outputs/results/tables.md and summary.json
wbr check --intervals        # compare point estimates and intervals with expected/
```

The reply to RC1 reported the results before the sequence networks were trained. Adding `--no-networks` to `wbr results` and `wbr check` (or running `wbr all --no-networks`) leaves the networks out and reproduces those results; `expected/` holds them. With the networks, they are candidates in the 2019 model selection for the target without recorded stops, at 1, 3, 6, 12, 24 and 48 h (`PROTOCOL.md`, 7 Nov 2026), and `wbr check` compares with `expected/revision/`. GRU is then selected at Penmanshiel 3 and 6 h in 2019; no other row changes (`CLAIMS.md`, Revision).

Block-length sensitivity (blocks of 24 and 96 hours instead of 48):

```bash
wbr results --block 24       # likewise --block 96; one job per site and target with --sites and --targets
wbr results --block 24 --merge
wbr tables --blocks 24,48,96 # outputs/results/block_sensitivity.md and .csv
```

The 2018 analyses need only the preprint files and archived forecasts in this repository and the two 2018 SCADA archives (after `wbr download`):

```bash
wbr y2018                    # outputs/y2018/, about 2-4 minutes
wbr check --y2018            # compare with expected/y2018/
wbr y2018 --retrain          # optional: retrain the five networks at 48 h (several hours)
python -m wbr.y2018.purging  # retrain them at 2, 3, 6, 12, 18, 24 and 36 h for the revision
```

The reply promised to retrain all networks under the purging rule. `python -m wbr.y2018.purging` trains the five networks at the other horizons with h > 1, with the original and the purged splits, so that the effect of purging is measured on runs from the same machine; at h = 1 purging removes nothing. It accepts `--sites` and `--horizons`, so it can be split across processes, and skips finished units on restart. Its forecasts are archived in `archived/retrained_2018` (one record per unit with the stopping epochs, run time, threads and TensorFlow version), and `wbr y2018` writes `retrained_scores.csv` for all eight horizons from them.

They reproduce every file in `expected/y2018/` exactly, including the bootstrap intervals (seeds 20261102, 31 and 7). The 126 published scores are recomputed from the archived forecasts, not by retraining the networks. Without `--retrain`, the networks retrained at 48 h are scored from `archived/retrained_h48`.

The work can be split across processes: `wbr forecasts` accepts `--sites` and `--years`, `wbr results` accepts `--sites` and `--targets`, and `wbr networks` takes one site and, with `--models LSTM,GRU`, a subset of the networks; with `--horizons 48,24` a second process can work on the same site and model from the long horizons down. `wbr forecasts` and `wbr networks` skip completed cells on restart, and a network cell that another process finished first is not overwritten.

Measured run times on an Apple M5 with 10 cores and 16 GB of memory: preprocessing 2.5 minutes, benchmark 2.5 minutes, persistence and linear regression under a minute (`wbr forecasts --no-mlp`), the MLP about 14 hours of computing in total (we split it over four to six processes), and the bootstrap 15–16 minutes with one process per site and target running in parallel. The download time depends on the connection. The sequence networks for 2019 (`wbr networks`, 48 cells of ten seeds) took about 30 s per training epoch for LSTM and GRU with four to eight processes of two threads running at once, so a cell took from 20 minutes, when every seed stopped after six epochs, to about four hours at 50 epochs; in total about 60 process-hours for LSTM and GRU and about 4 for DLinear and PatchTST. The run time in each `<cell>_done.json` is wall time and includes the pauses listed in `outputs/networks_2019/logs/pauses.txt` (the computer was on battery or asleep).

When several processes run at once, set `OPENBLAS_NUM_THREADS=1` (and `OMP_NUM_THREADS=1`) before starting them. Otherwise their BLAS threads compete for the cores: with four bootstrap processes, one resample took 14 s instead of 1 s.

## What is reproduced exactly

Preprocessing, splits, benchmark, model forecasts, persistence, linear regression and the bootstrap (fixed seed) are deterministic. We reran all of them with this code on the machine above. The persistence, linear-regression and model forecasts, the benchmark variances and the point estimates are identical to those behind the reply, and the 1,000-resample intervals agree with them to within 1e-14 percentage points, because the order of a few floating-point operations differs. `expected/` holds the results of this rerun. On another processor or BLAS build the last digits can differ, far below the two decimals of the reply.

We also ran a fresh copy of the repository, without outputs or processed data, from the raw archives (`wbr preprocess`, `wbr benchmark`, `wbr forecasts --no-mlp`, `wbr y2018`). The hourly series, the 181 benchmark files, the 360 model-forecast files and the 20 files of the 2018 analyses were byte-identical to the stored ones, and the persistence and linear-regression forecasts were identical in all 360 forecast files.

MLP training with TensorFlow on a CPU did not repeat exactly in our tests, even with fixed seeds. We reran one cell (Kelmarsh, 2019, 48 h) three times on the same machine, with one thread, two threads and TensorFlow's default setting. Each rerun changed the stopping epochs of several seeds, and the test nRMSE of the ten-seed mean differed from the archived value by −0.005 to +0.16 percentage points. The archived forecasts were computed with one or two threads per process (`--threads`). For this reason the MLP forecasts are archived, and `wbr check` rebuilds every reported number from them.

## Licence and citation

Code: MIT licence (`LICENSE`). Data: CC BY 4.0, Cubico Sustainable Investments via Zenodo (`DATA.md`). Please cite this repository (Ramadani, B. and Fustic, V.: Wind-benchmark reanalysis, version 1.0.0, Zenodo, https://doi.org/10.5281/zenodo.24355795, 2026; details in `CITATION.cff`), together with the preprint and the two data releases.
