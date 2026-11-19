# Data

## Sources

All inputs are public SCADA releases by Cubico Sustainable Investments, distributed on Zenodo under the Creative Commons Attribution 4.0 licence (CC BY 4.0):

- Plumley, C. and Takeuchi, R.: Kelmarsh wind farm data, Zenodo, doi:10.5281/zenodo.16807551 (version of 12 August 2025).
- Plumley, C. and Takeuchi, R.: Penmanshiel wind farm data, Zenodo, doi:10.5281/zenodo.16807304 (version of 13 August 2025).

The preprint used earlier versions of the same releases (doi:10.5281/zenodo.5841834 and doi:10.5281/zenodo.5946808). The Penmanshiel 2018 archive has the same md5 sum in both versions.

This repository uses one turbine per farm, as in the preprint: Kelmarsh turbine 1 and Penmanshiel turbine 01. For each year it reads the ten-minute turbine file (`Turbine_Data_...`) and the status log (`Status_...`) from the yearly archive.

| Turbine | Years | Remarks |
|---|---|---|
| Kelmarsh 1 | 2016–2023 | 2016 starts on 3 January. 2024 exists but is not used, so that both turbines cover the same years. |
| Penmanshiel 01 | 2016–2023 | 2016 starts on 6 June. 2023 is split into four archives and starts on 1 February. 2024 is not released for turbines 1–10. |

`data/zenodo_files.tsv` lists the 19 archives with their record, size and md5 sum. `wbr download` fetches them into `data/raw/` and refuses any file whose md5 does not match. The archives are 7.2 GB in total and are not redistributed here.

## Processing

`wbr preprocess` writes one hourly series per turbine to `data/processed/`. The rules are fixed in `config/protocol.json` and documented in `src/wbr/preprocess.py`:

- only time, `Wind speed (m/s)` and `Power (kW)` are read, so a missing temperature never removes an hour;
- time stamps are UTC; the hour labelled t is the mean of the ten-minute records labelled t to t + 50 min;
- negative hourly mean power is set to zero after averaging;
- an hour is valid with at least four of six ten-minute values of both wind and power, and is invalid if wind is below 2 m/s while power exceeds 100 kW;
- nothing is interpolated;
- an hour is flagged as a stop if it overlaps an interval event with `Status = Stop` in the status log.

Applied to 2018, these rules give the same hourly wind speed as the preprint's hourly files in every hour that both contain. The differences are these:

- Power at Kelmarsh: the preprint clipped negative ten-minute power before averaging, so its value is higher in 779 hours, by at most 4.9 kW.
- Missing hours at Penmanshiel: the preprint file lacks 2,317 hours from February to May 2018, because they had no temperature value.
- Hours with few records: the preprint files keep 10 hours at Kelmarsh and 8 at Penmanshiel that have fewer than four ten-minute values; these hours are invalid here.

## Files of the preprint (2018 analyses)

The 2018 analyses (`wbr y2018`) examine the preprint itself and read three further sources.

- `preprint/` holds the files of the preprint repository (github.com/blerantramadani/wind-state-predictability-benchmark, commit 13c0bba) that these analyses need: scripts 07, 08, 10 and 13 and `architectures.py`, the two hourly 2018 files with their README, the results of scripts 07 and 10, and the score summary of script 08. They are copied unchanged; `preprint/SHA256SUMS` lists their checksums, and `tests/test_y2018_sources.py` checks them. The preprint repository is under the MIT licence (`preprint/LICENSE`); the hourly files are derived from the CC BY 4.0 releases cited in `preprint/data/README.md`.
- `archived/preprint_forecasts/` holds the per-origin test forecasts behind Table 3 of the preprint (18 files, persistence, linear regression and the ten seeds of each network) and the Diebold-Mariano table of the same run. They were not part of commit 13c0bba. Rescored, they reproduce all 126 published scores to within 1.4 × 10⁻⁷ percentage points.
- `archived/retrained_h48/` holds the test forecasts of the five networks retrained at h = 48 with the original and the purged splits (point 4 of the reply), with the run log. Network training on a CPU does not repeat exactly, so the scores are computed from these archived forecasts.
- `archived/retrained_2018/` holds the same for the revision at h = 2, 3, 6, 12, 18, 24 and 36, one record per site, horizon and split with the stopping epochs, run time, threads and TensorFlow version.
- The 2018 turbine data and status logs come from the same Zenodo archives as above (`Kelmarsh_SCADA_2018_3084.zip`, `Penmanshiel_SCADA_2018_WT01-10_3113.zip`).

## Stored outputs (Zenodo)

The SCADA archives are not redistributed. The Zenodo record of this repository (https://doi.org/10.5281/zenodo.24355795) holds `wind-benchmark-reanalysis-outputs-1.0.0.zip` with the files derived from them that `wbr check` needs: the hourly series and stop events, the benchmark variances, the forecasts of all methods for 2019–2023 and the forecasts of the 2019 sequence networks. `outputs/SHA256SUMS` in the archive lists every file. These derived files are under CC BY 4.0, like the releases they come from.

## Attribution

If you reuse the processed series, cite the two Zenodo releases above as well as this repository.
