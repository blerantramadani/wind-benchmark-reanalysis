# Analysis protocol

The protocol below was written on 3 November 2026, after the referee report (RC1) and before the 2016–2017 and 2019–2023 data were opened. The 2018 data and the preprint results had been seen. Changes made later are listed at the end with their date and reason. Machine-readable settings are in `config/protocol.json`.

## Questions

1. Is the model-conditional benchmark stable when it is fitted on earlier years and evaluated on later calendar years?
2. How large is the gap between the error of wind-only forecasts and the benchmark when both are computed on the same test origins, by horizon, with uncertainty that includes the fitting of the benchmark?
3. Does the gap depend on the season, and does a season-dependent AR model change it?

## Data and targets

Kelmarsh turbine 1 and Penmanshiel turbine 01, 2016–2023 (see `DATA.md`). Two targets are evaluated:

- normal operation (main): power curve and scatter fitted without stop hours; models trained and evaluated on target hours without a stop; input wind is never filtered;
- all operating states (secondary): no filter.

The normal-operation target assumes that a stop is known in advance. This holds for scheduled maintenance but not for faults.

## Splits

For each test year Y = 2019, …, 2023: training on 2016 to Y − 2, validation on Y − 1, test on Y. Samples whose targets are not yet known when the first forecast of the next block is issued are removed (purging; `src/wbr/split.py`). An origin needs 48 valid input hours and a valid target.

## Benchmark

As in the preprint: Gaussian AR(p), p = 1, …, 5 selected by BIC on a common contiguous response set; 40-bin empirical curve up to the 99.5th percentile with at least 20 hours per bin; bin variance as scatter; 64-node Gauss–Hermite quadrature. Evaluated at the test origins.

- Main fit: all valid hours known before the first test forecast is issued.
- Sensitivity fit: hours up to the end of the training targets, as for the forecasting models.
- Seasonal AR: the main fit with one AR model per meteorological season.

## Methods

Persistence through the power curve, linear regression on the last hourly wind speed, the MLP of the preprint (all test years and horizons), and LSTM, GRU, DLinear and PatchTST (test year 2019, horizons 1, 3, 6, 12, 24 and 48 h, normal target). Ten seeds per network; the forecast is the seed mean. In each test year the reported method is the one with the lowest validation error.

## Uncertainty

Moving-block bootstrap with 48-hour blocks and 1,000 resamples, blocks of 24 and 96 hours as sensitivity. In every resample the test calendar is resampled and the benchmark is refitted on a block resample of its fitting window. Results are given per test year, pooled over 2019–2023 and by season. The number of intervals that exclude zero is reported for each sign; no joint significance is claimed.

## Changes after approval

- 3 Nov 2026. The four sequence networks are trained for the normal target only, to limit computing time.
- 3 Nov 2026. In the bootstrap, the benchmark is refitted once per test year and target with the cut-off of h = 48 and used for all horizons. Point estimates use the exact cut-off of each horizon. The cut-offs differ by less than 47 hours on three to seven years of data.
- 3 Nov 2026. With more data, BIC selects AR(4) or AR(5) at Penmanshiel. This follows the rule above and is not a change of it.
- 3 Nov 2026, before any multi-year gap was seen. Calibration check added: the model's own conditional-mean forecast E[m(V) | wind history] is evaluated with the main and the sensitivity fit. If the model is calibrated, its error should match the benchmark. The main-fit variant is not eligible for model selection because its fit includes the validation year. Test-only intervals (benchmark fixed) are added for this check.
- 6 Nov 2026, after the multi-year results had been seen. Negative power is clipped after hourly averaging at both sites; before, ten-minute values were clipped first. The change aligns this analysis with a parallel analysis of the same data. Its effect is below 0.01 percentage points on the benchmark and below 0.005 percentage points on the gaps of persistence, linear regression and the model forecast.
- 6 Nov 2026, after an external review of the draft response. (1) The pooled 2019–2023 result now combines the methods selected in each year; before, one method was chosen from the mean of the five validation errors. (2) Bootstrap blocks are taken on the full test calendar, so that a block never spans a data gap. (3) A season (for example January, February and December of the same year) is made of separate runs of hours; season blocks are drawn within each run and never join the end of one run to the start of the next.
- 7 Nov 2026, wording only. In the reply and in the revised manuscript the main target is called "hours without a recorded stop" instead of "normal operation". Power limitations logged as warnings, without a stop, are not excluded, so the evaluation is conditional on the status recorded at the target hour. The code and the file names keep the name `normal`. No computation changed.
- 7 Nov 2026. The block-length sensitivity (24 and 96 hours) had not been run when the reply was posted. It will be reported with the revision.
- 7 Nov 2026, after the reply was posted and before any forecast of the sequence networks was scored. How the networks enter the results: in 2019, for the normal target, they are candidates in the model selection at their six horizons (1, 3, 6, 12, 24 and 48 h); at 2, 18 and 36 h, where they are not trained, the selection is among the other methods, as before. Because they exist only in 2019, they have no pooled or seasonal rows of their own, but a network selected in 2019 enters the pooled and seasonal results of the selected method. The networks were trained in five processes with two TensorFlow threads each. The reply reported the results without the networks; `--no-networks` reproduces them.
