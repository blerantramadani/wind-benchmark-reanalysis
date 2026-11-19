# Numbers in the reply to RC1

This file lists the numbers in the authors' reply to referee 1 (RC1) in the WES discussion, with the file, rows and column they come from. The multi-year numbers (point 3b(i) and the revised conclusion) are checked by `tests/test_claims.py` against `expected/`, the 2018 numbers (points 1, 2, 3a, 3b(ii) and 4) by `tests/test_claims_2018.py` against `expected/y2018/`. `pytest tests/test_claims.py tests/test_claims_2018.py` therefore confirms that the reply and the published results agree. After a full run, the same rows can be read from `outputs/` and `outputs/y2018/`.

In the code and in file names, the main target is called `normal`. In the reply it is called "hours without a recorded stop": the curve and scatter are fitted without hours that overlap a logged stop, and only target hours without such an overlap are evaluated. Warnings without a stop are not excluded.

## Files

| Short name | File | Written by |
|---|---|---|
| G | `expected/gaps_with_intervals_L48.csv` | `wbr results --merge` |
| P | `expected/point_estimates.csv` | `wbr results --points` |
| S | `expected/benchmark_summary.csv` | `wbr benchmark` |

Unless stated otherwise, the rows are `target = normal` and `period = 2019-2023` (the five test years pooled, with the method selected in each year).

## Point 3b(i) and the revised conclusion

| Statement in the reply | Value | Rows | Column |
|---|---|---|---|
| 1-h benchmark by test year | 9.7–10.6 % (Kelmarsh), 11.0–11.6 % (Penmanshiel) | S, `horizon_h = 1` | min and max of `bench_test_main_pct` |
| 48-h benchmark by test year | 28.2–29.1 % and 32.9–33.4 % | S, `horizon_h = 48` | same |
| AR order at Penmanshiel | 4 or 5 | S, `site = penmanshiel` | `ar_order_main` |
| Selected method, 18 site–horizon cases | −0.90 to +0.31 pp | G, `method = Selected` | `gap_main_pp` |
| Gaps and intervals listed for 1, 6, 24 and 48 h | e.g. Penmanshiel 24 h: −0.90 [−1.59, −0.20] | G, `method = Selected` | `gap_main_pp`, `gap_ci95_low_pp`, `gap_ci95_high_pp` |
| Intervals above zero; below zero | none; Penmanshiel 18, 24 and 36 h | G, `method = Selected` | `gap_ci95_low_pp > 0`; `gap_ci95_high_pp < 0` |
| Selected method at Penmanshiel, 18–48 h | linear regression in four of five test years | G, `method = Selected` | `picks` |
| Upper end of the 48-h interval at Penmanshiel | +0.01 pp | G, `method = Selected` | `gap_ci95_high_pp` |
| Linear regression alone at Penmanshiel, 18–48 h | 0.76–0.94 pp below, all four intervals exclude zero | G, `method = LinearRegression` | `gap_main_pp`, `gap_ci95_high_pp` |
| Model's own forecast | −0.42 to +0.92 pp; about ±0.4 pp up to 24 h | G, `method = Bayes_main` | `gap_main_pp` |
| Model-forecast intervals excluding zero (test hours resampled) | one: Kelmarsh 48 h, +0.92 [0.11, 1.65] | G, `method = Bayes_main` | `gap_ci95_testonly_low_pp`, `gap_ci95_testonly_high_pp` |
| Lowest test error among the individual methods | linear regression or model forecast in 17 of 18 cases, MLP at Penmanshiel 6 h | P, methods `WindPersistenceCurve`, `LinearRegression`, `MLP`, `Bayes_main`, `Bayes_sens` | lowest `rmse_pct` per site and horizon |
| Single test years, 48 h | −2.4 to +3.1 pp, positive in 2020 at both sites | G, `method = Selected`, `horizon_h = 48`, `period = 2019 … 2023` | `gap_main_pp` |
| Model forecast by season, 48 h, one AR model per year | winter +3.8 to +6.7 pp, summer −2.4 to −4.2 pp | G, `method = Bayes_main`, `horizon_h = 48`, `period = season_DJF`, `season_JJA` | `gap_main_pp` |
| Same, one AR model per season | winter +1.3 to +3.1 pp; summer at Penmanshiel +3.6 pp | same rows | `gap_seasonal_ar_pp` |
| Bootstrap | 48-h blocks, 1,000 resamples, benchmark refitted | `config/protocol.json`, G | `bootstrap`; `n_resamples` |

The revised conclusion repeats three of these results: the range of the selected method, the ±0.4 pp of the model forecast up to 24 h, and the Penmanshiel gaps of −0.70 to −0.90 pp at 18–48 h.

## The 2018 configuration of the preprint (points 1, 2, 3a, 3b(ii) and 4)

These numbers come from `wbr y2018` (`src/wbr/y2018/`), which reads the preprint's files in `preprint/`, the archived per-origin forecasts in `archived/` and the 2018 SCADA archives. All files below are in `expected/y2018/`.

| Statement in the reply | Value | File | Rows and column |
|---|---|---|---|
| Published scores reproduced from the archived forecasts | all 126 within 1.4 × 10⁻⁷ pp | `reproduction_summary.json` | `max_abs_diff_pp`; per score in `reproduction_scores.csv` |
| Stops behind the Kelmarsh zero-output hours | manual stop 13:32 UTC on 6 March; "Battery test", Technical Standby, 70.1 h, until 12:53 on 9 March | `status_logs.json` | `notable_stops.kelmarsh.stops_6_to_9_march` |
| Zero output at wind above 5 m/s, with a logged stop | 62 of 62 (Kelmarsh), 19 of 19 (Penmanshiel) | `status_logs.json` | `zero_output_hours` |
| Pre-test hours touched by stops | 2.8 % and 2.0 % | `status_logs.json` | `hours_touched_pretest.stop` / `hours.pretest` |
| Hours per retained bin above 10 m/s | 20 to 146 at Penmanshiel | `status_logs.json` | `penmanshiel.hours_per_retained_bin_above_10ms` |
| 1-h benchmark (as Table 2) without stop hours | 10.65 → 9.68 % (Kelmarsh), 11.61 → 11.21 % (Penmanshiel); below 0.2 pp change for h ≥ 12 h | `filter_effect.csv` | `bench_pretest_pct`, variants `V0_published`, `V1_without_stops` |
| Penmanshiel hours lost for lack of temperature | 2,317 (26 % of the year), 1,910 with positive power, 177 overlapping a stop, February to May | `status_logs.json` | `penmanshiel_missing_temperature`; temperature coverage in `penmanshiel.valid_share_by_month` |
| Rebuilt Penmanshiel series | 1-h benchmark 14.25 %, 11.01 % without stops; 134-h icing stop | `filter_effect.csv`, `status_logs.json` | variants `V3_rebuilt`, `V4_rebuilt_without_stops`; `notable_stops.penmanshiel` |
| Curvature at the expansion point of Fig. 1 | 725, 424 and 0 kW (m s⁻¹)⁻² on 400, 800 and 1,600 or more points; 0.02–0.03 m/s from a knot; 0 at Penmanshiel | `figure1_grid.csv` | `curvature_kw_per_ms2`, `distance_to_nearest_knot_ms` |
| Bin width (point 3a) | 0.36 and 0.51 m/s | `figure1_grid.csv` | `knot_spacing_ms` |
| Taylor percentages of the preprint | 22–61 % on the standard-deviation scale; 48–70 % and 109–158 % on the variance scale | `taylor_units.csv` | `sd_error_pct`, `variance_error_pct` at `horizon_h = 48` |
| Smooth curves without stop hours | 3.3–3.4 % and 31–38 % (73–90 % variance scale) at Kelmarsh; 5.4–5.5 % and 61–77 % (161–213 %) at Penmanshiel; grows with horizon; quadrature change below 0.007 % | `taylor_smooth_curves.csv` | `taylor2_sd_error_pct`, `taylor2_var_error_pct`, `quad_change_128_256_pct`, `data = without_stops` |
| Benchmark at the test origins | Kelmarsh higher at every horizon (11.61 % against 10.65 % at 1 h); Penmanshiel −0.12 to +0.20 pp | `matched_all_methods.csv` | `bench_test_public_fit_pct` − `bench_pretest_published_pct` |
| Best method on matched origins | −0.07 to +4.17 pp (0.89–4.52 pp in the preprint); one of 18 intervals excludes zero with the test sample resampled, none with refit | `matched_best_method.csv` | `gap_test_public_fit_pp`, `gap_pretest_published_pp`, `gap_ci95_test_*`, `gap_ci95_fit_*` |
| Autocorrelation of the errors | at 48 h: 0.85–0.89 at a lag of 1 h, 0.40–0.53 at 6 h, −0.01 to 0.17 at 24 h, about zero at 48 h | `matched_summary.json` | `autocorrelation_of_squared_error_minus_variance` |
| Model's own forecast at 48 h | 32.84 % against 28.51 % (Kelmarsh), mean error +149 kW; 35.35 % against 33.12 % (Penmanshiel); best method in Table 3 32.68 % | `calibration.csv`, `matched_best_method.csv` | `target = all_states`, `horizon_h = 48` |
| Test weeks windier than the pre-test mean | 6.78 against 5.99 m/s and 8.06 against 7.23 m/s | `reproduction_summary.json` | `test_window_wind` |
| Hour convention | label t is the mean of the records t to t + 50 min | `reproduction_summary.json` | `hour_convention` |
| Targets after the first forecast of the next block | the last h − 1 training and validation targets; benchmark fit ends h − 1 hours after the first test forecast; no test target in training, validation or fit | `reproduction_splits.csv`, `purged_splits_summary.json` | `*_after_first_*`, `benchmark_fit_hours_after_first_test_issue`, `test_targets_in_*` |
| Effect of purging | persistence −0.15 to +0.17 pp, linear regression −0.02 to +0.11 pp, Table 2 benchmark −0.02 to +0.05 pp | `purged_splits_summary.json` | `*_change_pp_range` |
| Networks retrained at 48 h | published scores reproduced within 0.13 pp; purging +0.33 to +0.60 pp (Kelmarsh), −0.07 to +0.56 pp (Penmanshiel) | `retrained_h48_scores.csv` | `rerun_minus_published_pp`, `purging_change_pp` |

Two remarks:

- "The errors are autocorrelated over about a day" holds at long horizons, where the intervals are wide. At 1 h the autocorrelation is 0.05–0.50 at a lag of one hour and near zero after six hours.
- The 126 scores are recomputed from the archived forecasts; the networks were not retrained for that check. The retrained networks at 48 h are scored from `archived/retrained_h48`; scores read from these files differ from the values computed in memory during training by less than 10⁻⁷ pp.

## Numbers this repository does not reproduce

- "Together these changes move the pooled gaps of the selected method by at most 0.16 pp." The figure compares the final results with those obtained under the rules in force before the changes listed in `PROTOCOL.md` (the largest change is 0.155 pp). The earlier results are not part of this repository.
- "Penmanshiel WT01 has no 2024 data, and its 2016 record starts in June." The first statement is visible in `data/zenodo_files.tsv` (tested); the second in `data/processed/availability.json` after `wbr preprocess`.

## Revision

These results were computed after the reply, as promised there: the sequence networks of 2019 as selection candidates, the block-length sensitivity of the intervals and the 2018 networks retrained under the purging rule at every horizon of Table 3. `tests/test_revision.py` checks them against `expected/revision/` and `expected/y2018/retrained_scores.csv`.

| Result | Value | File | Column |
|---|---|---|---|
| Networks selected in 2019 | GRU at Penmanshiel 3 and 6 h; at Kelmarsh no network at any horizon | `revision/point_estimates.csv`, `period = 2019` | `picks` |
| Rows that differ from the reply | only the selected method at Penmanshiel 3 and 6 h (2019, pooled, four seasons) | `revision/gaps_with_intervals_L48.csv` against `gaps_with_intervals_L48.csv` | all |
| Pooled gap at Penmanshiel | 3 h: −0.11 [−0.53, +0.28] → −0.13 [−0.55, +0.27]; 6 h: −0.19 [−0.70, +0.34] → −0.23 [−0.74, +0.30] pp | `revision/gaps_with_intervals_L48.csv` | `gap_main_pp`, `gap_ci95_*` |
| Statements of the reply | range −0.90 to +0.31 pp, no interval above zero, below zero at Penmanshiel 18, 24 and 36 h: unchanged | `revision/gaps_with_intervals_L48.csv` | as in point 3b(i) |
| Block length 24 / 48 / 96 h | pooled selected method: no interval above zero; 4 / 3 / 3 below; median width 1.08 / 1.21 / 1.38 pp | `revision/block_sensitivity.csv`, `revision/gaps_with_intervals_L{24,48,96}.csv` | `above_zero`, `below_zero`, `median_width_pp` |
| 2018 networks retrained at 2–48 h | MLP, LSTM, GRU and DLinear reproduce Table 3 within 0.18 pp; PatchTST within 1.74 pp | `y2018/retrained_scores.csv` | `rerun_minus_published_pp` |
| Effect of purging, 2–36 h | MLP, LSTM, GRU and DLinear −0.42 to +0.33 pp, within ±0.16 pp up to 12 h; PatchTST −2.07 to +0.95 pp | `y2018/retrained_scores.csv` | `purging_change_pp` |

PatchTST changes by up to 1.74 pp between two trainings with the same split, so its purging changes cannot be told apart from that spread.
