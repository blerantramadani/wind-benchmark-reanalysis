# Block-length sensitivity of the bootstrap intervals

Selected method: full intervals (benchmark refitted in each resample). Model forecast (main fit): test-only intervals, as in the calibration check.

| Target | Method | Scope | Cells | L = 24 h: above / below 0, median width (pp) | L = 48 h: above / below 0, median width (pp) | L = 96 h: above / below 0, median width (pp) |
|---|---|---|---:|---|---|---|
| normal | selected method | pooled 2019-2023 | 18 | 0 / 4, 1.08 | 0 / 3, 1.21 | 0 / 3, 1.38 |
| normal | selected method | single years | 90 | 8 / 18, 2.39 | 7 / 17, 2.69 | 6 / 15, 3.00 |
| normal | model forecast (main fit) | pooled 2019-2023 | 18 | 1 / 0, 0.91 | 1 / 0, 1.00 | 0 / 0, 1.10 |
| normal | model forecast (main fit) | single years | 90 | 12 / 18, 1.94 | 11 / 15, 2.13 | 9 / 13, 2.38 |
| all_states | selected method | pooled 2019-2023 | 18 | 0 / 8, 1.18 | 0 / 8, 1.29 | 0 / 6, 1.48 |
| all_states | selected method | single years | 90 | 5 / 23, 2.61 | 5 / 21, 2.96 | 5 / 15, 3.39 |
| all_states | model forecast (main fit) | pooled 2019-2023 | 18 | 2 / 7, 0.96 | 1 / 6, 1.04 | 0 / 6, 1.16 |
| all_states | model forecast (main fit) | single years | 90 | 7 / 21, 2.05 | 7 / 19, 2.29 | 7 / 17, 2.50 |

Cells whose classification differs from L = 48 h:

- L = 24 h, kelmarsh normal 2020 h = 6, model forecast (main fit): [-0.01, +1.95] (includes 0) → [+0.12, +1.86] (above)
- L = 24 h, kelmarsh normal 2020 h = 6, selected method: [-0.05, +2.14] (includes 0) → [+0.03, +2.06] (above)
- L = 24 h, kelmarsh normal 2022 h = 12, model forecast (main fit): [-2.18, +0.01] (includes 0) → [-2.12, -0.16] (below)
- L = 24 h, kelmarsh normal 2022 h = 24, model forecast (main fit): [-2.70, +0.04] (includes 0) → [-2.61, -0.08] (below)
- L = 24 h, kelmarsh all_states 2020 h = 3, selected method: [-2.28, +0.01] (includes 0) → [-2.11, -0.16] (below)
- L = 24 h, kelmarsh all_states 2022 h = 24, model forecast (main fit): [-2.63, +0.09] (includes 0) → [-2.46, -0.01] (below)
- L = 24 h, kelmarsh all_states 2019-2023 h = 36, model forecast (main fit): [-0.06, +1.41] (includes 0) → [+0.09, +1.32] (above)
- L = 24 h, penmanshiel normal 2021 h = 3, selected method: [-1.82, +0.06] (includes 0) → [-1.61, -0.09] (below)
- L = 24 h, penmanshiel normal 2021 h = 24, model forecast (main fit): [-2.02, +0.06] (includes 0) → [-2.00, -0.00] (below)
- L = 24 h, penmanshiel normal 2019-2023 h = 48, selected method: [-1.43, +0.01] (includes 0) → [-1.36, -0.02] (below)
- L = 24 h, penmanshiel all_states 2019-2023 h = 2, model forecast (main fit): [-1.03, +0.04] (includes 0) → [-0.98, -0.02] (below)
- L = 24 h, penmanshiel all_states 2023 h = 3, model forecast (main fit): [-1.96, +0.02] (includes 0) → [-1.82, -0.02] (below)
- L = 24 h, penmanshiel all_states 2023 h = 3, selected method: [-2.07, +0.02] (includes 0) → [-1.91, -0.09] (below)
- L = 96 h, kelmarsh normal 2022 h = 18, model forecast (main fit): [-2.59, -0.08] (below) → [-2.82, +0.05] (includes 0)
- L = 96 h, kelmarsh normal 2022 h = 18, selected method: [-2.89, -0.14] (below) → [-3.15, +0.18] (includes 0)
- L = 96 h, kelmarsh normal 2022 h = 24, selected method: [-3.20, -0.20] (below) → [-3.58, +0.10] (includes 0)
- L = 96 h, kelmarsh normal 2019-2023 h = 48, model forecast (main fit): [+0.11, +1.65] (above) → [-0.07, +1.83] (includes 0)
- L = 96 h, kelmarsh all_states 2022 h = 12, selected method: [-2.32, -0.11] (below) → [-2.46, +0.08] (includes 0)
- L = 96 h, kelmarsh all_states 2022 h = 18, model forecast (main fit): [-2.60, -0.11] (below) → [-2.80, +0.08] (includes 0)
- L = 96 h, kelmarsh all_states 2022 h = 18, selected method: [-2.93, -0.20] (below) → [-3.21, +0.17] (includes 0)
- L = 96 h, kelmarsh all_states 2022 h = 24, selected method: [-3.20, -0.20] (below) → [-3.63, +0.13] (includes 0)
- L = 96 h, kelmarsh all_states 2019-2023 h = 48, model forecast (main fit): [+0.17, +1.73] (above) → [-0.01, +1.87] (includes 0)
- L = 96 h, penmanshiel normal 2020 h = 1, model forecast (main fit): [+0.05, +1.97] (above) → [-0.05, +1.96] (includes 0)
- L = 96 h, penmanshiel normal 2020 h = 3, selected method: [+0.07, +2.20] (above) → [-0.06, +2.32] (includes 0)
- L = 96 h, penmanshiel normal 2021 h = 3, model forecast (main fit): [-1.55, -0.06] (below) → [-1.68, +0.00] (includes 0)
- L = 96 h, penmanshiel normal 2020 h = 6, model forecast (main fit): [+0.04, +2.23] (above) → [-0.07, +2.18] (includes 0)
- L = 96 h, penmanshiel normal 2021 h = 24, model forecast (main fit): [-2.02, +0.06] (includes 0) → [-2.13, -0.08] (below)
- L = 96 h, penmanshiel normal 2023 h = 48, model forecast (main fit): [-2.36, -0.15] (below) → [-2.43, +0.12] (includes 0)
- L = 96 h, penmanshiel all_states 2023 h = 1, selected method: [-2.04, -0.01] (below) → [-2.11, +0.12] (includes 0)
- L = 96 h, penmanshiel all_states 2019-2023 h = 1, selected method: [-1.68, -0.05] (below) → [-1.78, +0.08] (includes 0)
- L = 96 h, penmanshiel all_states 2023 h = 2, selected method: [-1.98, -0.04] (below) → [-2.01, +0.12] (includes 0)
- L = 96 h, penmanshiel all_states 2023 h = 6, model forecast (main fit): [-2.32, -0.24] (below) → [-2.37, +0.03] (includes 0)
- L = 96 h, penmanshiel all_states 2023 h = 6, selected method: [-2.51, -0.09] (below) → [-2.59, +0.18] (includes 0)
- L = 96 h, penmanshiel all_states 2019-2023 h = 48, selected method: [-1.50, -0.03] (below) → [-1.66, +0.07] (includes 0)
