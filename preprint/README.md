# A Semi-Analytic Wind-State Predictability Benchmark for Single-Turbine Wind Power Forecasting

This repository accompanies the manuscript *A Semi-Analytic Wind-State Predictability Benchmark for Single-Turbine Wind Power Forecasting*. It provides the processed hourly input data, analysis code, and final outputs required to reproduce the wind-only predictability benchmark at two UK wind-farm sites.

## Conceptual Overview

For a forecaster restricted exclusively to observed wind-speed history, this benchmark formally defines the conditional variance of future power generation under a fitted wind-dynamics model and an empirical power curve. It strictly serves as a model-conditional reference rather than an absolute information-theoretic limit. It is not a universal lower bound for multivariate forecasting systems supplied with external meteorological variables, such as Numerical Weather Prediction (NWP) inputs.

Specifically, this repository implements the full analytical pipeline to:
- Fit Gaussian AR(1)–AR(5) wind models on the pre-test period and select AR(3) at both sites via BIC, evaluated on a strictly contiguous response set.
- Evaluate the conditional future-wind distribution using deterministic 64-node Gauss–Hermite quadrature, averaged across observed AR(3) state vectors.
- Decompose the benchmark into horizon-dependent curve variation and residual conditional scatter.
- Audit the geometric failure of local Taylor approximations across three-, five-, and seven-bin derivative-smoothing windows.
- Evaluate established wind-only neural architectures (MLP, LSTM, GRU, DLinear, PatchTST) as seed-averaged ensembles across ten random weight initializations.

## Repository Structure

```text
data/                                  Processed hourly SCADA input data
scripts/07_windonly_bound.py           AR-order selection, exact benchmark derivation, and bootstrapping
scripts/08_final_benchmark.py          Final empirical evaluation of wind-only neural architectures
scripts/10_local_approximation_exact.py Audit of local Taylor-approximation errors
scripts/11_build_final_paper_tables.py Compiles Tables 1–3
scripts/12_make_final_figure2.py       Generates Figure 2 (Benchmark vs. Achieved Error)
scripts/13_make_final_figure1.py       Generates Figure 1 (Geometry of Predictability)
results/                               Outputs for the exact benchmark, audit, and paper tables
final_benchmark/                       Seed-ensemble summaries used for Table 3
figures/                               Final submission-ready Figures 1–2 (PNG and PDF)
