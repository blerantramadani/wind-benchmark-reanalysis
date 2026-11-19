"""The multi-year numbers of the reply to RC1, checked against expected/ (one test per row of CLAIMS.md).

Values are compared after rounding to the precision used in the reply.
"""
from collections import Counter

import pandas as pd
import pytest

from wbr import config as C

E = C.ROOT / "expected"
pytestmark = pytest.mark.skipif(not (E / "gaps_with_intervals_L48.csv").exists(), reason="expected/ not present")

INDIVIDUAL = ["WindPersistenceCurve", "LinearRegression", "MLP", "Bayes_main", "Bayes_sens"]
YEARS = [str(y) for y in C.TEST_YEARS]


@pytest.fixture(scope="module")
def G():
    return pd.read_csv(E / "gaps_with_intervals_L48.csv", dtype={"period": str})


@pytest.fixture(scope="module")
def P():
    return pd.read_csv(E / "point_estimates.csv", dtype={"period": str})


@pytest.fixture(scope="module")
def S():
    s = pd.read_csv(E / "benchmark_summary.csv")
    return s[s.target == "normal"]


def pick(df, period="2019-2023", **kw):
    m = (df.target == "normal") & (df.period == period)
    for k, v in kw.items():
        m &= df[k].isin(v) if isinstance(v, list) else df[k] == v
    return df[m]


def one(df, **kw):
    r = pick(df, **kw)
    assert len(r) == 1, kw
    return r.iloc[0]


def test_benchmark_by_year(S):
    def span(site, h):
        x = S[(S.site == site) & (S.horizon_h == h)].bench_test_main_pct
        assert len(x) == len(C.TEST_YEARS)
        return round(x.min(), 1), round(x.max(), 1)
    assert span("kelmarsh", 1) == (9.7, 10.6)
    assert span("penmanshiel", 1) == (11.0, 11.6)
    assert span("kelmarsh", 48) == (28.2, 29.1)
    assert span("penmanshiel", 48) == (32.9, 33.4)


def test_ar_order_penmanshiel(S):
    assert set(S[S.site == "penmanshiel"].ar_order_main) <= {4, 5}


def test_selected_range(G):
    sel = pick(G, method="Selected")
    assert len(sel) == 2 * len(C.HORIZONS) == 18
    assert (round(sel.gap_main_pp.min(), 2), round(sel.gap_main_pp.max(), 2)) == (-0.90, 0.31)


@pytest.mark.parametrize("site,h,gap,lo,hi", [
    ("kelmarsh", 1, -0.05, -0.23, 0.14), ("kelmarsh", 6, 0.19, -0.28, 0.63),
    ("kelmarsh", 24, -0.31, -1.01, 0.46), ("kelmarsh", 48, 0.31, -0.55, 1.16),
    ("penmanshiel", 1, -0.22, -0.54, 0.10), ("penmanshiel", 6, -0.19, -0.70, 0.34),
    ("penmanshiel", 24, -0.90, -1.59, -0.20), ("penmanshiel", 48, -0.70, -1.43, 0.01),
])
def test_selected_listed(G, site, h, gap, lo, hi):
    r = one(G, method="Selected", site=site, horizon_h=h)
    assert (round(r.gap_main_pp, 2), round(r.gap_ci95_low_pp, 2), round(r.gap_ci95_high_pp, 2)) == (gap, lo, hi)


def test_selected_intervals_excluding_zero(G):
    sel = pick(G, method="Selected")
    assert (sel.gap_ci95_low_pp > 0).sum() == 0
    below = sel[sel.gap_ci95_high_pp < 0]
    assert sorted(zip(below.site, below.horizon_h)) == [("penmanshiel", 18), ("penmanshiel", 24), ("penmanshiel", 36)]


def test_selected_penmanshiel_long_horizons(G):
    sel = pick(G, method="Selected", site="penmanshiel", horizon_h=[18, 24, 36, 48])
    assert all(Counter(p.split(","))["LinearRegression"] == 4 for p in sel.picks)
    assert (round(-sel.gap_main_pp.max(), 2), round(-sel.gap_main_pp.min(), 2)) == (0.70, 0.90)


def test_linear_regression_penmanshiel(G):
    lr = pick(G, method="LinearRegression", site="penmanshiel", horizon_h=[18, 24, 36, 48])
    assert len(lr) == 4
    assert (round(-lr.gap_main_pp.max(), 2), round(-lr.gap_main_pp.min(), 2)) == (0.76, 0.94)
    assert (lr.gap_ci95_high_pp < 0).all()


def test_model_forecast(G):
    b = pick(G, method="Bayes_main")
    assert (round(b.gap_main_pp.min(), 2), round(b.gap_main_pp.max(), 2)) == (-0.42, 0.92)
    assert round(b[b.horizon_h <= 24].gap_main_pp.abs().max(), 1) == 0.4
    excl = b[(b.gap_ci95_testonly_low_pp > 0) | (b.gap_ci95_testonly_high_pp < 0)]
    assert list(zip(excl.site, excl.horizon_h)) == [("kelmarsh", 48)]
    r = excl.iloc[0]
    assert (round(r.gap_main_pp, 2), round(r.gap_ci95_testonly_low_pp, 2), round(r.gap_ci95_testonly_high_pp, 2)) == (0.92, 0.11, 1.65)


def test_lowest_test_error(P):
    ind = pick(P, method=INDIVIDUAL)
    best = ind.loc[ind.groupby(["site", "horizon_h"]).rmse_pct.idxmin()]
    assert len(best) == 18
    assert best.method.isin(["LinearRegression", "Bayes_main", "Bayes_sens"]).sum() == 17
    mlp = best[best.method == "MLP"]
    assert list(zip(mlp.site, mlp.horizon_h)) == [("penmanshiel", 6)]


def test_single_years(G):
    yr = G[(G.target == "normal") & G.period.isin(YEARS) & (G.method == "Selected") & (G.horizon_h == 48)]
    assert len(yr) == 10
    assert (round(yr.gap_main_pp.min(), 1), round(yr.gap_main_pp.max(), 1)) == (-2.4, 3.1)
    assert (yr[yr.period == "2020"].gap_main_pp > 0).all()


def test_seasons(G):
    def gaps(season, col):
        r = pick(G, period=f"season_{season}", method="Bayes_main", horizon_h=48).set_index("site")[col]
        return round(r["kelmarsh"], 1), round(r["penmanshiel"], 1)
    assert gaps("DJF", "gap_main_pp") == (6.7, 3.8)
    assert gaps("JJA", "gap_main_pp") == (-4.2, -2.4)
    assert gaps("DJF", "gap_seasonal_ar_pp") == (3.1, 1.3)
    assert gaps("JJA", "gap_seasonal_ar_pp")[1] == 3.6


def test_bootstrap_settings(G):
    assert C.P["bootstrap"]["resamples"] == 1000 and C.P["bootstrap"]["block_hours"] == 48
    pooled = pick(G, method="Selected")
    assert (pooled.n_resamples == 1000).all()
