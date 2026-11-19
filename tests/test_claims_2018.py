"""The 2018 numbers of the reply to RC1 (points 1, 2, 3a, 3b(ii), 4), checked against expected/y2018 (see CLAIMS.md).

Values are compared after rounding to the precision used in the reply.
"""
import json
import math

import pandas as pd
import pytest

from wbr import config as C

E = C.ROOT / "expected" / "y2018"
pytestmark = pytest.mark.skipif(not (E / "filter_effect.csv").exists(), reason="expected/y2018 not present")
SITES = ("kelmarsh", "penmanshiel")


def csv(name):
    return pd.read_csv(E / name)


def js(name):
    return json.loads((E / name).read_text())


# --- introduction and point 3b(ii): reproduction --------------------------------------------------

def test_published_scores_reproduced():
    s = js("reproduction_summary.json")
    assert s["rows"] == 126 and s["unmatched_rows"] == 0
    assert s["max_abs_diff_pp"] < 1.4e-7 and round(s["max_abs_diff_pp"] * 1e7, 1) == 1.4


# --- point 1 ---------------------------------------------------------------------------------------

def test_kelmarsh_march_stops():
    st = js("status_logs.json")["notable_stops"]["kelmarsh"]["stops_6_to_9_march"]
    first, second = st[0], st[1]
    assert first["start"].startswith("2018-03-06 13:32") and first["IEC category"] == "Scheduled Maintenance"
    assert second["Message"] == "Battery test" and second["IEC category"] == "Technical Standby"
    assert round(second["hours"], 1) == 70.1
    end = pd.Timestamp(second["end"])
    assert end.round("min") == pd.Timestamp("2018-03-09 12:53")


def test_zero_output_hours_explained_by_stops():
    logs = js("status_logs.json")
    for site, n in (("kelmarsh", 62), ("penmanshiel", 19)):
        z = logs[site]["zero_output_hours"]["power<=0_wind>5"]
        assert z == {"hours": n, "with_stop": n}


def test_share_of_pretest_hours_touched_by_stops():
    logs = js("status_logs.json")
    share = {s: logs[s]["hours_touched_pretest"]["stop"] / logs[s]["hours"]["pretest"] for s in SITES}
    assert (round(100 * share["kelmarsh"], 1), round(100 * share["penmanshiel"], 1)) == (2.8, 2.0)


def test_referee_bin_counts_match_penmanshiel():
    b = js("status_logs.json")["penmanshiel"]["hours_per_retained_bin_above_10ms"]
    assert (min(b), max(b)) == (20, 146)


def _filter(variant, site, h, col="bench_pretest_pct"):
    f = csv("filter_effect.csv")
    return f[(f.variant == variant) & (f.site == site) & (f.horizon_h == h)][col].iloc[0]


def test_benchmark_without_stops():
    assert (round(_filter("V0_published", "kelmarsh", 1), 2), round(_filter("V1_without_stops", "kelmarsh", 1), 2)) == (10.65, 9.68)
    assert (round(_filter("V0_published", "penmanshiel", 1), 2), round(_filter("V1_without_stops", "penmanshiel", 1), 2)) == (11.61, 11.21)
    f = csv("filter_effect.csv").pivot_table(index=["site", "horizon_h"], columns="variant", values="bench_pretest_pct")
    long = f[f.index.get_level_values("horizon_h") >= 12]
    assert (long["V1_without_stops"] - long["V0_published"]).abs().max() < 0.2


def test_penmanshiel_temperature_gap():
    m = js("status_logs.json")["penmanshiel_missing_temperature"]
    assert m["hours_missing_from_public"] == 2317 and round(100 * m["share_of_year"]) == 26
    assert m["with_positive_power"] == 1910 and m["overlapping_a_stop"] == 177
    assert m["months"] == ["2018-02", "2018-03", "2018-04", "2018-05"]
    temp = js("status_logs.json")["penmanshiel"]["valid_share_by_month"]
    assert sum(temp[f"2018-0{k}"]["nacelle_temperature"] for k in (2, 3, 4, 5)) / 4 < 0.5


def test_penmanshiel_rebuilt_series_and_icing_stop():
    assert round(_filter("V3_rebuilt", "penmanshiel", 1), 2) == 14.25
    assert round(_filter("V4_rebuilt_without_stops", "penmanshiel", 1), 2) == 11.01
    icing = js("status_logs.json")["notable_stops"]["penmanshiel"]["longest_stop_february_to_may"][0]
    assert icing["Message"] == "Icing (stop)" and round(icing["hours"]) == 134


def test_penmanshiel_has_no_2024_archive_for_turbine_01():
    files = pd.read_csv(C.ROOT / "data" / "zenodo_files.tsv", sep="\t")["file"]
    assert not files.str.contains("Penmanshiel_SCADA_2024").any()


# --- points 2 and 3a -------------------------------------------------------------------------------

def test_figure1_curvature_depends_on_grid():
    g = csv("figure1_grid.csv").set_index(["site", "grid_points"])
    k = g.loc["kelmarsh"]
    assert (round(k.loc[400, "curvature_kw_per_ms2"]), round(k.loc[800, "curvature_kw_per_ms2"])) == (725, 424)
    assert (k.loc[[1600, 2000, 3200], "curvature_kw_per_ms2"].abs() < 1e-6).all()
    d = k.loc[[400, 800, 1600], "distance_to_nearest_knot_ms"]
    assert round(d.min(), 2) >= 0.02 and round(d.max(), 2) <= 0.03
    assert (g.loc["penmanshiel", "curvature_kw_per_ms2"].abs() < 1e-6).all()


def test_bin_width():
    g = csv("figure1_grid.csv").groupby("site")["knot_spacing_ms"].first()
    assert (round(g["kelmarsh"], 2), round(g["penmanshiel"], 2)) == (0.36, 0.51)


def test_taylor_units():
    u = csv("taylor_units.csv")
    h48 = u[u.horizon_h == 48]
    assert (round(h48.sd_error_pct.min()), round(h48.sd_error_pct.max())) == (22, 61)
    v = h48.groupby("dataset")["variance_error_pct"].agg(["min", "max"]).round()
    assert tuple(v.loc["kelmarsh"]) == (48, 70) and tuple(v.loc["penmanshiel"]) == (109, 158)


def test_smooth_curves():
    t = csv("taylor_smooth_curves.csv")
    assert t.quad_change_128_256_pct.abs().max() < 0.007
    w = t[t.data == "without_stops"]

    def rng(site, h, col, nd):
        x = w[(w.site == site) & (w.horizon_h == h)][col]
        return round(x.min(), nd), round(x.max(), nd)
    assert rng("kelmarsh", 1, "taylor2_sd_error_pct", 1) == (3.3, 3.4)
    assert rng("kelmarsh", 48, "taylor2_sd_error_pct", 0) == (31, 38)
    assert rng("kelmarsh", 48, "taylor2_var_error_pct", 0) == (73, 90)
    assert rng("penmanshiel", 1, "taylor2_sd_error_pct", 1) == (5.4, 5.5)
    assert rng("penmanshiel", 48, "taylor2_sd_error_pct", 0) == (61, 77)
    assert rng("penmanshiel", 48, "taylor2_var_error_pct", 0) == (161, 213)
    for _, g in t.groupby(["site", "data", "family"]):
        assert g.sort_values("horizon_h").taylor2_sd_error_pct.is_monotonic_increasing


# --- point 3b(ii) ----------------------------------------------------------------------------------

def test_matched_benchmark_against_pretest():
    a = csv("matched_all_methods.csv").drop_duplicates(["site", "horizon_h"])
    d = a.bench_test_public_fit_pct - a.bench_pretest_published_pct
    assert (d[a.site == "kelmarsh"] > 0).all()
    k1 = a[(a.site == "kelmarsh") & (a.horizon_h == 1)].iloc[0]
    assert (round(k1.bench_test_public_fit_pct, 2), round(k1.bench_pretest_published_pct, 2)) == (11.61, 10.65)
    p = d[a.site == "penmanshiel"]
    assert (round(p.min(), 2), round(p.max(), 2)) == (-0.12, 0.20)


def test_matched_gaps_of_best_method():
    b = csv("matched_best_method.csv")
    assert len(b) == 18
    assert (round(b.gap_test_public_fit_pp.min(), 2), round(b.gap_test_public_fit_pp.max(), 2)) == (-0.07, 4.17)
    assert (round(b.gap_pretest_published_pp.min(), 2), round(b.gap_pretest_published_pp.max(), 2)) == (0.89, 4.52)
    assert b[(b.site == "kelmarsh") & (b.horizon_h == 1)].gap_test_public_fit_pp.iloc[0] < 0
    test_excl = ((b.gap_ci95_test_low_pp > 0) | (b.gap_ci95_test_high_pp < 0)).sum()
    fit_excl = ((b.gap_ci95_fit_low_pp > 0) | (b.gap_ci95_fit_high_pp < 0)).sum()
    assert (test_excl, fit_excl) == (1, 0)
    assert round(1000 / 48) == 21


def test_errors_autocorrelated_over_about_a_day_at_48h():
    # The statement holds at long horizons, where the intervals are wide; at h = 1 the autocorrelation
    # is 0.05-0.50 at a lag of one hour and near zero after six hours (see CLAIMS.md).
    acf = js("matched_summary.json")["autocorrelation_of_squared_error_minus_variance"]
    for site in SITES:
        v = acf[f"{site}_h48"]
        assert v["1"] > 0.8 and v["6"] > 0.3 and v["24"] < 0.2 and abs(v["48"]) < 0.1


def test_calibration_2018():
    c = csv("calibration.csv").set_index(["site", "target", "horizon_h"])
    k = c.loc[("kelmarsh", "all_states", 48)]
    assert (round(k.rmse_model_forecast_pct, 2), round(k.bench_test_pct, 2)) == (32.84, 28.51)
    assert round(k.mean_error_model_forecast_kw) == 149
    p = c.loc[("penmanshiel", "all_states", 48)]
    assert (round(p.rmse_model_forecast_pct, 2), round(p.bench_test_pct, 2)) == (35.35, 33.12)
    best = csv("matched_best_method.csv").set_index(["site", "horizon_h"])
    assert round(best.loc[("kelmarsh", 48), "rmse_pct"], 2) == 32.68


def test_test_weeks_windier_than_pretest_mean():
    w = js("reproduction_summary.json")["test_window_wind"]
    got = {s: (round(w[s]["pretest_mean_ms"], 2), round(w[s]["test_mean_ms"], 2)) for s in SITES}
    assert got == {"kelmarsh": (5.99, 6.78), "penmanshiel": (7.23, 8.06)}


# --- point 4 ---------------------------------------------------------------------------------------

def test_hour_convention():
    hc = js("reproduction_summary.json")["hour_convention"]
    for s in SITES:
        assert hc[s]["start_of_hour"]["wind_share_within_1e-6"] == 1.0
        assert hc[s]["end_of_hour"]["wind_share_within_1e-6"] < 0.01


def test_boundaries_for_h_above_1():
    s = csv("reproduction_splits.csv")
    assert (s.train_targets_after_first_val_issue == s.horizon_h - 1).all()
    assert (s.val_targets_after_first_test_issue == s.horizon_h - 1).all()
    assert (s.benchmark_fit_hours_after_first_test_issue == s.horizon_h - 1).all()
    p = js("purged_splits_summary.json")
    assert p["test_targets_in_train_total"] == 0 and p["test_targets_in_val_total"] == 0
    assert p["removed_equals_h_minus_1"]


def test_purging_effect():
    p = js("purged_splits_summary.json")
    r = lambda k: tuple(round(x, 2) for x in p[k])
    assert r("wpc_change_pp_range") == (-0.15, 0.17)
    assert r("lr_change_pp_range") == (-0.02, 0.11)
    assert r("bench_change_pp_range") == (-0.02, 0.05)


def test_retrained_networks_at_48h():
    t = csv("retrained_h48_scores.csv")
    assert round(t.rerun_minus_published_pp.abs().max(), 2) == 0.13
    k, p = t[t.site == "kelmarsh"].purging_change_pp, t[t.site == "penmanshiel"].purging_change_pp
    assert (round(k.min(), 2), round(k.max(), 2)) == (0.33, 0.60)
    assert (round(p.min(), 2), round(p.max(), 2)) == (-0.07, 0.56)
