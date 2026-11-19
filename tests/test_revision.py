"""Results for the revision, checked against expected/revision and expected/y2018.

expected/revision holds the multi-year results with the sequence networks of 2019
as selection candidates (PROTOCOL.md, 7 Nov 2026) and the block-length
sensitivity of the intervals; expected/ holds the results of the reply to RC1.
The numbers are listed in CLAIMS.md under "Revision".
"""
import pandas as pd
import pytest

from wbr import config as C

E, R = C.ROOT / "expected", C.ROOT / "expected" / "revision"
KEYS = ["site", "target", "period", "horizon_h", "method"]
NETWORKS = ["LSTM", "GRU", "DLinear", "PatchTST"]
pytestmark = pytest.mark.skipif(not (R / "gaps_with_intervals_L48.csv").exists(), reason="expected/revision not present")


def read(path):
    return pd.read_csv(path, float_precision="round_trip").astype({"period": str})


def pooled_selected(df):
    return df[(df.target == "normal") & (df.period == "2019-2023") & (df.method == "Selected")].set_index(["site", "horizon_h"])


def test_network_rows_only_in_2019_normal_at_six_horizons():
    rev = read(R / "point_estimates.csv")
    net = rev[rev.method.isin(NETWORKS)]
    assert len(net) == 2 * 6 * 4
    assert set(net.period) == {"2019"} and set(net.target) == {"normal"}
    assert sorted(set(net.horizon_h)) == [1, 3, 6, 12, 24, 48]


def test_only_selected_rows_at_penmanshiel_3_and_6_h_change():
    old, new = read(E / "gaps_with_intervals_L48.csv"), read(R / "gaps_with_intervals_L48.csv")
    m = old.merge(new, on=KEYS, suffixes=("_old", "_new"))
    assert len(m) == len(old) == len(new) - 48
    cols = [c for c in old.columns if c not in KEYS]
    differs = pd.Series(False, index=m.index)
    for c in cols:
        a, b = m[f"{c}_old"], m[f"{c}_new"]
        differs |= ~((a == b) | (a.isna() & b.isna()))
    changed = m[differs]
    assert set(changed.method) == {"Selected"}
    assert set(changed.site) == {"penmanshiel"} and set(changed.target) == {"normal"}
    assert sorted(set(changed.horizon_h)) == [3, 6]
    assert len(changed) == 12      # 2019, pooled and four seasons, at 3 and 6 h


def test_gru_selected_at_penmanshiel_3_and_6_h_in_2019():
    rev = read(R / "point_estimates.csv")
    sel = rev[(rev.target == "normal") & (rev.period == "2019") & (rev.method == "Selected")].set_index(["site", "horizon_h"])
    assert sel.loc[("penmanshiel", 3), "picks"] == "GRU" and sel.loc[("penmanshiel", 6), "picks"] == "GRU"
    for h in (1, 3, 6, 12, 24, 48):
        assert sel.loc[("kelmarsh", h), "picks"] not in NETWORKS


def test_penmanshiel_listed_gaps():
    s = pooled_selected(read(R / "gaps_with_intervals_L48.csv"))
    r = s.loc[("penmanshiel", 6)]
    assert (round(r.gap_main_pp, 2), round(r.gap_ci95_low_pp, 2), round(r.gap_ci95_high_pp, 2)) == (-0.23, -0.74, 0.30)
    r = s.loc[("penmanshiel", 3)]
    assert (round(r.gap_main_pp, 2), round(r.gap_ci95_low_pp, 2), round(r.gap_ci95_high_pp, 2)) == (-0.13, -0.55, 0.27)


def test_statements_of_the_reply_still_hold():
    s = pooled_selected(read(R / "gaps_with_intervals_L48.csv"))
    assert len(s) == 18
    assert (round(s.gap_main_pp.min(), 2), round(s.gap_main_pp.max(), 2)) == (-0.90, 0.31)
    assert not (s.gap_ci95_low_pp > 0).any()
    assert sorted(s[s.gap_ci95_high_pp < 0].index.tolist()) == [("penmanshiel", 18), ("penmanshiel", 24), ("penmanshiel", 36)]


def test_pooled_change_from_the_reply():
    old, new = pooled_selected(read(E / "gaps_with_intervals_L48.csv")), pooled_selected(read(R / "gaps_with_intervals_L48.csv"))
    change = (new.gap_main_pp - old.gap_main_pp).abs()
    assert round(change.max(), 2) == 0.03


@pytest.mark.parametrize("block, below", [(24, 4), (48, 3), (96, 3)])
def test_block_lengths(block, below):
    s = pooled_selected(read(R / f"gaps_with_intervals_L{block}.csv"))
    assert not (s.gap_ci95_low_pp > 0).any()
    assert int((s.gap_ci95_high_pp < 0).sum()) == below


def test_retrained_networks_2018():
    t = pd.read_csv(E / "y2018" / "retrained_scores.csv", float_precision="round_trip")
    assert len(t) == 2 * 8 * 5
    stable = t[t.model != "PatchTST"]
    assert stable.rerun_minus_published_pp.abs().max() < 0.19
    short = stable[stable.horizon_h <= 12]
    assert short.purging_change_pp.abs().max() < 0.17
    mid = stable[stable.horizon_h.between(2, 36)]
    assert (round(mid.purging_change_pp.min(), 2), round(mid.purging_change_pp.max(), 2)) == (-0.42, 0.33)
    h48 = t[t.horizon_h == 48].set_index(["site", "model"])
    assert round(h48.loc["kelmarsh", "purging_change_pp"].min(), 2) == 0.33
    assert round(h48.loc["kelmarsh", "purging_change_pp"].max(), 2) == 0.60
