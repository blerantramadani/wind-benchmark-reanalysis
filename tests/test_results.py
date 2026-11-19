import numpy as np
import pytest

from wbr import config as C
from wbr.benchmark import SEASONS
from wbr.results import NET, NETWORKS, POOLED, block_weights, point_rows, select


def test_block_weights_keep_the_sample_size_and_contiguity():
    rng = np.random.default_rng(3)
    positions = np.arange(1000, 1000 + 8760)
    w = block_weights(positions, 20_000, 48, rng)
    assert w.sum() == len(positions)
    assert (w[:1000] == 0).all() and (w[1000 + 8760:] == 0).all()


def test_hours_without_targets_get_no_weight():
    rng = np.random.default_rng(4)
    positions = np.arange(0, 1000)
    w = block_weights(positions, 1000, 48, rng)
    targets = np.arange(0, 1000, 3)            # only every third hour has a valid target
    assert w[targets].sum() < w.sum()


@pytest.mark.reproduction
def test_stored_results_reproduce():
    from wbr.check import main
    if not (C.FORECASTS.exists() and any(C.FORECASTS.glob("*_test.csv"))):
        pytest.skip("stored forecasts not present")
    assert main(["--no-networks"]) == 0     # expected/ holds the results of the reply, without the networks


def test_selection_includes_the_networks_but_never_the_main_fit():
    val = {"WindPersistenceCurve": 12.0, "LinearRegression": 11.0, "MLP": 10.5, "Bayes_main": 9.0, "Bayes_sens": 10.4}
    assert select(val) == "Bayes_sens"
    assert select({**val, "GRU": 10.3}) == "GRU"
    assert select({**val, "MLP": 10.4}) == "MLP"          # tie: the first eligible method in the list


def _synthetic_cells(rng, n=40):
    base = ("WindPersistenceCurve", "LinearRegression", "MLP", "Bayes_main", "Bayes_sens")
    cells = {}
    for y in C.TEST_YEARS:
        for h in C.HORIZONS:
            e2 = {m: rng.random(n) for m in base}
            val = {m: 10.0 + rng.random() for m in base}
            if y == NET["test_year"] and h in NET["horizons_h"]:
                e2.update({m: rng.random(n) for m in NETWORKS})
                val.update({m: 5.0 + rng.random() for m in NETWORKS})
            pick = select(val)
            e2["Selected"], val["Selected"] = e2[pick], val[pick]
            cells[(y, h)] = {"origin": np.arange(n), "e2": e2, "val": val, "pick": pick,
                             "var": {k: rng.random(n) for k in ("var_main", "var_sens", "var_seasonal_ar")},
                             "season": np.resize(np.array(SEASONS), n)}
    return cells


def test_networks_enter_the_selection_of_2019_only():
    cells = _synthetic_cells(np.random.default_rng(5))
    rows = point_rows("kelmarsh", "normal", cells)
    net = rows[rows.method.isin(NETWORKS)]
    assert set(net.period) == {str(NET["test_year"])}
    assert sorted(net.horizon_h.unique()) == sorted(NET["horizons_h"])
    sel = rows[(rows.method == "Selected") & (rows.period == POOLED)].set_index("horizon_h")
    for h in C.HORIZONS:
        first = sel.loc[h, "picks"].split(",")[0]
        assert (first in NETWORKS) == (h in NET["horizons_h"])
        e2 = np.concatenate([cells[(y, h)]["e2"]["Selected"] for y in C.TEST_YEARS])
        assert sel.loc[h, "rmse_pct"] == pytest.approx(np.sqrt(e2.mean()) * C.SCALE, rel=1e-12)
