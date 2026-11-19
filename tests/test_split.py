import numpy as np
import pandas as pd
import pytest

from wbr import config as C
from wbr.split import Grid, candidate_origins, split


@pytest.fixture(scope="module")
def grid():
    time = pd.date_range(f"{C.YEARS[0]}-01-01", f"{C.YEARS[-1]}-12-31 23:00", freq="h")
    rng = np.random.default_rng(0)
    valid = np.ones(len(time), bool)
    valid[(time >= "2017-03-01") & (time < "2017-03-05")] = False   # a gap inside the training period
    stop = rng.random(len(time)) < 0.02
    return Grid("synthetic", time, rng.gamma(4, 1.5, len(time)), rng.uniform(0, 2050, len(time)), valid, stop)


@pytest.mark.parametrize("h", [1, 6, 48])
@pytest.mark.parametrize("target", ["normal", "all_states"])
def test_blocks_are_chronological_and_purged(grid, h, target):
    s = split(grid, 2019, h, target)
    t = grid.time
    assert (t[s.test + h - 1].year == 2019).all()
    assert (t[s.val + h - 1].year == 2018).all()
    assert (t[s.train + h - 1].year <= 2017).all()
    first_val_target, first_test_target = s.val[0] + h - 1, s.test[0] + h - 1
    # every training target is known before the first validation forecast is issued, and so on
    assert (s.train + h - 1).max() <= first_val_target - h
    assert (s.val + h - 1).max() <= first_test_target - h
    assert s.bench_cut_main == first_test_target - h
    assert s.train_last_target == (s.train + h - 1).max()


def test_purge_removes_h_minus_1_samples_on_a_contiguous_boundary(grid):
    h = 24
    o = candidate_origins(grid, h, "all_states")
    ty = grid.time[o + h - 1].year
    s = split(grid, 2020, h, "all_states")
    assert (ty == 2019).sum() - len(s.val) == h - 1


def test_inputs_valid_and_normal_target_excludes_stops(grid):
    h = 6
    o = candidate_origins(grid, h, "normal")
    assert all(grid.valid[i - C.LOOK_BACK:i].all() for i in o[:2000])
    assert not grid.stop[o + h - 1].any()
    assert grid.valid[o + h - 1].all()
