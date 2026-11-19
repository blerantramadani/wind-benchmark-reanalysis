import numpy as np
import pandas as pd

from wbr.preprocess import hourly_from_ten_minute, seconds_overlapped


def _ten_minute(start, power, wind=None):
    t = pd.date_range(start, periods=len(power), freq="10min")
    wind = np.full(len(power), 8.0) if wind is None else wind
    return pd.DataFrame({"t": t, "wind": wind, "power": power})


def test_hour_label_is_interval_start():
    raw = _ten_minute("2020-01-01 00:00", np.arange(12, dtype=float))
    h = hourly_from_ten_minute(raw)
    assert list(h.index) == list(pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]))
    assert h["ActivePower_kW"].iloc[0] == np.mean(np.arange(6))
    assert h["ActivePower_kW"].iloc[1] == np.mean(np.arange(6, 12))


def test_negative_power_is_clipped_after_averaging():
    raw = _ten_minute("2020-01-01 00:00", [-6.0, -6.0, -6.0, 30.0, 0.0, 0.0,   # mean +2
                                           -6.0, -6.0, -6.0, -6.0, 6.0, 0.0])  # mean -3
    h = hourly_from_ten_minute(raw)
    assert h["ActivePower_kW"].iloc[0] == 2.0
    assert h["ActivePower_kW"].iloc[1] == 0.0


def test_counts_ignore_missing_values():
    power = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 5.0])
    h = hourly_from_ten_minute(_ten_minute("2020-01-01 00:00", power))
    assert h["n_power"].iloc[0] == 3
    assert h["ActivePower_kW"].iloc[0] == 3.0


def test_overlapping_stop_events_are_counted_once():
    hours = pd.date_range("2020-01-01 00:00", periods=3, freq="h")
    events = pd.DataFrame({"start": pd.to_datetime(["2020-01-01 00:30:00", "2020-01-01 00:45:00", "2020-01-01 02:59:59"]),
                           "end": pd.to_datetime(["2020-01-01 01:15:00", "2020-01-01 01:00:00", "2020-01-01 03:30:00"])})
    cover = seconds_overlapped(hours, events)
    assert list(cover) == [1800, 900, 1]


def test_point_events_are_not_extended():
    hours = pd.date_range("2020-01-01 00:00", periods=2, freq="h")
    events = pd.DataFrame({"start": pd.to_datetime(["2020-01-01 00:30"]), "end": pd.to_datetime(["2020-01-01 00:30"])})
    assert list(seconds_overlapped(hours, events)) == [0, 0]
