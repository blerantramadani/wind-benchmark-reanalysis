"""Operating states in 2018 and their effect on the benchmark (reply, point 1).

``status_logs``: events with a start and an end are matched to the hours of
the preprint's hourly files; an hour [t, t+1h) is touched by an event if they
overlap for at least one second. Events without an end are counted but not
extended in time. Reported: hours touched by stops and by IEC categories,
zero-output hours at high wind and whether a stop explains them, the hours
per retained curve bin above 10 m/s, and monthly coverage of the ten-minute
records (wind, power and nacelle temperature).

``filter_effect``: the benchmark under variants fixed before they were
computed:

  V0  the preprint's hourly file, no filter;
  V1  curve and scatter without hours that overlap a stop;
  V2  as V1, also without non-normal IEC categories;
  V3  Penmanshiel only: hourly series rebuilt with the preprint's rules but
      without requiring a temperature value; no filter;
  V4  V3 with the filter of V1.

In every variant the AR model is fitted on all pre-test hours and the curve
and scatter on the hours kept. Reported: the benchmark over pre-test states
(as Table 2), at the test origins, and for the best archived method its error
over all test targets and over targets without a stop.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import common as K
from .matched import conditional_variance

NON_NORMAL_IEC = ("Forced outage", "Scheduled Maintenance", "Technical Standby", "Requested Shutdown",
                  "Out of Electrical Specification", "Partial Performance")
IEC_COUNTED = ("Forced outage", "Scheduled Maintenance", "Technical Standby", "Requested Shutdown",
               "Out of Electrical Specification", "Out of Environmental Specification", "Full Performance")


def _public(site: str) -> pd.DataFrame:
    return pd.read_csv(K.PREPRINT / K.DATASETS[site]["file"], parse_dates=["Timestamp"]).set_index("Timestamp")


def status_logs():
    """Returns (summary per site, hourly flags per site)."""
    summary, flags_out = {}, {}
    for site in K.DATASETS:
        hr = _public(site)
        ev = K.status_events(site)
        dur = ev[ev["end"].notna()]
        res = {"events_total": int(len(ev)), "events_with_duration": int(len(dur)),
               "point_events": int(ev["end"].isna().sum()),
               "status_counts_with_duration": {k: int(v) for k, v in dur["Status"].value_counts().items()}}
        fl = {"stop": K.hours_touched(hr.index, dur[dur["Status"] == "Stop"])}
        for cat in IEC_COUNTED:
            fl[cat] = K.hours_touched(hr.index, dur[dur["IEC category"] == cat])
        fl["communication"] = K.hours_touched(hr.index, dur[dur["Status"] == "Communication"])
        fl = pd.DataFrame(fl, index=hr.index)
        train = np.arange(len(hr)) < len(hr) - K.TEST
        res["hours"] = {"pretest": int(train.sum()), "test": int((~train).sum())}
        res["hours_touched_pretest"] = {k: int((fl.loc[train, k] > 0).sum()) for k in fl.columns}
        res["hours_touched_test"] = {k: int((fl.loc[~train, k] > 0).sum()) for k in fl.columns}
        w, p = hr["WindSpeed_m_s"], hr["ActivePower_kW"]
        res["zero_output_hours"] = {}
        for name, mask in {"power<=0_wind>5": (p <= 0) & (w > 5), "power<=0_wind>10": (p <= 0) & (w > 10)}.items():
            m = mask.to_numpy()
            res["zero_output_hours"][name] = {"hours": int(m.sum()), "with_stop": int((m & (fl["stop"].to_numpy() > 0)).sum())}
        wt, pt = w[train].to_numpy(), p[train].to_numpy()
        edges = np.linspace(0, np.percentile(wt, 99.5), 41)
        which = np.digitize(wt, edges) - 1
        centers = (edges[:-1] + edges[1:]) / 2
        counts = np.bincount(which[(which >= 0) & (which < 40)], minlength=40)
        keep = (counts >= 20) & (centers > 10)
        res["hours_per_retained_bin_above_10ms"] = counts[keep].tolist()
        raw = K.ten_minute(site, ("Wind speed (m/s)", "Power (kW)", "Nacelle ambient temperature (°C)"))
        res["valid_share_by_month"] = {
            str(k.date())[:7]: {"wind": round(float(v["Wind speed (m/s)"].notna().mean()), 3),
                                "power": round(float(v["Power (kW)"].notna().mean()), 3),
                                "nacelle_temperature": round(float(v["Nacelle ambient temperature (°C)"].notna().mean()), 3)}
            for k, v in raw.resample("MS")}
        summary[site] = res
        flags_out[site] = fl.assign(wind=w, power=p, pretest=train)
    return summary, flags_out


def notable_stops() -> dict:
    """The stops behind the Kelmarsh zero-output hours of 6-9 March 2018 and the longest stop at each site."""
    out = {}
    for site in K.DATASETS:
        ev = K.status_events(site)
        st = ev[(ev["Status"] == "Stop") & ev["end"].notna()].copy()
        st["hours"] = (st["end"] - st["start"]).dt.total_seconds() / 3600
        cols = ["start", "end", "hours", "Code", "Message", "IEC category"]
        rec = lambda d: [{k: (str(v) if k in ("start", "end") else v) for k, v in r.items()} for r in d[cols].to_dict("records")]
        out[site] = {"longest_stop": rec(st.nlargest(1, "hours"))}
        if site == "kelmarsh":
            out[site]["stops_6_to_9_march"] = rec(st[(st["start"] >= "2018-03-06") & (st["start"] < "2018-03-10")])
        else:
            feb_may = st[(st["start"] >= "2018-02-01") & (st["start"] < "2018-06-01")]
            out[site]["longest_stop_february_to_may"] = rec(feb_may.nlargest(1, "hours"))
    return out


def penmanshiel_restored() -> pd.DataFrame:
    """The preprint's Penmanshiel preprocessing (hourly mean, clip, wind/power rule, interpolation of up to
    three hours) without requiring a temperature value."""
    d = K.ten_minute("penmanshiel").rename(columns={"Wind speed (m/s)": "WindSpeed_m_s", "Power (kW)": "ActivePower_kW"})
    d.index.name = "Timestamp"
    h = d.resample("1h").mean().dropna(how="all")
    h.loc[h["ActivePower_kW"] < 0, "ActivePower_kW"] = 0
    h = h[~((h["WindSpeed_m_s"] < 2) & (h["ActivePower_kW"] > 100))]
    return h.interpolate(method="linear", limit=3).dropna()


def missing_temperature_hours() -> dict:
    """Hours with wind and power in the rebuilt Penmanshiel series that the preprint's file lacks."""
    pub = _public("penmanshiel")
    res = penmanshiel_restored()
    res = res[res.index.year == 2018]
    gone = res[~res.index.isin(pub.index)]
    stop = K.stop_flags("penmanshiel", gone.index)
    return {"rebuilt_hours": int(len(res)), "public_hours": int(len(pub)), "hours_missing_from_public": int(len(gone)),
            "share_of_year": float(len(gone) / 8760), "with_positive_power": int((gone["ActivePower_kW"] > 0).sum()),
            "overlapping_a_stop": int(stop.sum()),
            "months": sorted({str(t)[:7] for t in gone.index})}


def _evaluate(site, data: pd.DataFrame, exclude: np.ndarray, label: str, rated: float, cases: dict) -> list:
    B07 = K.b07()
    train = data.iloc[:-K.TEST]
    keep = ~exclude[:len(train)]
    idx = pd.DatetimeIndex(train.index)
    v = train["WindSpeed_m_s"].to_numpy(float)
    order, (mu, phi, sigma2, _), _, _ = B07.choose_ar_order(idx, v, 5)
    m, s2 = B07.empirical_curve(v[keep], train["ActivePower_kW"].to_numpy(float)[keep], rated, 40, 20)
    f = {"order": order, "mu": mu, "phi": phi, "sigma2": sigma2, "m": m, "s2": s2}
    ids = B07.contiguous_windows(idx, order)
    pre_hist = np.column_stack([v[ids - k] for k in range(order)])
    wind = data["WindSpeed_m_s"]
    rows = []
    for h, (origin_last, target_times, actual, best_name, best_pred, stop_target) in cases.items():
        hist = np.column_stack([wind.reindex(origin_last - pd.Timedelta(hours=k)).to_numpy(float) for k in range(5)])
        assert not np.isnan(hist).any()
        var_test = conditional_variance(f, hist, h)
        normal = ~stop_target
        b_all, b_norm = (math.sqrt(var_test.mean()) * 100 / rated, math.sqrt(var_test[normal].mean()) * 100 / rated)
        r_all, r_norm = K.nrmse(actual, best_pred, rated), K.nrmse(actual[normal], best_pred[normal], rated)
        rows.append({"site": site, "variant": label, "horizon_h": h, "ar_order": order,
                     "pretest_hours": len(train), "curve_hours": int(keep.sum()),
                     "bench_pretest_pct": math.sqrt(conditional_variance(f, pre_hist, h).mean()) * 100 / rated,
                     "bench_test_all_pct": b_all, "bench_test_normal_pct": b_norm, "best_method": best_name,
                     "rmse_all_pct": r_all, "rmse_normal_pct": r_norm,
                     "gap_all_pp": r_all - b_all, "gap_normal_pp": r_norm - b_norm,
                     "test_targets_normal": int(normal.sum())})
    return rows


def filter_effect(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site, meta in K.DATASETS.items():
        rated = meta["rated"]
        df, pub = K.site_data(site), _public(site)[["WindSpeed_m_s", "ActivePower_kW"]]
        ev = K.status_events(site)
        dur = ev[ev["end"].notna()]

        def flags(index):
            stop = K.hours_touched(index, dur[dur["Status"] == "Stop"]) > 0
            nn = K.hours_touched(index, dur[(dur["Status"] == "Stop") | dur["IEC category"].isin(NON_NORMAL_IEC)]) > 0
            return stop, nn

        stop_pub, nn_pub = flags(pub.index)
        stop_series = pd.Series(stop_pub, index=pub.index)
        cases = {}
        for h in K.HORIZONS:
            te = K.valid_origins(df.index, h)[-K.TEST:]
            pred = K.archived_forecasts(site, h)
            b = best[(best.site == site) & (best.horizon_h == h)].iloc[0]
            targets = df.index[te + h - 1]
            cases[h] = (df.index[te - 1], targets, pred["actual"].to_numpy(float), b.method,
                        pred[K.forecast_column(b.method)].to_numpy(float), stop_series.reindex(targets).to_numpy(bool))
        rows += _evaluate(site, pub, np.zeros(len(pub), bool), "V0_published", rated, cases)
        rows += _evaluate(site, pub, stop_pub, "V1_without_stops", rated, cases)
        rows += _evaluate(site, pub, nn_pub, "V2_without_stops_and_iec", rated, cases)
        if site == "penmanshiel":
            res = penmanshiel_restored()
            assert res.index[-K.TEST:].equals(pub.index[-K.TEST:]), "the test hours must not change"
            stop_res, _ = flags(res.index)
            rows += _evaluate(site, res, np.zeros(len(res), bool), "V3_rebuilt", rated, cases)
            rows += _evaluate(site, res, stop_res, "V4_rebuilt_without_stops", rated, cases)
    return pd.DataFrame(rows)
