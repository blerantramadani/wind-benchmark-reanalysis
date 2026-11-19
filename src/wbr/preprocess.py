"""Hourly series and stop flags from the raw ten-minute SCADA archives.

Rules (protocol, Sect. 4-5):

1. Only time stamp, ``Wind speed (m/s)`` and ``Power (kW)`` are read. Time
   stamps are UTC. Temperature is not read, so a missing temperature never
   removes an hour.
2. The hourly value labelled t is the mean of the ten-minute records labelled
   t, t+10, ..., t+50 min.
3. A negative hourly power mean is set to zero (clipping after averaging).
4. An hour is valid if wind and power each have at least four of six
   ten-minute values, and it is not of the form wind < 2 m/s with
   power > 100 kW.
5. Nothing is interpolated; invalid hours remain gaps.
6. An hour is flagged as a stop if it overlaps, for at least one second, a
   status-log event with ``Status == "Stop"`` that has both a start and an end.
   Point events (no end) are not extended in time.

Archives are read in place (zip members are streamed, never extracted).
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

WIND, POWER, TIME = "Wind speed (m/s)", "Power (kW)", "# Date and time"


def archives(site: str, year: int) -> list[Path]:
    spec = C.P["sites"][site]
    found = sorted(p for p in C.RAW.glob(f"{spec['archive_prefix']}{year}_*.zip")
                   if spec.get("archive_must_contain", "") in p.name)
    if not found:
        raise FileNotFoundError(f"no raw archive for {site} {year} in {C.RAW}; run `wbr download` first")
    return found


def _members(archive: Path, site: str):
    tag = C.P["sites"][site]["turbine_tag"]
    names = zipfile.ZipFile(archive).namelist()
    base = {n.rsplit("/", 1)[-1]: n for n in names}
    turbine = [n for b, n in base.items() if b.startswith(f"Turbine_Data_{tag}")]
    status = [n for b, n in base.items() if b.startswith(f"Status_{tag}")]
    if len(turbine) != 1 or len(status) != 1:
        raise RuntimeError(f"{archive.name}: expected one turbine and one status file for {tag}")
    return turbine[0], status[0]


def _read_table(archive: Path, member: str, header_start: str, usecols=None) -> pd.DataFrame:
    with zipfile.ZipFile(archive) as z:
        text = z.read(member).decode("utf-8")
    lines = text.splitlines()
    k = next(i for i, line in enumerate(lines[:40]) if line.startswith(header_start))
    return pd.read_csv(io.StringIO("\n".join(lines[k:])), usecols=usecols, low_memory=False)


def hourly_from_ten_minute(raw: pd.DataFrame) -> pd.DataFrame:
    """Rules 2-3: hourly means over [t, t+1h), value counts, negative means clipped."""
    g = raw.set_index("t").sort_index().resample("h", label="left", closed="left")
    return pd.DataFrame({"WindSpeed_m_s": g["wind"].mean(),
                         "ActivePower_kW": g["power"].mean().clip(lower=0),
                         "n_wind": g["wind"].count(), "n_power": g["power"].count()})


def stop_events(archive: Path, member: str) -> pd.DataFrame:
    s = _read_table(archive, member, "Timestamp start").drop_duplicates()
    s["start"] = pd.to_datetime(s["Timestamp start"])
    s["end"] = pd.to_datetime(s["Timestamp end"], errors="coerce")
    return s[(s["Status"] == "Stop") & s["end"].notna()][["start", "end", "Code", "Message", "IEC category"]]


def seconds_overlapped(hours: pd.DatetimeIndex, events: pd.DataFrame) -> np.ndarray:
    """Seconds of each hour [t, t+1h) covered by the union of the event intervals."""
    h0 = hours.values.astype("datetime64[s]").astype(np.int64)
    h1 = h0 + 3600
    intervals = sorted(zip(events["start"].values.astype("datetime64[s]").astype(np.int64),
                           events["end"].values.astype("datetime64[s]").astype(np.int64)))
    merged: list[list[int]] = []
    for s, e in intervals:
        if e <= s:
            continue
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    cover = np.zeros(len(hours))
    for s, e in merged:
        for i in range(np.searchsorted(h1, s, side="right"), np.searchsorted(h0, e, side="left")):
            cover[i] += max(0, min(h1[i], e) - max(h0[i], s))
    return cover


def build_site(site: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_n = C.P["min_values_per_hour"]
    low_wind, high_power = C.P["invalid_if_wind_below_ms_and_power_above_kw"]
    frames, stops = [], []
    for year in C.YEARS:
        for archive in archives(site, year):
            turbine, status = _members(archive, site)
            raw = _read_table(archive, turbine, TIME, usecols=[TIME, WIND, POWER])
            raw.columns = ["t", "wind", "power"]
            raw["t"] = pd.to_datetime(raw["t"])
            frames.append(hourly_from_ten_minute(raw))
            stops.append(stop_events(archive, status))
    h = pd.concat(frames).sort_index()
    h = h[~h.index.duplicated(keep="first")]
    h = h[(h.index >= f"{C.YEARS[0]}-01-01") & (h.index < f"{C.YEARS[-1] + 1}-01-01")]
    ev = pd.concat(stops).drop_duplicates().sort_values("start")
    suspicious = (h["WindSpeed_m_s"] < low_wind) & (h["ActivePower_kW"] > high_power)
    h["valid"] = (h["n_wind"] >= min_n) & (h["n_power"] >= min_n) & ~suspicious
    h["stop"] = seconds_overlapped(h.index, ev) > 0
    h.index.name = "Timestamp"
    return h, ev


def main():
    C.ensure_dirs()
    summary = {}
    for site in C.SITES:
        h, ev = build_site(site)
        h.to_csv(C.PROCESSED / f"{site}_hourly_{C.YEARS[0]}_{C.YEARS[-1]}.csv")
        ev.to_csv(C.PROCESSED / f"{site}_stop_events.csv", index=False)
        per_year = {}
        for y, g in h.groupby(h.index.year):
            per_year[int(y)] = {"first_hour": str(g.index.min()), "valid_hours": int(g["valid"].sum()),
                                "valid_share_of_calendar_year": round(float(g["valid"].sum()) / (8784 if y % 4 == 0 else 8760), 3),
                                "stop_hours_among_valid": int((g["valid"] & g["stop"]).sum())}
        summary[site] = {"stop_events": int(len(ev)), "per_year": per_year}
        print(f"{site}: {int(h['valid'].sum())} valid hours", flush=True)
    (C.PROCESSED / "availability.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
