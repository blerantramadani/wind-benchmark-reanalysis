# Data provenance

The processed hourly CSV files in this directory were derived from the public
ten-minute SCADA releases below:

- Kelmarsh Wind Farm Data — C. Plumley, Zenodo (2022), DOI:
  [10.5281/zenodo.5841834](https://doi.org/10.5281/zenodo.5841834)
- Penmanshiel Wind Farm Data — C. Plumley, Zenodo (2022), DOI:
  [10.5281/zenodo.5946808](https://doi.org/10.5281/zenodo.5946808)

The source releases are distributed under CC BY 4.0. Retain the source
attribution and comply with the original licence when reusing the derived
files.

Each hourly file contains:

```text
Timestamp
WindSpeed_m_s
Temperature_C
ActivePower_kW
```

The reported benchmark and empirical evaluation use wind speed and active
power. Temperature is retained in the processed files but is not an input to
the reported wind-only forecasts. The scripts apply chronological splits and
discard forecast windows that cross a missing hourly timestamp.
