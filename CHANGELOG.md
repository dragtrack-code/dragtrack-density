# Changelog

Changes to the `dragtrack` package.  Newest first.  The data products have
their own schema version (`meta.schema_version`); see the README.

## 0.7.0 (2026-10-30)

- Plots of the density products, with the optional extra
  `pip install dragtrack[plots]` (plotly, kaleido).  The core package still
  needs only numpy.
  - `dragtrack.plots.atmosphere_figure(rho_realtime, rho_forecast=None, alt_km=400)`:
    density at one altitude vs time (solver values with their uncertainty,
    the realtime fit and its scenarios, the forecast if given), the solver's
    data-cadence strip, the latitude and local-time panels, and the S and G
    drivers.  Inputs may be dicts, paths or URLs.
  - `splice_forecast(rho_realtime, rho_forecast)` and `save_png(fig, path)`.
  - Command line: `python -m dragtrack plot --rho-realtime F [--rho-forecast F] [--alt 400] -o OUT.png`.

## 0.6.16 (2026-10-04)

- README: the seasonal reference of `rho_forecast` (data schema 0.10.7).

## 0.6.15 (2026-10-03)

- README: the `rho_forecast` calibration as it is now (seasonal factor,
  Ap raise; data schema 0.10.6).

## 0.6.12 – 0.6.14 (2026-09-30)

- README: `rho_forecast` calibration (data schemas 0.10.2 – 0.10.3).

## 0.6.11 (2026-09-30)

- README: `leo_sats_history`; data-schema vs package versions;
  interval-matched `leo_sats` density.

## 0.6.10 (2026-09-28)

- `DragTrackDensity` loads `rho_forecast` (the SWPC-driven forecast to
  about 45 days ahead).

## 0.6.6 – 0.6.9 (2026-09-27 – 09-28)

- `load_sats` for the per-satellite drag data (`leo_sats`); `stable` and
  `rmspe` fields (`stable_sats` merged into `leo_sats`).
- README walks through all the data products; fixes from a code review.

## 0.6.3 – 0.6.5 (2026-09-27)

- Density at geodetic height; the J70 `Tx` relation and LUT units; apparent
  local solar time (as the pipeline); UTC time parsing fix; tests in CI.

## 0.6.0 – 0.6.2 (2026-09-26)

- Repository cleanup; per-source coverage ends; Space-Track citation in the
  README.

## 0.3.0 – 0.5.0 (2026-09-22 – 09-26)

- Coverage from 2014 with the `rho_full` history; density for a state vector
  (`rho_from_state`); `Tinf` renamed `TJ70`.
