# dragtrack

Thermospheric mass density and satellite drag data from the DragTrack JSON
products, and a small Python package (`dragtrack`) to use them.

```python
from dragtrack import DragTrackDensity, load_sats

BASE = 'https://huggingface.co/datasets/djgersh/dragtrack-density-data/resolve/main/latest'
m = DragTrackDensity.from_url(f'{BASE}/rho_realtime.json', f'{BASE}/rho_full.json',
                              rho_forecast_url=f'{BASE}/rho_forecast.json')
rho = m.rho('2026-09-13T12:00:00Z', lat_deg=45, lon_deg=0, alt_km=450)   # kg/m^3

sats = load_sats(f'{BASE}/leo_sats.json')
bc = sats[25544]['bc_inv_m2_per_kg']       # ISS, daily Cd·A/m (m^2/kg)
```

The DragTrack products are derived from USSPACECOM data obtained via
[Space-Track.org](https://www.space-track.org); please cite it when
publishing results based on them.  Space-weather inputs come from NOAA
SWPC/NCEI, NASA SPDF (OMNI2) and GFZ Potsdam.

## Install

```bash
pip install https://huggingface.co/datasets/djgersh/dragtrack-density-data/resolve/main/latest/dragtrack-0.6.10-py3-none-any.whl
```

Python >= 3.9, numpy >= 1.21.  Not on PyPI.

## The data products

Four JSON files in `latest/` on the
[Hugging Face dataset](https://huggingface.co/datasets/djgersh/dragtrack-density-data),
regenerated together after each solver update and sharing one
`meta.run_id`.  Earlier versions are kept in dated directories, listed in
`manifest.json`.

| File | Contents |
|---|---|
| `rho_realtime.json` | Global density: recent solver values, a fit to them, and forecast scenarios |
| `rho_full.json` | Global density: every solver value since 2014 |
| `rho_forecast.json` | Global density: the `rho_realtime` fit continued to about 45 days ahead with the NOAA SWPC forecasts |
| `leo_sats.json` | Daily drag data for every LEO object with measurable drag, flagging the satellites the density solver uses |

Each file defines its fields, units and conventions in its `meta` block.
Times are UTC.

### rho_realtime and rho_full: density

* `altitude_km`: the altitude grid, geodetic height above the WGS-84
  ellipsoid.
* `solver`: the TLE-based solution, one row per time: `rho` (kg/m^3,
  globe-average density at each altitude), `uncertainty` (relative
  1-sigma), `TJ70_K` and `J70_scale` (the Jacchia-1970 temperature and
  scale that give `rho` at any height), `source` and `bin_days` (the
  averaging behind each row).
* `spatial`: 16 spherical-harmonic coefficients per row (`TJ70_lm`)
  describing how the temperature, and so the density, varies with
  latitude and local solar time.
* `fit` (`rho_realtime` only): hourly rows of a regression
  `rho_400 = a·S + b·G(t − t0) + c` on the recent solver values, driven by
  solar EUV (`S`) and geomagnetic activity (`G`).  `rho_fit` uses the
  measured drivers and extends past the last solver value;
  `rho_<scenario>` holds G at fixed levels.

`rho_realtime` covers the recent window (`meta.window_days`) plus the
forecast; `rho_full` covers 2014 to the last solver value.  Use both for a
continuous record.

### rho_forecast: SWPC-driven forecast

The `rho_realtime` fit continued past its last `rho_fit` value to the
horizon of the NOAA SWPC solar and geomagnetic forecasts (about 45 days),
with `S` and `G` derived from those forecasts.  It has only a `fit` block
and `spatial.fit`, with the same fields as `rho_realtime` (`rho_fit`,
the scenarios, `TJ70_K`, `J70_scale`, `uncertainty_*`, `S`, `G`), every
row a forecast.  The uncertainty grows with lead time.  How the SWPC
values are converted is in its `meta.conventions`; the SWPC issue times
are in `meta.provenance`.

### leo_sats: satellite drag data

One record per object, with daily values on the file's `time_utc_daily`
axis:

* orbit: `sma_km`, `eccentricity`, `perigee_km`, `apogee_km`,
  `alt_mean_km`, `inc_deg`, `raan_deg`, `v`, `n_tle`;
* drag: `vdot` (TLE-derived in-track drag deceleration) and `sigma_vdot`;
* `rho`: the density product averaged along the object's orbit;
* `bc_inv_m2_per_kg`: inverse ballistic coefficient `Cd·A/m` from
  `vdot = ½·(Cd·A/m)·rho·v²`, and `sigma_bc_inv_m2_per_kg`;
* `stable`: true for the satellites the density solver used;
* `rmspe`: quality of the object's drag fit (null without one).

The file holds every LEO object whose drag is measurable plus every
satellite the solver used; the selection is in `meta.selection`.

## Density at a point

`DragTrackDensity` reads `rho_realtime` (and optionally `rho_full` and
`rho_forecast`) and returns the density at any time and place:

```python
m = DragTrackDensity.from_files('rho_realtime.json', 'rho_full.json',
                                rho_forecast='rho_forecast.json')
m.rho(t, lat_deg, lon_deg, alt_km, source='baseline')
```

* `t`: UTC, as an ISO-8601 string, `datetime`, `np.datetime64` or days
  since 1970-01-01.  Times without a timezone are UTC.
* `lat_deg`: geodetic latitude (WGS-84), degrees.
* `lon_deg`: degrees east.
* `alt_km`: geodetic height above the WGS-84 ellipsoid, km, within the
  Jacchia table (100-1000 km; NaN outside).

Arrays broadcast; all-scalar input returns a float.  `from_files` and
`from_url` take local paths, URLs or `.json.gz` files.

`m.coverage` gives the valid time span, per source.  Outside it `rho()`
returns NaN; nothing is extrapolated.

`source` selects the forecast after the last solver value; before it every
source returns the same solver values.  With `rho_forecast` loaded, its
rows replace the `rho_realtime` fit from the forecast's first time on, and
every source runs to the SWPC horizon; use the two files from the same
run (`meta.run_id`; a mismatch warns).  `'baseline'` (the default) uses the
measured drivers; the scenarios `'solar_only'`, `'quiet'`, `'median'`,
`'moderate'`, `'intense'` and `'extreme'` hold G at the levels defined in
the file's `meta.scenarios`.

How a point density is computed:

```
lst  = apparent local solar time at (t, lon): 12 h where the Sun is overhead
T    = TJ70(t) · (1 + Σ_k TJ70_lm(t, k) · Y_k(lat, lst))     16 spherical harmonics
rho  = K(t) · J70(alt, T)
```

`TJ70` and `TJ70_lm` come from the JSON, interpolated linearly in time.
`K(t)` makes the globe-average density at 400 km equal the product's
value.  `J70` is the Jacchia (1970) profile from the bundled lookup table.

### Position vectors

```python
rho = m.rho_from_state(t, r_km, frame='EarthMJ2000Eq')   # or 'EarthFixed'
```

`r_km` is `(3,)` or `(N, 3)` in km, one time per row.  `EarthMJ2000Eq` is
rotated to Earth-fixed by GMST only (no precession or nutation); pass
`EarthFixed` coordinates for full accuracy.

## Satellite drag data

`load_sats` reads the leo_sats file into `{norad_id: record}` with the product's
field names; daily values are float arrays (null -> NaN) and `date` is the
UTC day of each.

Any LEO object, from `leo_sats`:

```python
from dragtrack import load_sats

leo = load_sats(f'{BASE}/leo_sats.json')
iss = leo[25544]                   # by NORAD ID
iss['date']                        # UTC day of each value
iss['bc_inv_m2_per_kg']            # daily Cd·A/m (m^2/kg), NaN where not available
iss['rho']                         # density along its orbit, each day (kg/m^3)
```

The satellites behind the density solution (`stable`):

```python
import numpy as np

stable = {n: s for n, s in leo.items() if s['stable']}
for norad, s in stable.items():
    print(norad, s['name'], f"{np.nanmedian(s['alt_mean_km']):.0f} km",
          f"BC {np.nanmedian(s['bc_inv_m2_per_kg']):.4f} m^2/kg",
          f"fit rmspe {s['rmspe']:.3f}")
```

## Command line

```
python -m dragtrack rho <time> <lat> <lon> <alt> --rho-realtime PATH_OR_URL \
    [--rho-full PATH_OR_URL] [--rho-forecast PATH_OR_URL] [--source NAME] [--json]
python -m dragtrack info --rho-realtime PATH_OR_URL [--rho-forecast PATH_OR_URL]
```
