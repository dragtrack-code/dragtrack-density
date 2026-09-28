# dragtrack

Thermospheric mass density and satellite drag data from the DragTrack JSON
products, and a small Python package (`dragtrack`) to use them.

```python
from dragtrack import DragTrackDensity, load_sats

BASE = 'https://huggingface.co/datasets/djgersh/dragtrack-density-data/resolve/main/latest'
m = DragTrackDensity.from_url(f'{BASE}/rho_realtime.json', f'{BASE}/rho_full.json')
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
pip install https://huggingface.co/datasets/djgersh/dragtrack-density-data/resolve/main/latest/dragtrack-0.6.8-py3-none-any.whl
```

Python >= 3.9, numpy >= 1.21.  Not on PyPI.

## The data products

Four JSON files, all in `latest/` on the
[Hugging Face dataset](https://huggingface.co/datasets/djgersh/dragtrack-density-data).
They are regenerated together after each solver update (about twice a
day) and carry the same `meta.run_id`.  Past `rho_realtime`, `stable_sats`
and `leo_sats` files are kept in dated directories (`YYYY-MM-DD/`,
versioned file names), listed in `manifest.json`.

| File | What it is | Covers | Size |
|---|---|---|---|
| `rho_realtime.json` | Global density: recent solver values, fit and forecast scenarios | last ~27 days + ~1-5 day forecast | ~6 MB |
| `rho_full.json` | Global density: full solver history | 2014 to the latest solver value | ~19 MB |
| `stable_sats.json` | Daily drag data for the satellites the solver uses | last ~27 days | ~1.5 MB |
| `leo_sats.json` | Daily drag data for every LEO object with measurable drag | last ~27 days | ~30 MB |

Every file describes its own fields, units and conventions in its `meta`
block; read that for exact definitions.  Times are UTC.

### rho_realtime and rho_full: density

Both files hold the density of the whole thermosphere as a function of
time:

* `altitude_km`: 250-600 km in 50 km steps, geodetic height above the
  WGS-84 ellipsoid.
* `solver`: the TLE-based solution, one row per time.  `rho` (kg/m^3,
  globe-average density at each altitude), `uncertainty` (relative
  1-sigma), `TJ70_K` and `J70_scale` (the Jacchia-1970 temperature and
  scale that give `rho` at any other height), `source` and `bin_days`
  (the averaging behind each row).  Daily values are stamped at noon;
  from 2022 on, most days also have 3-hour values.
* `spatial`: 16 spherical-harmonic coefficients per row (`TJ70_lm`) that
  describe how the temperature, and so the density, varies with latitude
  and local solar time.
* `fit` (`rho_realtime` only): hourly rows from a regression
  `rho_400 = a·S + b·G(t − t0) + c` on the solver values of the past 27
  days, driven by solar EUV (`S`, GOES) and geomagnetic activity (`G`,
  from the solar wind).  `rho_fit` extends past the latest solver value
  with measured drivers (~1-2 days); `rho_<scenario>` holds G fixed at
  set levels (~4-5 days).

`rho_realtime` has the last ~27 days of `solver` rows plus the `fit`;
`rho_full` has every `solver` row since 2014.  Use both for a continuous
record from 2014 to the forecast.

### stable_sats and leo_sats: satellite drag data

One record per object, with daily values on the file's `time_utc_daily`
axis (the last ~27 complete UTC days):

* orbit: `sma_km`, `eccentricity`, `perigee_km`, `apogee_km`,
  `alt_mean_km` (heights above a 6371.010 km sphere; they describe the
  orbit only), `inc_deg`, `raan_deg`, `v` (km/s), `n_tle`;
* drag: `vdot` (TLE-derived in-track drag deceleration, m/s^2) and
  `sigma_vdot`;
* `rho`: the density product averaged along the object's orbit (kg/m^3);
* `bc_inv_m2_per_kg`: inverse ballistic coefficient `Cd·A/m` (m^2/kg)
  from `vdot = ½·(Cd·A/m)·rho·v²`, and `sigma_bc_inv_m2_per_kg`.  Null on
  days with a reboost (vdot <= 0) or a manoeuvre (more than 3x from the
  object's own 27-day median).

`stable_sats` lists the satellites the density solver used on its latest
complete day (~170), with the `rmspe` of each satellite's drag fit.
`leo_sats` covers every LEO object whose drag is measurable (~3500):
perigee 150-600 km, apogee <= 1000 km, at least 10 days with vdot and a
clear signal (`meta.selection`).

## Density at a point

`DragTrackDensity` reads `rho_realtime` (and optionally `rho_full`) and
returns the density at any time and place:

```python
m = DragTrackDensity.from_files('rho_realtime.json', 'rho_full.json')
m.rho(t, lat_deg, lon_deg, alt_km, source='baseline')
```

* `t`: UTC, as an ISO-8601 string, `datetime`, `np.datetime64` or days
  since 1970-01-01.  Times without a timezone are UTC.
* `lat_deg`: geodetic latitude (WGS-84), degrees.
* `lon_deg`: degrees east, `[-180, 180]` or `[0, 360]`.
* `alt_km`: geodetic height above the WGS-84 ellipsoid, km, 100-1000 km
  (the Jacchia table; NaN outside).

Arrays broadcast; all-scalar input returns a float.  `from_files` and
`from_url` take local paths, URLs or `.json.gz` files.

`m.coverage` gives the valid span: `solver_start`, `solver_edge` (the
product's `tle_edge_utc`), `end` (last valid time for the default source)
and `end_by_source`.  Outside it `rho()` returns NaN; nothing is
extrapolated.

`source` picks the forecast after `solver_edge`; before it every source
returns the same solver values:

| `source` | Fit uses |
|---|---|
| `'baseline'` (default) | measured S and G |
| `'solar_only'` | G = 0 (lower envelope) |
| `'quiet'` | G at about the 2014-2026 median |
| `'median'` | median G of the past 27 days |
| `'moderate'` | G at about the 2014-2026 97th percentile |
| `'intense'` | G at about the 2014-2026 99.4th percentile |
| `'extreme'` | G at about the 2014-2026 99.9th percentile (upper envelope) |

A scenario missing from a file raises an error.

How a point density is computed:

```
lst  = apparent local solar time at (t, lon): 12 h where the Sun is overhead
T    = TJ70(t) · (1 + Σ_k TJ70_lm(t, k) · Y_k(lat, lst))     16 spherical harmonics
rho  = K(t) · J70(alt, T)
```

`TJ70` and `TJ70_lm` come from the JSON, interpolated linearly in time.
`K(t)` makes the globe-average density at 400 km equal the product's
value.  `J70` is the Jacchia (1970) profile from the bundled lookup table.

### Position vectors (GMAT and other propagators)

```python
rho = m.rho_from_state(t, r_km, frame='EarthMJ2000Eq')   # or 'EarthFixed'
```

`r_km` is `(3,)` or `(N, 3)` in km, one time per row.  `EarthFixed` is
used as-is.  `EarthMJ2000Eq` is rotated to Earth-fixed by GMST only (no
precession or nutation: ~0.37° in 2026, about 1.5 min of local solar
time); for better accuracy pass `EarthFixed` coordinates.  GMAT's
`Latitude`, `Longitude` and `Altitude` from an `EarthFixed` coordinate
system can go straight into `rho()`.

## Satellite drag data

`load_sats` reads either file into `{norad_id: record}`, with the product's
field names; daily values are float arrays (null -> NaN) and `date` gives
the UTC day of each.

Any LEO object, from `leo_sats`:

```python
from dragtrack import load_sats

leo = load_sats(f'{BASE}/leo_sats.json')
iss = leo[25544]                   # by NORAD ID
iss['date']                        # UTC day of each value (datetime64[D])
iss['bc_inv_m2_per_kg']            # daily Cd·A/m (m^2/kg), NaN where not available
iss['rho']                         # density along its orbit, each day (kg/m^3)
```

The satellites behind the density solution, from `stable_sats`:

```python
import numpy as np

stable = load_sats(f'{BASE}/stable_sats.json')
for norad, s in stable.items():
    print(norad, s['name'], f"{np.nanmedian(s['alt_mean_km']):.0f} km",
          f"BC {np.nanmedian(s['bc_inv_m2_per_kg']):.4f} m^2/kg",
          f"fit rmspe {s['rmspe']:.3f}")
```

## Command line

```
python -m dragtrack rho <time> <lat> <lon> <alt> --rho-realtime PATH_OR_URL \
    [--rho-full PATH_OR_URL] [--source NAME] [--json]
python -m dragtrack info --rho-realtime PATH_OR_URL
```
