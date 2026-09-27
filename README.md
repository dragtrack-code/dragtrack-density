# dragtrack

Thermospheric mass density at any point, `rho(t, lat, lon, alt)`, from the
DragTrack JSON products.

```python
from dragtrack import DragTrackDensity

BASE = 'https://huggingface.co/datasets/djgersh/dragtrack-density-data/resolve/main/latest'
m = DragTrackDensity.from_url(f'{BASE}/rho_realtime.json', f'{BASE}/rho_full.json')
rho = m.rho('2026-09-13T12:00:00Z', lat_deg=45, lon_deg=0, alt_km=450)   # kg/m^3
```

The DragTrack products are derived from USSPACECOM data obtained via
[Space-Track.org](https://www.space-track.org); please cite it when
publishing results based on them.

## Install

```bash
pip install https://huggingface.co/datasets/djgersh/dragtrack-density-data/resolve/main/latest/dragtrack-0.6.5-py3-none-any.whl
```

Python >= 3.9, numpy >= 1.20.  Not on PyPI.

## Data

| File | Covers | Needed |
|---|---|---|
| `rho_realtime.json` | last ~27 days, then a short forecast | always |
| `rho_full.json` | solver history from 2014 | for times before the realtime window |

`from_files` and `from_url` take local paths, URLs or `.json.gz` files.
For a past snapshot, replace `latest/` with a dated directory (listed in
`manifest.json` on the dataset).

`m.coverage` gives the covered span: `solver_start`, `solver_edge` (the
product's `tle_edge_utc`), `end` (last valid time for the default source)
and `end_by_source`.  Outside it `rho()` returns NaN; nothing is
extrapolated.

## Inputs

* `t`: UTC, as an ISO-8601 string, `datetime`, `np.datetime64` or days
  since 1970-01-01.  Times without a timezone are UTC.
* `lat_deg`: geodetic latitude (WGS-84), degrees.
* `lon_deg`: degrees east, `[-180, 180]` or `[0, 360]`.
* `alt_km`: geodetic height above the WGS-84 ellipsoid, km.

Arrays broadcast.  All-scalar input returns a float.

## Sources

Up to `solver_edge` the density is the TLE-based solver reanalysis.  After
it, a fit `rho_400 = a·S + b·G(t − t0) + c` driven by solar EUV (`S`) and
geomagnetic activity (`G`) takes over.  `source` picks the fit:

| `source` | Fit uses |
|---|---|
| `'baseline'` (default) | measured S and G; runs ~1-2 days past `solver_edge` |
| `'solar_only'` | G = 0 (lower envelope) |
| `'quiet'` | G at about the 2014-2026 median |
| `'median'` | median G of the past 27 days |
| `'moderate'` | G at about the 2014-2026 97th percentile |
| `'intense'` | G at about the 2014-2026 99.4th percentile |
| `'extreme'` | G at about the 2014-2026 99.9th percentile (upper envelope) |

Scenarios run ~4-5 days past `solver_edge`.  Before `solver_edge` every
source returns the same values.  `dragtrack.SCENARIOS` lists the scenario
names.  The product's `meta` block defines S, G and every field.

## Method

```
lst  = apparent local solar time at (t, lon): 12 h where the Sun is overhead
T    = TJ70(t) · (1 + Σ_k TJ70_lm(t, k) · Y_k(lat, lst))     16 spherical harmonics
rho  = K(t) · J70(alt, T)
```

`TJ70(t)` and `TJ70_lm(t, k)` come from the JSON and are interpolated
linearly in time.  `K(t)` makes the globe-average density at 400 km equal
the product's global value.  `J70` is the Jacchia (1970) profile, from the
bundled lookup table.

## Command line

```
python -m dragtrack rho <time> <lat> <lon> <alt> --rho-realtime PATH_OR_URL \
    [--rho-full PATH_OR_URL] [--source NAME] [--json]
python -m dragtrack info --rho-realtime PATH_OR_URL
```

## Position vectors (GMAT and other propagators)

```python
rho = m.rho_from_state(t, r_km, frame='EarthMJ2000Eq')   # or 'EarthFixed'
```

`r_km` is `(3,)` or `(N, 3)` in km, with one time per row.  `EarthFixed` is
used as-is.  `EarthMJ2000Eq` is rotated to Earth-fixed by GMST only (no
precession or nutation: ~0.37° in 2026, about 1.5 min of local solar
time).  For better accuracy, pass `EarthFixed` coordinates.  GMAT's
`Latitude`, `Longitude` and `Altitude` from an `EarthFixed` coordinate
system can go straight into `rho()`.

Ballistic coefficients are not part of this package.  Per-satellite
`Cd·A/m` (m²/kg) is `bc_inv_m2_per_kg` in the `leo_sats` and `stable_sats`
products.

## References

* Gershman (2026), *Enabling Near-Real-Time Thermospheric Density
  Monitoring*, NASA-TM/20260006171.
* Jacchia, L.G. (1970), *New Static Models of the Thermosphere and
  Exosphere with Empirical Temperature Profiles*, SAO Special Report 313.
* Picone, J. M., J. T. Emmert, and J. L. Lean (2005), *Thermospheric
  densities derived from spacecraft orbits: Accurate processing of
  two-line element sets*, J. Geophys. Res., 110, A03301,
  doi:10.1029/2004JA010585.
