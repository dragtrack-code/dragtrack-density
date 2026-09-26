# dragtrack

Total thermospheric mass density at a point rho(t, lat, lon, alt)`

```python
from dragtrack import DragTrackDensity

m = DragTrackDensity.from_files('dragtrack_rho_realtime.json',
                                 'dragtrack_rho_full.json')
rho = m.rho('2026-09-13T12:00:00Z', lat_deg=45, lon_deg=0, alt_km=450)
```

The package ships a bundled Jacchia-1970 altitude profile lookup table and reads the
DragTrack JSON data products.

The DragTrack products are derived from USSPACECOM data obtained via
[Space-Track.org](https://www.space-track.org); please cite it when publishing results
based on them.

---

## Install

```bash
pip install https://huggingface.co/datasets/djgersh/dragtrack-density-data/resolve/main/latest/dragtrack-0.6.2-py3-none-any.whl
```

Runtime dependency: `numpy>=1.20`.  Not on PyPI — the wheel above is
shipped alongside the data on Hugging Face.

---

## Getting the data

Two JSONs describe the atmosphere:

| File | Time coverage | Required? |
|---|---|---|
| `dragtrack_rho_realtime_*.json` | past ~27 days + short extrapolation | yes |
| `dragtrack_rho_full_*.json`     | solver history from 2014-01-01      | optional (needed for times before the realtime window) |

Fetch the latest snapshot from the DragTrack Hugging Face dataset:

```python
BASE = 'https://huggingface.co/datasets/djgersh/dragtrack-density-data/resolve/main/latest'
m = DragTrackDensity.from_url(
    f'{BASE}/rho_realtime.json',
    rho_full_url=f'{BASE}/rho_full.json',      # optional: 2014+ history
)
print(m.coverage)   # {'solver_start': ..., 'solver_edge': ..., 'end': ..., 'end_by_source': {...}}
```

Queries outside `m.coverage` (`solver_start` → `end`) return `NaN` — the
package never extrapolates or holds the edge value.  `end` is for the
default `source='baseline'`; scenarios run further (`end_by_source[name]`).

Pin to a historic snapshot by swapping `latest/` for a dated directory
(`2026-09-23/` etc.) — see `manifest.json` on the dataset for the list.
Or download the files and use `from_files(...)`.

---

## Coordinate conventions

* **Time** — UTC.  ISO-8601 strings (`'2026-09-13T12:00:00Z'`),
  `datetime`, `np.datetime64`, or numeric days-since-1970 UTC.
* **Latitude** — WGS84 geodetic, degrees, `[-90, 90]`.
* **Longitude** — degrees, either `[-180, 180]` or `[0, 360]`.
* **Altitude** — WGS84 geodetic, km.  The products' own altitude grid
  (`altitude_km` in the JSONs) is height above a **6371.010 km sphere**
  (the TLE solver's mean-Earth radius); the package converts your
  geodetic altitude to that reference internally (+7 km at the equator,
  −14 km at the poles for the same geodetic altitude).

```python
import numpy as np
alt = np.linspace(250, 600, 8)
rho = m.rho('2026-09-13T12:00:00Z', lat_deg=45, lon_deg=0, alt_km=alt)
```

---

## JSON Contents

`DragTrackDensity` includes density estimates associated with a TLE-based solver and an extrapolation using measured energy drivers.


| Segment    | Where            | Time span                                     | What it is |
|------------|------------------|-----------------------------------------------|------------|
| `solver`   | `rho_realtime` (last ~27 d), `rho_full` (2014-01-01 →) | → solver edge (~1–2 days behind now) | TLE-based mass density profiles (reanalysis) |
| `fit`      | `rho_realtime`   | ~27 d back → `max(S_edge, G_edge) + horizon` | Hourly linear regression `rho_zref = a·S(t) + b·G(t − t0) + c` fit against the last 27 d of solver.  Extends past raw-driver edge by 2× LPF RC (S: 48-h RC → +96 h; G: 9-h RC → +18 h) via persist-pad + causal IIR. |

The package uses the **solver** segment up to its edge and the fit
segment only after it.  (The fit block also carries hindcast
rows over the solver window for comparison; the package ignores them, so
choosing a scenario never changes the past.)  Queries after the end of
the fit return `NaN`.

## Drivers (S, G)

* `S` — solar-EUV energy input (`W/m²`, 48-h low-pass).  Sourced from the
  **GOES-19 EUVS composite** (multi-band extreme-UV integration).
* `G` — geomagnetic input (`nT`).  Computed from OMNI solar-wind at L1
  via the Newell (2007) merging function
  `dΦ/dt = V^(4/3) · Bt^(2/3) · sin^(8/3)(|θ|/2)`, outlier-filtered and
  boxcar-smoothed hourly, mapped to a pseudo-Ap with fixed calibration
  constants `clip(3.51e-3·dΦ/dt − 2.59, 0)`, 9-hour low-pass, ×2 to nT,
  then a log-based softplus `λ · log(1 + exp((x − G0) / λ))` with
  `G0 = 30 nT`, `λ = 20 nT` to smooth the transition through low activity.

## Scenarios

`m.rho(..., source=name)` picks which activity scenario is desired.
Only the future portion (fit) responds to `source`; historic
solver data is the same across all sources.

| `source` | Description |
|---|---|
| `'baseline'` (default) | Best current estimate driven by observed forcing. |
| `'solar_only'` | Solar EUV driver (`S`) only, no geomagnetic driver (`G`). |
| `'quiet'` | Quiet geomagnetic conditions. |
| `'moderate'` | Moderate geomagnetic activity. |
| `'intense'` | Intense activity (strong geomagnetic storm). |
| `'extreme'` | Extreme activity (major storm). |
| `'median'` | Median activity level over the recent 27-d window. |

`dragtrack.SCENARIOS` returns the full tuple of scenario names.

---

## Command line

```
python -m dragtrack rho <time> <lat> <lon> <alt> \
    --rho-realtime PATH_OR_URL \
    [--rho-full PATH_OR_URL] \
    [--source baseline|solar_only|...]

python -m dragtrack info --rho-realtime PATH_OR_URL
```

Example:

```console
$ python -m dragtrack rho 2026-09-20T12:00:00Z 0 0 400 \
      --rho-realtime dragtrack_rho_realtime.json
3.002456e-12
```

Add `--json` to `rho` for a full record instead of a bare number.

---

## Obtaining sub-orbit densities 

The JSON providep:

* a **global average** — the horizontally-averaged density at 400 km
  (`solver.rho[:, i400]`) and the corresponding global mean thermospheric
  temperature (`solver.TJ70_K`), plus their `fit`-block extensions
  toward the present and future;
* a **spatial pattern** — sixteen spherical harmonic coefficients `TJ70_lm[t, k]` that
  describe how the thermospheric temperature varies over latitude and
  local solar time as a perturbation on top of the
  global background.

Point density at `(t, lat, lon, alt)` is reconstructed as:

```
lst_h    = (UT_hours(t) + lon_deg / 15) mod 24
Y_k      = 16 spherical-harmonic modes evaluated at (lat, lst_h)
TJ70     = TJ70_bg(t) · (1 + Σ_k TJ70_lm(t, k) · Y_k)
rho(alt) = K(t) · J70(alt, TJ70)
```

`K(t)` is a per-timestep amplitude that keeps the sphere-average of the
reconstructed density at 400 km equal to the global background from the
JSON.  Vertical structure comes from the packaged Jacchia-1970
lookup table.

---

## Using it with GMAT

GMAT (General Mission Analysis Tool) needs density at the spacecraft's
`(t, lat, lon, alt)`.  You can feed our densities in either as a
precomputed table or by calling the package at each propagation step.
The package accepts either a body-fixed or an inertial state vector, so
you don't have to convert coordinates yourself.

### Coordinates

* Latitude — WGS84 geodetic degrees, matches GMAT's `Latitude`.
* Longitude — Earth-body-fixed degrees, matches GMAT's `Longitude`
  extracted from an `EarthFixed` coordinate system.  **Don't use
  `atan2(y, x)` on an inertial state — that gives right ascension, not
  longitude.**
* Altitude — WGS84 geodetic km, matches GMAT's `Altitude`.
* Time — UTC.  The package accepts an ISO-8601 string or a
  `datetime`; GMAT users typically convert `UTCGregorian` to that.

Give it geodetic values from an `EarthFixed` coordinate system and you
match the model's conventions exactly.

### Passing an ECI or ECEF state directly

If you'd rather work with the raw 3-vector, call ``rho_from_state``:

```python
r_km = np.array([x, y, z])   # ECI or ECEF, in km
rho = m.rho_from_state('2026-09-13T12:00:00Z', r_km,
                        frame='EarthMJ2000Eq')      # or 'EarthFixed'
```

`frame='EarthMJ2000Eq'` matches GMAT's default inertial frame; the
package handles the rotation to Earth-fixed and the WGS84 conversion
internally.

**Frame caveat.** `EarthMJ2000Eq` is treated as TEME here —
only the GMST rotation is applied (no precession / nutation).  In 2026
precession since J2000 is ~0.37° (~45 km of along-track position at LEO),
which is mostly a longitude shift of ~1.5 min of local solar time — a
negligible density difference next to the atmosphere model's own
uncertainty (~15-30 % at LEO).  If you need geodetic accuracy better than
that, pre-transform your ECI state into `EarthFixed` inside GMAT and pass
`frame='EarthFixed'`.

### Precomputing a density table for GMAT

The simplest integration: propagate in GMAT with `AtmosphereModel = None`,
export the ephemeris, evaluate `dragtrack` at each sample, then
re-propagate with a tabular atmosphere.  A one-shot script:

```python
import numpy as np
from dragtrack import DragTrackDensity

m = DragTrackDensity.from_files('dragtrack_rho_realtime.json',
                                 'dragtrack_rho_full.json')

# ephem: n × (t_iso, x_km, y_km, z_km) in EarthMJ2000Eq, exported by GMAT.
ephem = np.loadtxt('trajectory.csv', dtype=object, delimiter=',',
                    skiprows=1)
t_iso = ephem[:, 0]
r_km  = ephem[:, 1:4].astype(float)

rho = m.rho_from_state(list(t_iso), r_km, frame='EarthMJ2000Eq')
np.savetxt('rho_table.csv',
           np.column_stack([t_iso, rho]), fmt='%s,%.6e',
           header='time_utc,rho_kg_m3')
```

Point GMAT's atmosphere model at `rho_table.csv` and re-propagate; the
drag now uses DragTrack densities.

### Ballistic coefficient input

DragTrack's per-satellite BC estimates are in **m²/kg** — the same
`Cd·A/m` product GMAT's drag model expects. **Cd is included in this BC** If your GMAT setup takes
`Cd`, drag area, and mass separately, distribute the product any way
you like (e.g. `if Cd = 2.2`, `Area/mass = BC / 2.2 m²/kg`).

### Velocity convention

GMAT's drag force uses velocity relative to the co-rotating atmosphere
(`v_rel`), not the inertial orbital speed.  DragTrack's BC estimation
integrates over the same co-rotation factor, so the numbers are
compatible to better than ~1% under normal LEO conditions.

---

## Relevant References

* Gershman (2026), *Enabling Near-Real-Time Thermospheric Density
  Monitoring*, NASA-TM/20260006171.
* Jacchia, L.G. (1970), *New Static Models of the Thermosphere and
  Exosphere with Empirical Temperature Profiles*, SAO Special Report 313.
* Picone, J. M., J. T. Emmert, and J. L. Lean (2005), *Thermospheric densities derived from spacecraft orbits: Accurate processing of two-line element sets*, J. Geophys. Res., 110, A03301, doi:10.1029/2004JA010585.
* Roberts, C.E. (1971), *An analytic model for upper atmosphere densities
  based upon Jacchia's 1970 models*, Celestial Mechanics 4, 368-377.
