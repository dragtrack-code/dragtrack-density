# dragtrack

Total thermospheric mass density at a point rho(t, lat, lon, alt)`

```python
from dragtrack import DragTrackDensity

m = DragTrackDensity.from_files('dragtrack_rho_realtime.json',
                                 'dragtrack_rho_forecast.json')
rho = m.rho('2026-09-13T12:00:00Z', lat_deg=45, lon_deg=0, alt_km=450)
```

The package ships a bundled Jacchia-1970 altitude profile lookup table and reads two
DragTrack JSON data products

---

## Install

```bash
pip install dragtrack
```

Runtime dependency: `numpy>=1.20`.

---

## Getting the data

Two JSONs describe the atmosphere:

| File | Time coverage | Required? |
|---|---|---|
| `dragtrack_rho_realtime_*.json` | past ~27 days + short extrapolation | yes |
| `dragtrack_rho_forecast_*.json` | up to 45 days ahead (SWPC-driven)   | optional |

Fetch the latest from the DragTrack HTTP endpoint:

```python
m = DragTrackDensity.from_url(
    'https://.../dragtrack_rho_realtime_v0.8.0.json',
    rho_forecast_url='https://.../dragtrack_rho_forecast_v0.8.0.json',
)
```

or download the files and use `from_files(...)`.

---

## Coordinate conventions

* **Time** — UTC.  ISO-8601 strings (`'2026-09-13T12:00:00Z'`),
  `datetime`, `np.datetime64`, or numeric days-since-1970 UTC.
* **Latitude** — WGS84 geodetic, degrees, `[-90, 90]`.
* **Longitude** — degrees, either `[-180, 180]` or `[0, 360]`.
* **Altitude** — WGS84 geodetic, km.

```python
import numpy as np
alt = np.linspace(250, 600, 8)
rho = m.rho('2026-09-13T12:00:00Z', lat_deg=45, lon_deg=0, alt_km=alt)
```

---

## JSON Contents

`DragTrackDensity` includes density estimates associated with a TLE-based solver, an extrapolation using measured energy drivers, and a forecast using forecasted energy drivers.


| Segment    | Where            | Time span                                     | What it is |
|------------|------------------|-----------------------------------------------|------------|
| `solver`   | `rho_realtime`   | 2014-01-01 → TLE-edge (~today)                | TLE-based mass density profiles |
| `fit`      | `rho_realtime`   | ~27 d back → `max(S_edge, G_edge) + horizon` | Hourly linear regression `rho_zref = a·S(t) + b·G(t) + c` fit against the last 27 d of solver.  Extends past raw-driver edge by 2× LPF RC (S: 48-h RC → +96 h; G: 9-h RC → +18 h) via persist-pad + causal IIR. |
| `forecast` | `rho_forecast`   | present → +45 d                               | Same fit function, driven with SWPC-predicted S / G (F10.7 → S via local 40-d linear fit; Kp → G via `b + k·exp(Kp/2)` with seeded constants). |

Point queries past the forecast edge fall back to the last-forecast value
(edge-hold / persistence) — not NaN.  Check `t <= forecast_end` yourself
if you need to detect out-of-range queries.

## Drivers (S, G)

* `S` — solar-EUV energy input (`W/m²`, 48-h low-pass).  Sourced from the
  **GOES-19 EUVS composite** (multi-band extreme-UV integration). F10.7 is used only as a linear proxy to drive the SWPC-forecast extension of `S`
* `G` — geomagnetic input (`nT`).  Computed from OMNI solar-wind at L1
  via the Newell (2007) merging function
  `dΦ/dt = V^(4/3) · Bt^(2/3) · sin^(8/3)(|θ|/2)`, boxcar-smoothed,
  mapped to a pseudo-Ap by a ~30-day rolling linear fit against Kp-Ap,
  then a log-based softplus `λ · log(1 + exp((x − G0) / λ))` with
  `G0 = 30 nT`, `λ = 20 nT` to smooth the transition through low activity.

## Per-satellite BC (`bc_track/<norad>.json`)

Alongside the atmosphere JSONs, the HF dataset ships a ~30-day per-satellite
BC time series (`Cd·A/m` in m²/kg) for each tracked satellite:

```python
import json, urllib.request
BASE = 'https://huggingface.co/datasets/djgersh/dragtrack-density-data/resolve/main/latest'
d = json.loads(urllib.request.urlopen(f'{BASE}/bc_track/28485.json').read())    # Swift
print(d['meta']['satellite'], d['n_records'])
for r in d['records'][-3:]:
    print(r['time_utc'], 'BC=', r['BC_m2_kg'],
                          'rho=', r['rho_kg_m3'], 'vdot=', r['vdot_m_s2'])
```

Recipe:

    BC = 2 · vdot / (rho_eff · v²)

* `vdot` = Picone et al. [2005] drag deceleration from consecutive raw TLE pairs
  (mean-motion change divided by an F·v³-weighted time integral,
  `F = (1 - r·ω·cos(inc)/v)²` = co-rotation factor).
* `rho_eff` = `Σρ·F·v³ / Σ F·v³` sampled at 5-min steps over the same
  pair interval
* `v` = pair-mean orbital speed (m/s)
  `v₀²` factor.
* Each record includes `state_t1` / `state_t2` (WGS84 geodetic at the
  bounding TLE epochs) for GMAT coordinate-convention verification.

Quick plot of Swift (28485) BC + drivers over the last month:

```python
import json, urllib.request, numpy as np
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from datetime import datetime

d = json.loads(urllib.request.urlopen(f'{BASE}/bc_track/28485.json').read())
r = d['records']
t   = np.array([mdates.date2num(datetime.strptime(x['time_utc'], '%Y-%m-%dT%H:%M:%SZ')) for x in r])
BC  = np.array([x['BC_m2_kg']  for x in r])
rho = np.array([x['rho_kg_m3'] for x in r])
vd  = np.array([x['vdot_m_s2'] for x in r])
alt = np.array([x['alt_km']    for x in r])

fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
ax[0].plot(t, BC,      'o-'); ax[0].set_ylabel('BC (m^2/kg)')
ax[0].set_title(f'{d["meta"]["satellite"]["name"]} ({d["meta"]["satellite"]["norad_id"]}) — {d["n_records"]} records')
ax[1].plot(t, vd*1e6,  'o-'); ax[1].set_ylabel('vdot (µm/s^2)')
ax[2].semilogy(t, rho, 'o-'); ax[2].set_ylabel('rho (kg/m^3)')
ax[3].plot(t, alt,     'o-'); ax[3].set_ylabel('alt (km)')
ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig('swift_bc.png', dpi=120)
```

## Scenarios

`m.rho(..., source=name)` picks which activity scenario is desired.
Only the future portion (fit + forecast) responds to `source`; historic
solver data is the same across all sources.

| `source` | Description |
|---|---|
| `'baseline'` (default) | Best current estimate driven by observed + SWPC-predicted forcing. |
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
    [--rho-forecast PATH_OR_URL] \
    [--source baseline|solar_only|...]

python -m dragtrack info --rho-realtime PATH_OR_URL
```

Example:

```console
$ python -m dragtrack rho 2026-09-20T12:00:00Z 0 0 400 \
      --rho-realtime dragtrack_rho_realtime.json \
      --rho-forecast dragtrack_rho_forecast.json
3.002456e-12
```

Add `--json` to `rho` for a full record instead of a bare number.

---

## Obtaining sub-orbit densities 

The JSON providep:

* a **global average** — the horizontally-averaged density at 400 km
  (`solver.rho[:, i400]`) and the corresponding global mean thermospheric
  temperature (`solver.J70_Tinf_K`), plus their `fit`-block extensions
  toward the present and future;
* a **spatial pattern** — sixteen spherical harmonic coefficients `Tinf_lm[t, k]` that
  describe how the thermospheric temperature varies over latitude and
  local solar time as a perturbation on top of the
  global background.

Point density at `(t, lat, lon, alt)` is reconstructed as:

```
lst_h    = (UT_hours(t) + lon_deg / 15) mod 24
Y_k      = 16 spherical-harmonic modes evaluated at (lat, lst_h)
Tinf     = Tinf_bg(t) · (1 + Σ_k Tinf_lm(t, k) · Y_k)
rho(alt) = K(t) · J70(alt, Tinf)
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
only the GMST rotation is applied (no precession / nutation).  For a
density lookup this is an ECEF position offset of ~50 arcsec at the epoch
(~1 km at 400 km altitude), well below the atmosphere model's own
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
                                 'dragtrack_rho_forecast.json')

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

Per-satellite BC time series (recipe:
`BC = 2·vdot / (rho_eff · v²)` with `vdot` from Picone et al. [2005] TLE-pair drag
integration and `rho_eff = Σρ·F·v³ / ΣF·v³` at the same F·v³ weighting)
are shipped alongside the density JSONs under
`latest/bc_track/<norad>.json` in the HF dataset:

```python
import json, urllib.request
BASE = 'https://huggingface.co/datasets/djgersh/dragtrack-density-data/resolve/main/latest'
d = json.loads(urllib.request.urlopen(f'{BASE}/bc_track/28485.json').read())
for r in d['records'][-3:]:
    print(r['time_utc'], 'BC=', r['BC_m2_kg'])
```

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
