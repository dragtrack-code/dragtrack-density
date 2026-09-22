# dragtrack

Point atmospheric density `rho(t, lat, lon, alt)` for satellite-drag
work, in a couple of lines of Python.

```python
from dragtrack import DragTrackDensity

m = DragTrackDensity.from_files('dragtrack_rho_realtime.json',
                                 'dragtrack_rho_forecast.json')
rho = m.rho('2026-09-13T12:00:00Z', lat_deg=45, lon_deg=0, alt_km=450)
```

The package ships a bundled Jacchia-1970 lookup table and reads two
JSON data products from the DragTrack service; you only need the JSONs.

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

Inputs broadcast the usual way — mix scalars and arrays freely:

```python
import numpy as np
alt = np.linspace(250, 600, 8)
rho = m.rho('2026-09-13T12:00:00Z', lat_deg=45, lon_deg=0, alt_km=alt)
```

---

## Scenarios

`m.rho(..., source=name)` picks which flavour of the forecast you want.
Only the future portion (fit + forecast) responds to `source`; historic
solver reanalysis is the same across all sources.

| `source` | What it is |
|---|---|
| `'baseline'` (default) | Best current estimate driven by observed + SWPC-predicted forcing. |
| `'solar_only'` | Solar (F10.7) driver only, no geomagnetic activity. |
| `'quiet'` | Quiet geomagnetic conditions. |
| `'moderate'` | Moderate geomagnetic activity. |
| `'intense'` | Intense activity (strong geomagnetic storm). |
| `'extreme'` | Extreme activity (major storm). |
| `'median'` | Median activity level over the recent window. |

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

## Combining the globals with the spatial pattern

The JSON gives you two things per timestep:

* a **global background** — the horizontally-averaged density at 400 km
  (`solver.rho[:, i400]`) and the corresponding global mean thermospheric
  temperature (`solver.J70_Tinf_K`), plus their `fit`-block extensions
  toward the present and future;
* a **spatial pattern** — sixteen coefficients `Tinf_lm[t, k]` that
  describe how the thermospheric temperature varies over latitude and
  local solar time as a small fractional perturbation on top of the
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
JSON — no matter what the spatial pattern does, the sphere mean is
preserved.  Vertical structure comes from the packaged Jacchia-1970
lookup table.

`DragTrackDensity.rho(...)` runs all of this for you, but the pieces
are trivial enough that you can reproduce it in your own code if you
want.

---

## References

* Gershman (2026), *Enabling Near-Real-Time Thermospheric Density
  Monitoring*, NASA-TM/20260006171.
* Jacchia, L.G. (1970), *New Static Models of the Thermosphere and
  Exosphere with Empirical Temperature Profiles*, SAO Special Report 313.
* Roberts, C.E. (1971), *An analytic model for upper atmosphere densities
  based upon Jacchia's 1970 models*, Celestial Mechanics 4, 368-377.
