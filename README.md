# dragtrack

Atmospheric density `rho(t, lat, lon, alt)` for satellite-drag applications,
reconstructed from DragTrack JSON data products.

`dragtrack` is a small, dependency-light Python package that combines:

1. A **near-real-time** density product emitted by the DragTrack Ops pipeline
   (`dragtrack_rho_realtime.json`, ~12-24 hr latency), which provides a
   time-varying global mean density profile plus per-day coefficients linking
   `Tinf` to `rho_400`;
2. A **static Fourier climatology** of the 15 spherical-harmonic modes of
   `beta_frac(lat, LST, doy)` (bundled — derived from 51 years of MSIS
   NRLMSISE-00 driven by real F10.7/Ap); and
3. A **Jacchia 1970** vertical density lookup table (bundled — 3D grid on
   altitude, `Tinf`, and Roberts `Tx`),

to deliver a self-contained point-density model that runs anywhere Python
runs, with no compiled code and no network calls at query time.

---

## Why

The DragTrack Ops pipeline produces a horizontally-averaged density track
`rho_bg(t)` at 400 km plus a linear per-day map `Tinf = mt*rho_400 + bt`.
For orbit propagation you need a **point** density that also captures the
spatial pattern (diurnal bulge, hemispheric asymmetry, etc.).  The pattern
is remarkably stable when normalized by the global mean, so a fixed Fourier
climatology of the 15-mode fractional pattern `beta_frac(lat, LST, doy)`
supplies the shape, while the runtime JSON supplies the amplitude and the
temperature.  Jacchia-70 handles the vertical slice.

---

## Install

```bash
pip install .
# or with orbit-propagation extras
pip install ".[ephem]"
```

Runtime dependencies: `numpy>=1.20` (required), `sgp4>=2.20` (only for
`propagate_tle` / `rho_along_tle`).

The bundled static assets (`dragtrack/data/*.json`) ship with the wheel via
`[tool.setuptools.package-data]`, so no extra download is needed.

---

## Quick start

```python
from dragtrack import DragTrackDensity

m = DragTrackDensity.from_files('dragtrack_rho_realtime.json')

# Single point
rho = m.rho('2026-09-13T12:00:00Z', lat_deg=45, lon_deg=0, alt_km=450)
# → ~1e-12 kg/m^3
```

### Along a TLE orbit

```python
l1 = '1 25544U 98067A   26256.50000000  .00000000  00000-0  00000-0 0  9997'
l2 = '2 25544  51.6400   0.0000 0001000   0.0000   0.0000 15.50000000000019'

times = [f'2026-09-13T{h:02d}:00:00Z' for h in range(24)]
t, lat, lon, alt, rho = m.rho_along_tle(l1, l2, times)
```

### Altitude sweep

```python
import numpy as np
alt = np.linspace(250, 600, 8)
rho = m.rho('2026-09-13T12:00:00Z',
            np.full_like(alt, 45.0),  # lat
            np.zeros_like(alt),        # lon
            alt)
```

### Diurnal / global pattern at a fixed time and altitude

```python
lat = np.linspace(-80, 80, 33)
lon = np.linspace(-180, 180, 65)
LAT, LON = np.meshgrid(lat, lon, indexing='ij')
rho = m.rho('2026-09-13T12:00:00Z', LAT.ravel(), LON.ravel(),
            np.full(LAT.size, 400.0)).reshape(LAT.shape)
```

### Fetch a runtime product over HTTP

```python
url = ('https://.../dragtrack_rho_realtime_v0.8.0.json')
m = DragTrackDensity.from_url(url)
```

---

## Command line

```
python -m dragtrack rho <isot> <lat> <lon> <alt> --rho-realtime PATH
python -m dragtrack info                          --rho-realtime PATH
```

Example:

```console
$ python -m dragtrack rho 2026-09-13T12:00:00Z 45 0 450 \
      --rho-realtime path/to/dragtrack_rho_realtime.json
1.076611e-12
```

Use `--json` for a full record instead of a bare number.

---

## Reconstruction math

Given a query point `(t, lat, lon, alt)`:

```
lst_h    = (UT_hours + lon_deg / 15) mod 24
doy      = day_of_year(t)                                 (1..366)
beta_l   = Fourier eval at doy for each of 15 SH modes    (clim JSON)
Y_l      = 15 SH basis @ (lat, lst)                       (basis.py)
rho_bg   = interp(solver.rho[:, i400], t)                 (400 km column)
mt, bt   = interp(solver.T_J70_m, solver.T_J70_b, t)      (per-day linear)
rho_400  = rho_bg * (1 + sum_l beta_l · Y_l)              (multiplicative)
Tinf_pt  = mt * rho_400 + bt
rho(alt) = rho_400 * J70(alt, Tinf_pt) / J70(400, Tinf_pt)
```

- `beta_l` is the *fractional* perturbation for SH mode l — dimensionless.
- `Y_l` is the un-normalized real SH basis defined in `dragtrack/basis.py`
  (`REG_NAMES = ('Y10','Y11c','Y11s','Y20','Y21c','Y21s','Y22c','Y22s',
  'Y30','Y31c','Y31s','Y32c','Y32s','Y33c','Y33s')`).  It must match the
  basis used to fit the climatology coefficients — **do not renormalize**.
- `J70(z, Tinf)` is trilinear in `log10(rho)` on the packaged LUT grid, with
  Roberts `Tx = calc_Tx(Tinf)` used as the third axis at query time.

---

## Data-source flow

```
+---------------------+     +----------------------------+
| dragtrack_ops       |     | packaged static assets     |
| pipeline            |     |                            |
| (produces JSON)     |     |  spatial_clim_analytical   |
+----------+----------+     |  j70_density_lookup_tx     |
           |                +--------------+-------------+
           v                               |
+--------------------------+               |
| dragtrack_rho_realtime   |               |
|  .solver.time_utc[]      |               |
|  .solver.rho[t, z]       |               |
|  .solver.T_J70_m/b[t]    |               |
|  .meta.tle_edge_utc      |               |
+-----------+--------------+               |
            |                              |
            +--------------+---------------+
                           v
                +---------------------+
                | DragTrackDensity    |
                |   .rho(t, lat,      |
                |         lon, alt)   |
                +----------+----------+
                           v
                   rho (kg / m^3)
```

---

## API surface

Top-level exports (see `from dragtrack import ...`):

| name | what it is |
|------|------------|
| `DragTrackDensity` | main class, `.rho(...)`, `.rho_along_tle(...)`, `.rho_along_kepler(...)` |
| `Y_modes(lat_deg, lst_h)` | 15-SH basis evaluator, returns `(N, 15)` ndarray |
| `REG_NAMES` | tuple of the 15 SH mode names |
| `J70` | callable Jacchia-70 lookup (`rho = J70()(alt_km, Tinf_K)`) |
| `calc_Tx(Tinf)` | Roberts (1971) composition formula |

Provenance / diagnostics on a loaded model:

```python
m.schema_version   # e.g. '0.8.0'
m.tle_edge_utc     # str
m.generated_utc    # str
m.altitude_km      # ndarray
m.time_utc         # list[str] — solver bin timestamps
m.recipe           # human-readable formula
m.provenance()     # dict summary
```

---

## Testing

```bash
pip install ".[test]"
pytest -q
```

---

## Accuracy caveats

- `ephem.py` uses a GMST-only TEME→ECEF rotation (IAU-1982, sub-arcsecond).
  No polar-motion / nutation corrections.  Adequate for drag; not adequate
  for centimetre-level positioning.
- The spatial climatology is a monthly-scale average pattern.  Storm-time
  and hemispheric asymmetries at the ~10-30% level are not resolved by
  the 15-mode / annual Fourier form.
- `Tinf(t, lat, lon)` is derived through the local `rho_400` back-map,
  giving a physically self-consistent vertical profile but *not* a proper
  local-time temperature field.  For research use, prefer models that carry
  an explicit `Tinf(lat, LST)` if available.

---

## References

- Gershman (2026), *Enabling Near-Real-Time Thermospheric Density
  Monitoring*, NASA-TM/20260006171.
- Jacchia, L. G. (1970), *New Static Models of the Thermosphere and
  Exosphere with Empirical Temperature Profiles*, SAO Special Report 313.
- Roberts, C. E. (1971), *An analytic model for upper atmosphere
  densities based upon Jacchia's 1970 models*, Cel. Mech., 4, 368-377.
- Vallado, D. A. (2013), *Fundamentals of Astrodynamics and Applications*,
  4th ed., Microcosm Press (for TEME/ECEF conversions).

## License

MIT.
