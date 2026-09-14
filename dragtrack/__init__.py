"""dragtrack — atmospheric density from DragTrack JSON products.

This package reconstructs point-density ``rho(t, lat, lon, alt)`` from JSON
data products emitted by the DragTrack Ops pipeline plus two bundled static
assets (a 15 spherical-harmonic Fourier climatology and a Jacchia 1970 LUT).

Reconstruction summary
----------------------
Given a runtime ``dragtrack_rho_realtime.json`` (``solver`` block carries
``time_utc``, ``rho[time, alt_grid]``, ``J70_Tinf_K``, ``T_J70_m``,
``T_J70_b``) and the packaged clim+J70 assets::

    doy      = day_of_year(t)
    beta_l   = Fourier eval at doy for 15 modes         (clim JSON)
    Y_l      = 15 SH basis @ (lat, lst)                 lst = UT + lon/15
    rho_bg   = interp(rho_realtime.solver.rho @ 400km, t)
    mt, bt   = interp(T_J70_m, T_J70_b, t)              per-day linear
    rho_400  = rho_bg * (1 + Σ_l β_l · Y_l)             multiplicative
    Tinf_pt  = mt · rho_400 + bt
    rho_out  = rho_400 * J70(alt, Tinf_pt) / J70(400, Tinf_pt)

Quick start
-----------
>>> from dragtrack import DragTrackDensity
>>> m = DragTrackDensity.from_files('dragtrack_rho_realtime.json')
>>> rho = m.rho('2026-01-15T12:00:00Z', lat_deg=45, lon_deg=0, alt_km=450)

References
----------
- Gershman (2026), "Enabling Near-Real-Time Thermospheric Density
  Monitoring", NASA-TM/20260006171.
- Jacchia, L.G. (1970), "New Static Models of the Thermosphere...",
  Smithsonian Astrophysical Observatory Special Report 313.
- Roberts, C.E. (1971), "An analytic model for upper atmosphere densities
  based upon Jacchia's 1970 models", Celestial Mechanics, 4, 368-377.
"""
from __future__ import annotations

from .basis import Y_modes, REG_NAMES
from .j70 import J70, calc_Tx
from .model import DragTrackDensity

__all__ = [
    'DragTrackDensity',
    'J70',
    'Y_modes',
    'REG_NAMES',
    'calc_Tx',
    '__version__',
]

__version__ = '0.1.0'
