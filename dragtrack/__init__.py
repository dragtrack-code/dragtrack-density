"""dragtrack — atmospheric density from DragTrack JSON products.

Point-density access: rho(t, lat, lon, alt) using
  1) subdaily solver ρ_bg + (mt, bt) from `dragtrack_rho_realtime.json`
  2) SH spatial pattern β_l (fractional) from
     `dragtrack_spatial_beta_realtime.json` or the static
     `dragtrack_spatial_clim_analytical.json`
  3) J70 lookup for vertical scaling (packaged asset).

Quick start:
    >>> from dragtrack import DragTrackDensity
    >>> m = DragTrackDensity.from_files(
    ...     rho_realtime='dragtrack_rho_realtime.json',
    ...     spatial='dragtrack_spatial_beta_realtime.json',
    ... )
    >>> rho = m.rho('2026-01-15T12:00Z', lat_deg=45, lon_deg=0, alt_km=450)
"""
from .model import DragTrackDensity
from .basis import Y_modes, REG_NAMES

__all__ = ['DragTrackDensity', 'Y_modes', 'REG_NAMES']
__version__ = '0.1.0'
