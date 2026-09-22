"""dragtrack — atmospheric density ``rho(t, lat, lon, alt)`` in a few lines.

>>> from dragtrack import DragTrackDensity
>>> m = DragTrackDensity.from_files('dragtrack_rho_realtime.json',
...                                  'dragtrack_rho_forecast.json')
>>> rho = m.rho('2026-09-13T12:00:00Z', lat_deg=45, lon_deg=0, alt_km=450)

See the README for coordinate conventions, scenarios, and the CLI.
"""
from __future__ import annotations

from .model import DragTrackDensity, SCENARIOS

__all__ = ['DragTrackDensity', 'SCENARIOS', '__version__']

__version__ = '0.3.0'
