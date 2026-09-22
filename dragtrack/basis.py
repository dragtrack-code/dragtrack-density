"""16-mode spherical-harmonic basis for the horizontal Tinf pattern.

Used by :class:`dragtrack.DragTrackDensity` internally.  Most users don't
need to touch this module.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

REG_NAMES: Tuple[str, ...] = (
    'Y00',
    'Y10', 'Y11c', 'Y11s',
    'Y20', 'Y21c', 'Y21s', 'Y22c', 'Y22s',
    'Y30', 'Y31c', 'Y31s',
    'Y32c', 'Y32s', 'Y33c', 'Y33s',
)


def Y_modes(lat_deg: ArrayLike, lst_h: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the 16 spherical-harmonic modes at ``(lat_deg, lst_h)``.

    Parameters
    ----------
    lat_deg : array_like
        Geodetic latitude in degrees.  Broadcast with ``lst_h``.
    lst_h : array_like
        Local solar time in hours, ``[0, 24)``.

    Returns
    -------
    ndarray, shape ``(*broadcast_shape, 16)``
        Column order matches :data:`REG_NAMES` (Y00, Y10, ..., Y33s).
    """
    lat = np.asarray(lat_deg, float)
    lst = np.asarray(lst_h, float)
    if lat.shape != lst.shape:
        lat, lst = np.broadcast_arrays(lat, lst)
    HA = 2.0 * np.pi * (lst - 12.0) / 24.0
    phi = np.deg2rad(lat)
    sp, cp = np.sin(phi), np.cos(phi)
    sp2, cp2 = sp * sp, cp * cp
    cp3 = cp2 * cp
    cH, sH = np.cos(HA), np.sin(HA)
    c2H, s2H = np.cos(2 * HA), np.sin(2 * HA)
    c3H, s3H = np.cos(3 * HA), np.sin(3 * HA)
    return np.stack([
        np.ones_like(phi),                          # Y00
        sp,                                         # Y10
        cp * cH,        cp * sH,                    # Y11c, Y11s
        0.5 * (3 * sp2 - 1.0),                      # Y20
        sp * cp * cH,   sp * cp * sH,               # Y21c, Y21s
        cp2 * c2H,      cp2 * s2H,                  # Y22c, Y22s
        0.5 * (5 * sp2 * sp - 3 * sp),              # Y30
        0.5 * cp * (5 * sp2 - 1.0) * cH,            # Y31c
        0.5 * cp * (5 * sp2 - 1.0) * sH,            # Y31s
        sp * cp2 * c2H, sp * cp2 * s2H,             # Y32c, Y32s
        cp3 * c3H,      cp3 * s3H,                  # Y33c, Y33s
    ], axis=-1)
