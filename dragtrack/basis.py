"""15 spherical-harmonic modes (l <= 3) at 400 km lat / LST.

The basis is a fixed set of un-normalized real associated Legendre-times-
Fourier products in ``(sin(lat), HA)`` where ``HA = 2*pi*(lst_h - 12)/24``
is the hour angle from local noon.  It matches exactly the basis used by
the DragTrack Ops pipeline to fit the ``beta_frac`` climatology
coefficients, so **do not renormalize**.

Column order (``REG_NAMES``)::

    l=1:  Y10, Y11c, Y11s
    l=2:  Y20, Y21c, Y21s, Y22c, Y22s
    l=3:  Y30, Y31c, Y31s, Y32c, Y32s, Y33c, Y33s

Convention::

    HA  = 2*pi * (lst_h - 12) / 24     (hour angle from local noon)
    phi = deg2rad(lat_deg)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

REG_NAMES: Tuple[str, ...] = (
    'Y10', 'Y11c', 'Y11s',
    'Y20', 'Y21c', 'Y21s', 'Y22c', 'Y22s',
    'Y30', 'Y31c', 'Y31s',
    'Y32c', 'Y32s', 'Y33c', 'Y33s',
)


def Y_modes(lat_deg: ArrayLike, lst_h: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the 15 spherical-harmonic modes.

    Parameters
    ----------
    lat_deg : array_like
        Geodetic latitude in degrees.  Scalar or 1D array.
    lst_h : array_like
        Local solar time in hours (0-24).  Same shape as ``lat_deg``.

    Returns
    -------
    ndarray, shape ``(N, 15)``
        Column order matches :data:`REG_NAMES`.  ``N = len(lat_deg)``
        after broadcasting to at least 1D.
    """
    lat = np.atleast_1d(np.asarray(lat_deg, float))
    lst = np.atleast_1d(np.asarray(lst_h, float))
    if lat.shape != lst.shape:
        lat, lst = np.broadcast_arrays(lat, lst)
    HA = 2.0 * np.pi * (lst - 12.0) / 24.0
    phi = np.deg2rad(lat)
    sp, cp = np.sin(phi), np.cos(phi)
    sp2 = sp * sp
    cp2 = cp * cp
    cp3 = cp2 * cp
    cH, sH = np.cos(HA), np.sin(HA)
    c2H, s2H = np.cos(2 * HA), np.sin(2 * HA)
    c3H, s3H = np.cos(3 * HA), np.sin(3 * HA)
    Y = np.stack([
        sp,                                              # Y10
        cH * cp,        sH * cp,                         # Y11c, Y11s
        0.5 * (3 * sp2 - 1),                             # Y20
        sp * cp * cH,   sp * cp * sH,                    # Y21c, Y21s
        cp2 * c2H,      cp2 * s2H,                       # Y22c, Y22s
        0.5 * sp * (5 * sp2 - 3),                        # Y30
        1.5 * cp * (5 * sp2 - 1) * cH,
        1.5 * cp * (5 * sp2 - 1) * sH,                   # Y31c, Y31s
        15.0 * sp * cp2 * c2H, 15.0 * sp * cp2 * s2H,    # Y32c, Y32s
        15.0 * cp3 * c3H,      15.0 * cp3 * s3H,         # Y33c, Y33s
    ], axis=-1)
    return Y
