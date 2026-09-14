"""15 spherical-harmonic modes (l ≤ 3) at 400 km lat/LST.

Convention:
    HA  = 2π (lst_h - 12) / 24         (hour angle from local noon)
    phi = deg2rad(lat_deg)
"""
from __future__ import annotations
import numpy as np

REG_NAMES = ('Y10', 'Y11c', 'Y11s',
             'Y20', 'Y21c', 'Y21s', 'Y22c', 'Y22s',
             'Y30', 'Y31c', 'Y31s',
             'Y32c', 'Y32s', 'Y33c', 'Y33s')


def Y_modes(lat_deg, lst_h):
    """Return (N, 15) matrix of Y_l evaluated at each (lat, lst) point.

    lat_deg, lst_h can be scalars or 1D arrays of equal length.
    Column order matches REG_NAMES.
    """
    lat_deg = np.atleast_1d(np.asarray(lat_deg, float))
    lst_h   = np.atleast_1d(np.asarray(lst_h,   float))
    HA  = 2.0 * np.pi * (lst_h - 12.0) / 24.0
    phi = np.deg2rad(lat_deg)
    sp, cp = np.sin(phi), np.cos(phi)
    sp2, cp2, cp3 = sp*sp, cp*cp, cp*cp*cp
    cH, sH   = np.cos(HA),   np.sin(HA)
    c2H, s2H = np.cos(2*HA), np.sin(2*HA)
    c3H, s3H = np.cos(3*HA), np.sin(3*HA)
    Y = np.stack([
        sp,                                                  # Y10
        cH*cp, sH*cp,                                        # Y11c, Y11s
        0.5*(3*sp2 - 1),                                     # Y20
        sp*cp*cH, sp*cp*sH,                                  # Y21c, Y21s
        cp2*c2H, cp2*s2H,                                    # Y22c, Y22s
        0.5*sp*(5*sp2 - 3),                                  # Y30
        1.5*cp*(5*sp2 - 1)*cH, 1.5*cp*(5*sp2 - 1)*sH,        # Y31c, Y31s
        15.0*sp*cp2*c2H, 15.0*sp*cp2*s2H,                    # Y32c, Y32s
        15.0*cp3*c3H, 15.0*cp3*s3H,                          # Y33c, Y33s
    ], axis=-1)
    return Y
