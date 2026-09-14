"""Ephemeris helpers: propagate TLE / Kepler state to geodetic (lat, lon, alt).

Wraps the standard :mod:`sgp4` library.  Coordinates are converted from
SGP4-TEME to ECEF via a GMST-only rotation about the Z axis (IAU-1982
approximation, sub-arcsecond accuracy), then to geodetic lat/lon/alt using
the WGS72 ellipsoid.

Accuracy caveats
----------------
- No polar-motion / nutation corrections are applied.  For short
  propagations (< days) and drag applications (~100 m altitude accuracy
  is plenty) this is fine.  For sub-cm accuracy or long-arc reproduction,
  use a dedicated reduction library (e.g., ``skyfield`` or SOFA).
- ``propagate_kepler`` initialises SGP4 with zero perturbation terms
  besides drag (BSTAR); output diverges from a true Keplerian orbit
  because SGP4 internally applies gravity harmonics.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

try:
    from sgp4.api import Satrec, WGS72, jday
    from sgp4.conveniences import jday_datetime
except ImportError:  # pragma: no cover - optional dependency
    Satrec = None
    WGS72 = None
    jday = None
    jday_datetime = None

# WGS72 constants (match SGP4).
_RE_KM = 6378.135
_FLAT = 1.0 / 298.26

TimeLike = Union[str, datetime]


def _ensure_sgp4() -> None:
    if Satrec is None:
        raise ImportError(
            "The `sgp4` package is required for ephemeris propagation. "
            "Install with:  pip install sgp4"
        )


def _parse_time(t: TimeLike) -> datetime:
    if isinstance(t, datetime):
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    if isinstance(t, str):
        return datetime.fromisoformat(t.replace('Z', '+00:00'))
    raise TypeError(f'Cannot parse time {t!r}')


def _gmst_rad(jd_ut1: NDArray[np.float64]) -> NDArray[np.float64]:
    """Greenwich Mean Sidereal Time (rad) at ``jd_ut1``, IAU-1982.

    Accepts scalar or array of Julian Dates.
    """
    jd = np.asarray(jd_ut1, float)
    T = (jd - 2451545.0) / 36525.0
    gmst_s = (67310.54841
              + (876600.0 * 3600 + 8640184.812866) * T
              + 0.093104 * T ** 2
              - 6.2e-6 * T ** 3)
    gmst_s = np.mod(gmst_s, 86400.0)
    return (gmst_s / 86400.0) * 2.0 * np.pi


def _teme_to_ecef_batch(r_teme: NDArray[np.float64],
                        gmst: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate a batch of TEME positions (N,3) by -GMST(N) about Z."""
    c = np.cos(gmst); s = np.sin(gmst)
    x = c * r_teme[:, 0] + s * r_teme[:, 1]
    y = -s * r_teme[:, 0] + c * r_teme[:, 1]
    z = r_teme[:, 2]
    return np.stack([x, y, z], axis=1)


def _ecef_to_geodetic(x, y, z) -> Tuple[NDArray, NDArray, NDArray]:
    """Vectorized WGS72 geodetic conversion — Bowring 5-iter."""
    a = _RE_KM
    f = _FLAT
    e2 = 2 * f - f * f
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat ** 2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + alt)))
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e2 * sin_lat ** 2)
    alt = p / np.cos(lat) - N
    return np.rad2deg(lat), np.rad2deg(lon), alt


def _propagate_satrec(sat, times: Iterable[TimeLike]):
    """Given a Satrec + iterable of times, return ``(times, lat, lon, alt)``."""
    _ensure_sgp4()
    t_list: List[TimeLike] = list(times)
    n = len(t_list)
    t_dt = [_parse_time(t) for t in t_list]
    jd_arr = np.empty(n); fr_arr = np.empty(n)
    for i, dt in enumerate(t_dt):
        jd_arr[i], fr_arr[i] = jday_datetime(dt.replace(tzinfo=None))
    # sgp4_array — vectorized propagation over the whole batch.
    if hasattr(sat, 'sgp4_array'):
        errs, r_teme, _v = sat.sgp4_array(jd_arr, fr_arr)
        r_teme = np.asarray(r_teme, float)
        errs = np.asarray(errs, int)
    else:  # pragma: no cover - defensive fallback
        r_teme = np.empty((n, 3))
        errs = np.zeros(n, int)
        for i in range(n):
            e, r, _ = sat.sgp4(jd_arr[i], fr_arr[i])
            errs[i] = e
            r_teme[i] = r if e == 0 else np.nan
    gmst = _gmst_rad(jd_arr + fr_arr)
    r_ecef = _teme_to_ecef_batch(r_teme, gmst)
    lat, lon, alt = _ecef_to_geodetic(r_ecef[:, 0], r_ecef[:, 1], r_ecef[:, 2])
    # Mask errors → NaN
    bad = errs != 0
    if np.any(bad):
        lat[bad] = np.nan; lon[bad] = np.nan; alt[bad] = np.nan
    # Wrap longitude into [-180, 180)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return t_list, lat, lon, alt


def propagate_tle(l1: str, l2: str, times: Iterable[TimeLike]):
    """Propagate a TLE (two lines) to each time.

    Parameters
    ----------
    l1, l2 : str
        Standard Two-Line Element strings.
    times : iterable of str/datetime
        Sample times, UTC.

    Returns
    -------
    tuple
        ``(times, lat_deg, lon_deg, alt_km)``.  Longitude is wrapped to
        ``[-180, 180)``.  Altitude is above the WGS72 ellipsoid.
    """
    _ensure_sgp4()
    sat = Satrec.twoline2rv(l1, l2)
    return _propagate_satrec(sat, times)


def propagate_kepler(*, a_km: float, e: float, inc_deg: float,
                     raan_deg: float, argp_deg: float, m0_deg: float,
                     bstar: float, epoch_utc: TimeLike,
                     times: Iterable[TimeLike]):
    """Build a Satrec from Kepler elements at ``epoch_utc`` and propagate.

    Parameters
    ----------
    a_km : float
        Semi-major axis, km.
    e : float
        Eccentricity, 0..1.
    inc_deg, raan_deg, argp_deg, m0_deg : float
        Inclination, RAAN, argument of perigee, and mean anomaly at epoch
        (degrees).
    bstar : float
        SGP4 drag term (Earth-radii^-1).  Use 0.0 if unknown.
    epoch_utc : str or datetime
        Element-set epoch, UTC.
    times : iterable
        Sample times.

    Returns
    -------
    tuple
        ``(times, lat_deg, lon_deg, alt_km)``.
    """
    _ensure_sgp4()
    from math import pi, sqrt
    MU_KM = 3.986004418e5  # km^3 / s^2
    n_mean_rev_per_day = (86400.0 / (2 * pi)) * sqrt(MU_KM / a_km ** 3)
    ep = _parse_time(epoch_utc)
    _base_jd = 2433281.5  # 1949-12-31 00:00 UT
    jd, fr = jday(
        ep.year, ep.month, ep.day,
        ep.hour, ep.minute, ep.second + ep.microsecond / 1e6,
    )
    dsat = (jd + fr) - _base_jd
    sat = Satrec()
    sat.sgp4init(
        WGS72, 'i', 0, dsat, float(bstar), 0.0, 0.0,
        float(e), np.deg2rad(argp_deg), np.deg2rad(inc_deg),
        np.deg2rad(m0_deg),
        n_mean_rev_per_day / (86400.0 / (2 * pi)),   # rad/min
        np.deg2rad(raan_deg),
    )
    return _propagate_satrec(sat, times)
