"""Ephemeris helpers: propagate TLE / Kepler state → geodetic (lat, lon, alt).

Wraps the standard `sgp4` library.  Coordinates are converted from
SGP4-TEME to ECEF via a simple GMST rotation, then to geodetic
lat/lon/alt using the WGS72 ellipsoid.
"""
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np

try:
    from sgp4.api import Satrec, WGS72, jday
    from sgp4.conveniences import jday_datetime
except ImportError:  # pragma: no cover
    Satrec = None
    WGS72 = None
    jday = None
    jday_datetime = None

# WGS72 constants (match SGP4)
_RE_KM = 6378.135
_FLAT  = 1.0 / 298.26


def _ensure_sgp4():
    if Satrec is None:
        raise ImportError(
            "The `sgp4` package is required for ephemeris propagation. "
            "Install with: pip install sgp4"
        )


def _parse_time(t):
    if isinstance(t, datetime):
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    if isinstance(t, str):
        return datetime.fromisoformat(t.replace('Z', '+00:00'))
    raise TypeError(f'Cannot parse time {t!r}')


def _gmst_rad(jd_ut1):
    """Simple GMST at 0h UT1 (IAU 1982 approximation, sub-arcsec)."""
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_s = (67310.54841 + (876600.0*3600 + 8640184.812866)*T
              + 0.093104*T**2 - 6.2e-6*T**3)
    gmst_s = gmst_s % 86400.0
    return (gmst_s / 86400.0) * 2 * np.pi


def _teme_to_ecef(r_teme, gmst_rad):
    """Rotate TEME position (km) → ECEF (km) by −GMST about Z."""
    c, s = np.cos(gmst_rad), np.sin(gmst_rad)
    R = np.array([[ c, s, 0.0],
                  [-s, c, 0.0],
                  [0.0, 0.0, 1.0]])
    return R @ r_teme


def _ecef_to_geodetic(x, y, z):
    """WGS72 geodetic lat/lon/alt (deg, deg, km) from ECEF (km)."""
    a = _RE_KM
    f = _FLAT
    e2 = 2*f - f*f
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    # Bowring iterative
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + alt)))
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    return np.rad2deg(lat), np.rad2deg(lon), alt


def _propagate_satrec(sat, times):
    """Given a Satrec + iterable of times (str/datetime), return (t, lat, lon, alt)."""
    _ensure_sgp4()
    times = list(times)
    t_dt = [_parse_time(t) for t in times]
    lat = np.empty(len(t_dt)); lon = np.empty(len(t_dt)); alt = np.empty(len(t_dt))
    for i, dt in enumerate(t_dt):
        jd, fr = jday_datetime(dt.replace(tzinfo=None))
        e, r_teme, _ = sat.sgp4(jd, fr)
        if e != 0:
            lat[i] = lon[i] = alt[i] = np.nan
            continue
        gmst = _gmst_rad(jd + fr)
        r_ecef = _teme_to_ecef(np.asarray(r_teme, float), gmst)
        la, lo, al = _ecef_to_geodetic(r_ecef[0], r_ecef[1], r_ecef[2])
        lat[i] = la; lon[i] = ((lo + 180.0) % 360.0) - 180.0; alt[i] = al
    return times, lat, lon, alt


def propagate_tle(l1: str, l2: str, times):
    """Propagate a TLE (2 lines) to each time; return (times, lat, lon, alt).

    Times may be ISO strings or datetimes.  Latitude/longitude in degrees,
    altitude in km (WGS72 ellipsoid).
    """
    _ensure_sgp4()
    sat = Satrec.twoline2rv(l1, l2)
    return _propagate_satrec(sat, times)


def propagate_kepler(*, a_km: float, e: float, inc_deg: float,
                     raan_deg: float, argp_deg: float, m0_deg: float,
                     bstar: float, epoch_utc, times):
    """Build a Satrec from Kepler elements at `epoch_utc` and propagate.

    a_km:     semi-major axis (km)
    e:        eccentricity
    inc_deg:  inclination (deg)
    raan_deg: right ascension of ascending node (deg)
    argp_deg: argument of perigee (deg)
    m0_deg:   mean anomaly at epoch (deg)
    bstar:    SGP4 drag coefficient (per Earth radii); 0.0 if unknown
    epoch_utc: ISO string or datetime
    """
    _ensure_sgp4()
    from math import sqrt, pi
    MU_KM = 3.986004418e5  # km³/s²
    n_mean_rev_per_day = (86400.0 / (2 * pi)) * sqrt(MU_KM / a_km**3)
    ep = _parse_time(epoch_utc)
    # SGP4 epoch = days since 1949-12-31 00:00 UT
    _base_jd = 2433281.5
    jd, fr = jday(ep.year, ep.month, ep.day,
                  ep.hour, ep.minute, ep.second + ep.microsecond/1e6)
    dsat = (jd + fr) - _base_jd
    sat = Satrec()
    sat.sgp4init(WGS72, 'i', 0, dsat, float(bstar), 0.0, 0.0,
                 float(e), np.deg2rad(argp_deg), np.deg2rad(inc_deg),
                 np.deg2rad(m0_deg),
                 n_mean_rev_per_day / (86400.0 / (2 * pi)),  # rad/min → per-min
                 np.deg2rad(raan_deg))
    return _propagate_satrec(sat, times)
