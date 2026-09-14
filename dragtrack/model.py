"""DragTrackDensity — point ρ from JSON products + J70 LUT.

Data sources (all JSON):
  1) `dragtrack_spatial_clim_analytical.json` — static Fourier β_frac(doy)
  2) `dragtrack_rho_realtime.json`.solver     — ρ_bg, Tinf_bg, mt, bt per bin
  3) J70 LUT (packaged, gzip-base64)

Reconstruction:
  ρ_400(t, lat, lst) = ρ_bg(t) × (1 + Σ β_l(doy) · Y_l(lat, lst))
  Tinf(t, lat, lst)  = mt(day)·ρ_400 + bt(day)
  ρ(alt)             = ρ_400 × J70(alt, Tinf) / J70(400, Tinf)
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

from .basis import Y_modes, REG_NAMES
from .j70 import J70


def _parse_time(t):
    if isinstance(t, (float, int)):
        return datetime(1970, 1, 1, tzinfo=timezone.utc).fromtimestamp(
            float(t) * 86400.0)
    if isinstance(t, str):
        return datetime.fromisoformat(t.replace('Z', '+00:00'))
    if isinstance(t, datetime):
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    raise TypeError(f'Cannot parse time {t!r}')


def _iso_to_mpl(t):
    dt = _parse_time(t)
    return (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0


class _Clim:
    """Static Fourier β_frac(doy) reconstruction from analytical JSON."""
    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = Path(__file__).parent / 'data' / 'dragtrack_spatial_clim_analytical.json'
        with open(path, 'r') as f:
            d = json.load(f)
        self.nh = int(d['basis_definition']['n_harmonics'])
        self.names = tuple(d['basis_definition']['Y_defs'].keys())
        self._coefs = d['coefficients']

    def beta_at_doy(self, doy):
        """Return (N, NM) fractional β matrix for each DOY in `doy`."""
        doy = np.atleast_1d(np.asarray(doy, float))
        w = 2 * np.pi * (doy - 1) / 365.25
        out = np.empty((len(doy), len(REG_NAMES)))
        for l, name in enumerate(REG_NAMES):
            if name not in self._coefs:
                out[:, l] = 0.0
                continue
            c = self._coefs[name]
            v = np.full_like(w, c['a0'])
            for k in range(1, self.nh + 1):
                v = v + c[f'a{k}'] * np.cos(k * w) + c[f'b{k}'] * np.sin(k * w)
            out[:, l] = v
        return out


class DragTrackDensity:
    """Load JSON products, expose rho(t, lat, lon, alt)."""

    def __init__(self, *, rho_realtime: dict,
                 clim: _Clim | None = None,
                 j70: J70 | None = None):
        self.j70  = j70 or J70()
        self.clim = clim or _Clim()

        blk = rho_realtime['solver']
        self._t = np.array([_iso_to_mpl(t) for t in blk['time_utc']])
        alts = np.asarray(rho_realtime['altitude_km'], float)
        i400 = int(np.argmin(np.abs(alts - 400.0)))
        self._rho400 = np.array([r[i400] for r in blk['rho']])
        self._Tinf_bg = np.array(blk['J70_Tinf_K'], float)
        # mt, bt per bin (linear T∞ = mt·ρ_400 + bt).  Required for new schema
        # (2026-09-13+); consumers of older rho_realtime JSONs will hit a
        # KeyError — clear error surface is fine.
        if 'T_J70_m' not in blk or 'T_J70_b' not in blk:
            raise KeyError(
                "rho_realtime JSON missing 'T_J70_m'/'T_J70_b' in solver block; "
                "regenerate with dragtrack_ops ≥ 2026-09-13 or pass legacy=True."
            )
        self._mt = np.array(blk['T_J70_m'], float)
        self._bt = np.array(blk['T_J70_b'], float)

    @classmethod
    def from_files(cls, rho_realtime, clim=None, j70_lut=None):
        with open(rho_realtime, 'r') as f: rt = json.load(f)
        c = _Clim(clim) if clim else None
        j = J70(j70_lut) if j70_lut else None
        return cls(rho_realtime=rt, clim=c, j70=j)

    def _interp(self, t_mpl, arr):
        return np.interp(t_mpl, self._t, arr, left=arr[0], right=arr[-1])

    # ── Convenience: propagate a TLE and get ρ along the orbit ────────
    def rho_along_tle(self, l1: str, l2: str, times):
        """Propagate a TLE to each time; return (times, lat, lon, alt, rho)."""
        from .ephem import propagate_tle
        t_arr, lat, lon, alt = propagate_tle(l1, l2, times)
        rho = self.rho(t_arr, lat, lon, alt)
        return t_arr, lat, lon, alt, rho

    def rho_along_kepler(self, *, a_km, e, inc_deg, raan_deg, argp_deg,
                         m0_deg, bstar, epoch_utc, times):
        """Propagate Kepler state at epoch to `times`; return orbit + ρ."""
        from .ephem import propagate_kepler
        t_arr, lat, lon, alt = propagate_kepler(
            a_km=a_km, e=e, inc_deg=inc_deg, raan_deg=raan_deg,
            argp_deg=argp_deg, m0_deg=m0_deg, bstar=bstar,
            epoch_utc=epoch_utc, times=times)
        rho = self.rho(t_arr, lat, lon, alt)
        return t_arr, lat, lon, alt, rho

    def rho(self, t, lat_deg, lon_deg, alt_km):
        t_arr = np.atleast_1d(t)
        lat = np.atleast_1d(np.asarray(lat_deg, float))
        lon = np.atleast_1d(np.asarray(lon_deg, float))
        alt = np.atleast_1d(np.asarray(alt_km,  float))
        t_mpl = np.array([_iso_to_mpl(x) for x in t_arr])
        dts = [_parse_time(x) for x in t_arr]
        ut_h = np.array([d.hour + d.minute/60 + d.second/3600 for d in dts])
        lst_h = (ut_h + lon / 15.0) % 24
        doy = np.array([d.timetuple().tm_yday for d in dts])
        # Backbone: ρ_bg + mt/bt via time-interp on solver bins
        rho_bg = self._interp(t_mpl, self._rho400)
        mt = self._interp(t_mpl, self._mt)
        bt = self._interp(t_mpl, self._bt)
        # Spatial β from analytical clim + Y basis
        beta_frac = self.clim.beta_at_doy(doy)   # (N, NM)
        Y = Y_modes(lat, lst_h)                  # (N, NM)
        frac = (beta_frac * Y).sum(axis=1)
        rho_400_pt = np.maximum(rho_bg * (1.0 + frac), 1e-30)
        Tinf_pt = mt * rho_400_pt + bt
        r_alt = self.j70(alt, Tinf_pt)
        r_400 = self.j70(np.full_like(alt, 400.0), Tinf_pt)
        out = rho_400_pt * r_alt / r_400
        if np.isscalar(t) and np.isscalar(lat_deg):
            return float(out[0])
        return out
