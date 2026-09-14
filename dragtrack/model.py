"""DragTrackDensity — point ``rho`` from JSON products + J70 LUT.

Data sources (all JSON)
-----------------------
1. ``dragtrack_spatial_clim_analytical.json`` — static Fourier
   ``beta_frac(doy)`` climatology, 6 harmonics x 15 SH modes.
   Packaged with the library.
2. ``dragtrack_rho_realtime.json`` — runtime product from the
   DragTrack Ops pipeline; ``solver`` block carries
   ``time_utc``, ``rho[time, alt_grid]``, ``J70_Tinf_K``,
   ``T_J70_m``, ``T_J70_b``, ``bin_days``.
3. Jacchia-70 LUT — packaged gzip-base64 3D grid on
   ``(z, Tinf, Tx)``.

Reconstruction
--------------
::

    rho_400(t, lat, lst) = rho_bg(t) * (1 + sum_l beta_l(doy) * Y_l(lat, lst))
    Tinf(t, lat, lst)    = mt(t) * rho_400 + bt(t)
    rho(alt)             = rho_400 * J70(alt, Tinf) / J70(400, Tinf)

Where ``lst = UT + lon/15`` (hours, mod 24) and ``doy`` is the day-of-year
(1..366) of the UTC timestamp.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union
from urllib.request import urlopen

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import REG_NAMES, Y_modes
from .j70 import J70

PathLike = Union[str, Path]
TimeLike = Union[str, datetime, float, int, np.datetime64]

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


# ─── Time helpers ────────────────────────────────────────────────────────

def _parse_time(t: TimeLike) -> datetime:
    """Parse ``t`` into a timezone-aware UTC ``datetime``."""
    if isinstance(t, datetime):
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    if isinstance(t, str):
        # Accept trailing 'Z' as UTC.
        return datetime.fromisoformat(t.replace('Z', '+00:00'))
    if isinstance(t, np.datetime64):
        # Convert to POSIX seconds then to aware datetime.
        secs = (t - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        return datetime.fromtimestamp(float(secs), tz=timezone.utc)
    if isinstance(t, (float, int, np.floating, np.integer)):
        # Treat as days since 1970-01-01 UTC (matches _to_days).
        return _EPOCH_UTC + (datetime.fromtimestamp(float(t) * 86400.0,
                                                    tz=timezone.utc) - _EPOCH_UTC)
    raise TypeError(f'Cannot parse time {t!r}')


def _to_days(t: TimeLike) -> float:
    """Days since 1970-01-01 UTC (fractional)."""
    dt = _parse_time(t)
    return (dt - _EPOCH_UTC).total_seconds() / 86400.0


def _vec_time_features(times: Iterable[TimeLike]):
    """Return arrays ``(t_days, ut_hours, doy)`` for a sequence of times."""
    dts = [_parse_time(t) for t in times]
    n = len(dts)
    t_days = np.empty(n, float)
    ut_h = np.empty(n, float)
    doy = np.empty(n, int)
    for i, d in enumerate(dts):
        t_days[i] = (d - _EPOCH_UTC).total_seconds() / 86400.0
        ut_h[i] = d.hour + d.minute / 60.0 + d.second / 3600.0 \
            + d.microsecond / 3.6e9
        doy[i] = d.timetuple().tm_yday
    return t_days, ut_h, doy


# ─── Climatology ─────────────────────────────────────────────────────────

class _Clim:
    """Static Fourier ``beta_frac(doy)`` reconstruction from the analytical JSON.

    The JSON stores per-mode coefficients ``{a0, a1..aH, b1..bH}`` fitted at
    fundamental angular frequency ``omega = 2*pi/365.25``.
    """

    def __init__(self, path: PathLike | None = None):
        if path is None:
            path = Path(__file__).parent / 'data' / 'dragtrack_spatial_clim_analytical.json'
        self.path = Path(path)
        with open(self.path, 'r') as f:
            d = json.load(f)
        self.nh = int(d['basis_definition']['n_harmonics'])
        self.names = tuple(d['basis_definition']['Y_defs'].keys())
        self._coefs: Mapping[str, Mapping[str, float]] = d['coefficients']
        # Pre-pack coefficient tables for fast vectorized eval.
        # a[k, l] = a_k for mode l ; b[k, l] = b_k for mode l ; a[0] = a0.
        NM = len(REG_NAMES)
        self._a = np.zeros((self.nh + 1, NM))
        self._b = np.zeros((self.nh + 1, NM))
        for l, name in enumerate(REG_NAMES):
            if name not in self._coefs:
                continue
            c = self._coefs[name]
            self._a[0, l] = c.get('a0', 0.0)
            for k in range(1, self.nh + 1):
                self._a[k, l] = c.get(f'a{k}', 0.0)
                self._b[k, l] = c.get(f'b{k}', 0.0)

    def beta_at_doy(self, doy: ArrayLike) -> NDArray[np.float64]:
        """Return ``(N, 15)`` fractional-beta matrix for each DOY value."""
        doy = np.atleast_1d(np.asarray(doy, float))
        w = 2.0 * np.pi * (doy - 1) / 365.25            # (N,)
        # Build cos/sin harmonic stacks (H+1, N) with row 0 = ones for a0.
        H = self.nh
        k = np.arange(H + 1)[:, None]                    # (H+1, 1)
        kw = k * w[None, :]                              # (H+1, N)
        cosk = np.cos(kw); cosk[0] = 1.0                 # a0 rides on cos row-0
        sink = np.sin(kw); sink[0] = 0.0
        # (H+1, N) x (H+1, NM) → (N, NM)
        return cosk.T @ self._a + sink.T @ self._b


# ─── Main model ──────────────────────────────────────────────────────────

class DragTrackDensity:
    """Load DragTrack JSON products and expose ``rho(t, lat, lon, alt)``.

    Parameters
    ----------
    rho_realtime : dict
        Parsed contents of a ``dragtrack_rho_realtime.json`` file (schema
        >= 0.8 with ``solver.T_J70_m`` and ``solver.T_J70_b``).
    clim : :class:`_Clim`, optional
        Fourier climatology.  Defaults to the packaged asset.
    j70 : :class:`J70`, optional
        J70 lookup.  Defaults to the packaged asset.

    See Also
    --------
    DragTrackDensity.from_files, DragTrackDensity.from_url
    """

    def __init__(
        self,
        *,
        rho_realtime: Mapping[str, Any],
        clim: _Clim | None = None,
        j70: J70 | None = None,
    ) -> None:
        self.j70 = j70 or J70()
        self.clim = clim or _Clim()
        self._raw = dict(rho_realtime)  # shallow copy

        if 'solver' not in rho_realtime:
            raise KeyError("rho_realtime JSON has no 'solver' block")
        blk = rho_realtime['solver']

        alts = np.asarray(rho_realtime.get('altitude_km', []), float)
        if alts.size == 0:
            raise KeyError("rho_realtime JSON missing 'altitude_km' array")
        i400 = int(np.argmin(np.abs(alts - 400.0)))
        if abs(alts[i400] - 400.0) > 25.0:
            # 400 km isn't in the altitude grid — bail loudly.
            raise ValueError(
                f'altitude_km grid does not include 400 km '
                f'(closest is {alts[i400]} km)'
            )
        self._alt_grid = alts
        self._i400 = i400

        self._t = np.array([_to_days(t) for t in blk['time_utc']])
        rho_arr = np.asarray(blk['rho'], float)  # (n_t, n_alt)
        self._rho400 = rho_arr[:, i400]
        self._Tinf_bg = np.asarray(blk['J70_Tinf_K'], float)

        # mt, bt per bin (linear T_inf = mt * rho_400 + bt).
        for req in ('T_J70_m', 'T_J70_b'):
            if req not in blk:
                raise KeyError(
                    f"rho_realtime.solver missing {req!r}. "
                    "Regenerate with dragtrack_ops >= v0.8 (schema requires "
                    "T_J70_m and T_J70_b per bin)."
                )
        self._mt = np.asarray(blk['T_J70_m'], float)
        self._bt = np.asarray(blk['T_J70_b'], float)
        self._bin_days = np.asarray(blk.get('bin_days', np.full_like(self._t, np.nan)), float)

        # Sanity: NaNs in the background series → clean interp with edge fills.
        for name, arr in (
            ('rho_bg@400', self._rho400),
            ('T_J70_m', self._mt),
            ('T_J70_b', self._bt),
        ):
            if not np.all(np.isfinite(arr)):
                n_bad = int(np.sum(~np.isfinite(arr)))
                arr = self._forward_fill(arr)
                # Store back on self.
                if name == 'rho_bg@400':
                    self._rho400 = arr
                elif name == 'T_J70_m':
                    self._mt = arr
                else:
                    self._bt = arr
                # Not fatal — warn via attribute for programmatic access.
                self._nan_fills = getattr(self, '_nan_fills', {})
                self._nan_fills[name] = n_bad

    # ── Constructors ───────────────────────────────────────────────────

    @classmethod
    def from_files(
        cls,
        rho_realtime: PathLike | Mapping[str, Any],
        clim: PathLike | None = None,
        j70_lut: PathLike | None = None,
    ) -> "DragTrackDensity":
        """Build from a filesystem path (or already-parsed dict) plus optional overrides."""
        if isinstance(rho_realtime, Mapping):
            rt = dict(rho_realtime)
        else:
            with open(rho_realtime, 'r') as f:
                rt = json.load(f)
        c = _Clim(clim) if clim is not None else None
        j = J70(j70_lut) if j70_lut is not None else None
        return cls(rho_realtime=rt, clim=c, j70=j)

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "DragTrackDensity":
        """Fetch a ``dragtrack_rho_realtime.json`` over HTTP(S).

        Parameters
        ----------
        url : str
            URL of the JSON product.
        **kwargs
            Forwarded to :meth:`from_files` (``clim``, ``j70_lut``).
        """
        with urlopen(url) as r:
            rt = json.load(r)
        return cls.from_files(rt, **kwargs)

    # ── Provenance ─────────────────────────────────────────────────────

    @property
    def meta(self) -> dict:
        """Return the ``meta`` block from the rho_realtime JSON (or ``{}``)."""
        return dict(self._raw.get('meta', {}))

    @property
    def tle_edge_utc(self) -> str | None:
        return self.meta.get('tle_edge_utc')

    @property
    def schema_version(self) -> str | None:
        return self.meta.get('schema_version')

    @property
    def generated_utc(self) -> str | None:
        return self.meta.get('generated_utc')

    @property
    def time_utc(self) -> list[str]:
        """The ``solver.time_utc`` grid (ISO strings)."""
        return list(self._raw['solver']['time_utc'])

    @property
    def altitude_km(self) -> NDArray[np.float64]:
        return self._alt_grid.copy()

    @property
    def recipe(self) -> str:
        """Human-readable reconstruction recipe string."""
        return (
            "rho_400(t, lat, lst) = rho_bg(t) * (1 + sum_l beta_l(doy) * Y_l(lat, lst))\n"
            "Tinf(t, lat, lst)   = mt(t) * rho_400 + bt(t)\n"
            "rho(alt)            = rho_400 * J70(alt, Tinf) / J70(400, Tinf)"
        )

    def provenance(self) -> dict:
        """One-shot dict describing what/where/when the density was built from."""
        return {
            'schema_version': self.schema_version,
            'generated_utc': self.generated_utc,
            'tle_edge_utc': self.tle_edge_utc,
            'n_solver_bins': int(len(self._t)),
            'time_range_utc': (
                self.time_utc[0] if self.time_utc else None,
                self.time_utc[-1] if self.time_utc else None,
            ),
            'altitude_km': self._alt_grid.tolist(),
            'clim_source': str(self.clim.path),
            'j70_source': str(self.j70.path),
            'recipe': self.recipe,
        }

    # ── Internals ──────────────────────────────────────────────────────

    @staticmethod
    def _forward_fill(a: NDArray[np.float64]) -> NDArray[np.float64]:
        """Replace NaNs by forward-fill then backward-fill (edge safe)."""
        a = np.asarray(a, float).copy()
        if not np.any(~np.isfinite(a)):
            return a
        # Forward
        last = np.nan
        for i in range(a.size):
            if np.isfinite(a[i]):
                last = a[i]
            else:
                a[i] = last
        # Backward
        last = np.nan
        for i in range(a.size - 1, -1, -1):
            if np.isfinite(a[i]):
                last = a[i]
            else:
                a[i] = last
        return a

    def _interp(self, t_days: NDArray[np.float64], arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.interp(t_days, self._t, arr, left=arr[0], right=arr[-1])

    # ── Public API ─────────────────────────────────────────────────────

    def rho(
        self,
        t: TimeLike | Sequence[TimeLike] | NDArray,
        lat_deg: ArrayLike,
        lon_deg: ArrayLike,
        alt_km: ArrayLike,
    ) -> Union[float, NDArray[np.float64]]:
        """Evaluate atmospheric density at one or many ``(t, lat, lon, alt)`` points.

        Parameters
        ----------
        t : ISO-string, datetime, POSIX-days float, or 1D sequence thereof
            Sample time(s), UTC.
        lat_deg, lon_deg : array_like
            Geodetic latitude / longitude, degrees.  Broadcast with ``t``.
        alt_km : array_like
            Geodetic altitude, km.  Broadcast with ``t``.

        Returns
        -------
        float or ndarray
            Density in kg / m^3.  Scalar if every input is scalar, otherwise
            an ndarray broadcast to the common shape.
        """
        scalar_in = (
            _is_scalar_time(t)
            and np.ndim(lat_deg) == 0
            and np.ndim(lon_deg) == 0
            and np.ndim(alt_km) == 0
        )
        t_arr = _as_time_seq(t)
        lat = np.atleast_1d(np.asarray(lat_deg, float))
        lon = np.atleast_1d(np.asarray(lon_deg, float))
        alt = np.atleast_1d(np.asarray(alt_km, float))
        # Broadcast all to common length.
        try:
            t_arr, lat, lon, alt = np.broadcast_arrays(
                np.asarray(t_arr, object), lat, lon, alt
            )
        except ValueError as e:
            raise ValueError(
                f'Could not broadcast inputs: t={len(t_arr)} lat={lat.shape} '
                f'lon={lon.shape} alt={alt.shape}'
            ) from e

        t_days, ut_h, doy = _vec_time_features(np.ravel(t_arr).tolist())
        lat_f = np.ravel(lat).astype(float)
        lon_f = np.ravel(lon).astype(float)
        alt_f = np.ravel(alt).astype(float)
        lst_h = (ut_h + lon_f / 15.0) % 24.0

        # Time-interp solver arrays.
        rho_bg = self._interp(t_days, self._rho400)
        mt = self._interp(t_days, self._mt)
        bt = self._interp(t_days, self._bt)

        # Spatial fractional perturbation.
        beta_frac = self.clim.beta_at_doy(doy)  # (N, 15)
        Y = Y_modes(lat_f, lst_h)               # (N, 15)
        frac = np.einsum('ij,ij->i', beta_frac, Y)
        rho_400_pt = np.maximum(rho_bg * (1.0 + frac), 1e-30)

        # Point Tinf then vertical scaling via J70.
        Tinf_pt = mt * rho_400_pt + bt
        r_alt = self.j70(alt_f, Tinf_pt)
        r_400 = self.j70(np.full_like(alt_f, 400.0), Tinf_pt)
        out = rho_400_pt * r_alt / r_400

        if scalar_in:
            return float(out[0])
        return out.reshape(np.shape(np.asarray(t_arr)))

    # ── Convenience: orbit propagation → density ──────────────────────

    def rho_along_tle(self, l1: str, l2: str, times: Iterable[TimeLike]):
        """Propagate a TLE to each time; return ``(times, lat, lon, alt, rho)``."""
        from .ephem import propagate_tle
        t_out, lat, lon, alt = propagate_tle(l1, l2, times)
        rho = self.rho(t_out, lat, lon, alt)
        return t_out, lat, lon, alt, rho

    def rho_along_kepler(
        self,
        *,
        a_km: float,
        e: float,
        inc_deg: float,
        raan_deg: float,
        argp_deg: float,
        m0_deg: float,
        bstar: float,
        epoch_utc: TimeLike,
        times: Iterable[TimeLike],
    ):
        """Propagate Kepler elements at ``epoch_utc`` to ``times``; return orbit + rho."""
        from .ephem import propagate_kepler
        t_out, lat, lon, alt = propagate_kepler(
            a_km=a_km, e=e, inc_deg=inc_deg, raan_deg=raan_deg,
            argp_deg=argp_deg, m0_deg=m0_deg, bstar=bstar,
            epoch_utc=epoch_utc, times=times,
        )
        rho = self.rho(t_out, lat, lon, alt)
        return t_out, lat, lon, alt, rho


# ─── Small helpers exposed for scalar/vector routing ───────────────────

def _is_scalar_time(t: Any) -> bool:
    if isinstance(t, (str, datetime, np.datetime64)):
        return True
    if isinstance(t, (int, float, np.floating, np.integer)):
        return True
    return False


def _as_time_seq(t):
    """Coerce ``t`` into a 1-D sequence of time-like elements."""
    if _is_scalar_time(t):
        return [t]
    # np.array with dtype=object handles heterogeneous lists.
    return list(t)
