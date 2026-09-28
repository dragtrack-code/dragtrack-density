"""Point atmospheric density from DragTrack JSON products.

Public entry point: :class:`DragTrackDensity` and the constant
:data:`SCENARIOS`.  See the top-level README for usage and coordinate
conventions.
"""
from __future__ import annotations

import gzip
import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple, Union
from urllib.request import urlopen

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import Y_modes
from .j70 import J70

PathLike = Union[str, Path]
TimeLike = Union[str, datetime, float, int, np.datetime64]

SCENARIOS: Tuple[str, ...] = (
    'solar_only', 'quiet', 'moderate', 'intense', 'extreme', 'median',
)

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


# ─── Time helpers ─────────────────────────────────────────────────────────

def _parse_time(t: TimeLike) -> datetime:
    """Parse ``t`` into a timezone-aware UTC ``datetime``.  Naive datetimes
    and ISO strings without an offset are taken as UTC."""
    if isinstance(t, str):
        t = datetime.fromisoformat(t.replace('Z', '+00:00'))
    if isinstance(t, datetime):
        return (t.replace(tzinfo=timezone.utc) if t.tzinfo is None
                else t.astimezone(timezone.utc))
    if isinstance(t, np.datetime64):
        secs = (t - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        return datetime.fromtimestamp(float(secs), tz=timezone.utc)
    if isinstance(t, (float, int, np.floating, np.integer)):
        return _EPOCH_UTC + timedelta(days=float(t))
    raise TypeError(f'Cannot parse time {t!r}')


def _to_days(t: TimeLike) -> float:
    """Days since 1970-01-01 UTC (fractional)."""
    return (_parse_time(t) - _EPOCH_UTC).total_seconds() / 86400.0


def _vec_to_days(times: Iterable[TimeLike]) -> NDArray[np.float64]:
    """Days since 1970-01-01 UTC for a sequence of times."""
    return np.array([_to_days(t) for t in times], float)


# ─── JSON parsing helpers ─────────────────────────────────────────────────

def _as_2d(v) -> NDArray[np.float64]:
    """Convert list-of-lists (with ``None`` for NaN) → ``(N, K)`` float array."""
    if v is None:
        return np.zeros((0, 0), float)
    return np.array([[np.nan if x is None else float(x) for x in row]
                     for row in v], float)


def _as_1d(v) -> NDArray[np.float64]:
    if v is None:
        return np.zeros(0, float)
    return np.array([np.nan if x is None else float(x) for x in v], float)


def _spatial_lm(d: Mapping[str, Any], block: str, times, key: str = 'TJ70_lm') -> NDArray:
    """spatial.<block>.<key> as (n, 16), checked against the block's times."""
    sp = ((d.get('spatial') or {}).get(block)) or {}
    if list(sp.get('time_utc', [])) != list(times):
        raise ValueError(f"spatial.{block}.time_utc does not match {block}.time_utc")
    lm = _as_2d(sp.get(key)).reshape(-1, 16) if len(times) else np.zeros((0, 16))
    if lm.shape != (len(times), 16):
        raise ValueError(f"spatial.{block}.{key} missing or not (n_time, 16)")
    return lm


# ─── Internal helpers ────────────────────────────────────────────────────

def _sphere_grid():
    """(lat, lst) grid + cos-lat weights for globe means: 5 deg latitude
    centres x 0.5 h local time, the producer's grid."""
    lat = np.arange(-87.5, 88.0, 5.0)
    lst = np.arange(0.0, 24.0, 0.5)
    LAT_g, LST_g = np.meshgrid(lat, lst, indexing='ij')
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], LAT_g.shape).astype(float)
    return LAT_g, LST_g, w


# ─── Main class ───────────────────────────────────────────────────────────

class DragTrackDensity:
    """Point atmospheric density ``rho(t, lat, lon, alt)``.

    Construct one of these from a DragTrack realtime JSON (and optionally
    the historical-archive JSON), then call :meth:`rho` with any mix of scalar and
    array-like coordinates.  See the README for coordinate conventions
    and the list of scenarios.

    Parameters
    ----------
    rho_realtime : dict
        Parsed realtime JSON (see :meth:`from_files` / :meth:`from_url`
        for the convenience loaders that handle file paths and URLs).
    rho_full : dict, optional
        Parsed historical-archive JSON (``dragtrack_rho_full_*``).  Adds
        solver history from 2014 onward for times before the realtime
        file's window.
    j70 : J70, optional
        Preloaded Jacchia-70 lookup table.  Loaded from the packaged
        asset if omitted.
    rho_forecast : dict, optional
        Parsed SWPC-driven forecast JSON (``dragtrack_rho_forecast_*``),
        ideally from the same run (``meta.run_id``) as ``rho_realtime``.
        Its rows replace the realtime fit rows from its first time on,
        extending ``'baseline'`` and every scenario to the SWPC horizon
        (about 45 days).
    """

    def __init__(
        self,
        rho_realtime: Mapping[str, Any],
        rho_full: Mapping[str, Any] | None = None,
        j70: J70 | None = None,
        rho_forecast: Mapping[str, Any] | None = None,
    ) -> None:
        if 'solver' not in rho_realtime:
            raise KeyError(
                "This file doesn't look like a DragTrack realtime JSON "
                "(no 'solver' block).  Get a current file from the "
                "DragTrack service.")
        self._meta = dict(rho_realtime.get('meta', {}))

        alts = np.asarray(rho_realtime.get('altitude_km', []), float)
        if alts.size == 0:
            raise KeyError("rho_realtime JSON missing 'altitude_km' array")
        i400 = int(np.argmin(np.abs(alts - 400.0)))
        if abs(alts[i400] - 400.0) > 25.0:
            raise ValueError(
                f'altitude_km grid does not include 400 km '
                f'(closest is {alts[i400]} km)')
        self._alt_grid = alts
        self._i400 = i400
        self.j70 = j70 or J70()

        # ── Extract solver block (always the reanalysis) ────────────────
        sv = rho_realtime['solver']
        sv_t = np.array([_to_days(t) for t in sv['time_utc']], float)
        sv_rho400 = _as_2d(sv['rho'])[:, i400]
        sv_TJ70 = _as_1d(sv['TJ70_K'])
        sv_lm = _spatial_lm(rho_realtime, 'solver', sv['time_utc'])

        # ── Optional historical archive (rho_full): prepend solver rows
        # strictly before the realtime solver window.
        if rho_full is not None:
            hs = rho_full['solver']
            h_t = np.array([_to_days(t) for t in hs['time_utc']], float)
            h_rho400 = _as_2d(hs['rho'])[:, i400]
            h_TJ70 = _as_1d(hs['TJ70_K'])
            h_lm = _spatial_lm(rho_full, 'solver', hs['time_utc'])
            if not np.array_equal(np.asarray(rho_full.get('altitude_km', []), float), alts):
                raise ValueError("rho_full and rho_realtime altitude_km grids differ")
            keep = h_t < (sv_t.min() if sv_t.size else np.inf)
            sv_t = np.concatenate([h_t[keep], sv_t])
            sv_rho400 = np.concatenate([h_rho400[keep], sv_rho400])
            sv_TJ70 = np.concatenate([h_TJ70[keep], sv_TJ70])
            sv_lm = np.vstack([h_lm[keep], sv_lm])

        # Solver edge: last solver (reanalysis) time.  Fit rows
        # are only used AFTER it -- before the edge every source is the
        # same reanalysis, so scenarios cannot change the past.
        self._solver_edge = float(np.nanmax(sv_t)) if sv_t.size else -np.inf

        # ── Extract fit block (rho_realtime.fit) ────────────────────────
        ft = rho_realtime.get('fit') or {}
        ft_t = np.array([_to_days(t) for t in ft.get('time_utc', [])], float)
        ft_rho_main = _as_2d(ft.get('rho_fit'))
        ft_TJ70_main = _as_1d(ft.get('TJ70_K'))
        ft_lm_main = _spatial_lm(rho_realtime, 'fit', ft.get('time_utc', []))
        # (rho400, TJ70, TJ70_lm) of the fit segment per source; a scenario
        # is available only when the file carries all three of its arrays.
        fits = {'baseline': (ft_rho_main[:, i400] if ft_rho_main.size else np.zeros(0),
                             ft_TJ70_main, ft_lm_main)}
        for k in SCENARIOS:
            if f'rho_{k}' not in ft:
                continue
            if f'TJ70_K_{k}' not in ft:
                raise KeyError(f"fit.TJ70_K_{k} missing: the file predates "
                               "schema 0.10.0; get a current file")
            fits[k] = (_as_2d(ft[f'rho_{k}'])[:, i400], _as_1d(ft[f'TJ70_K_{k}']),
                       _spatial_lm(rho_realtime, 'fit', ft['time_utc'], f'TJ70_lm_{k}'))

        # ── Optional SWPC-driven forecast (rho_forecast): its rows replace
        # the realtime fit rows from its first time on.
        if rho_forecast is not None:
            ft_t, fits = _splice_forecast(rho_realtime, rho_forecast, alts, i400,
                                          ft_t, fits)

        # ── Merge per source: solver → fit ──────────────────────────────
        LAT_g, LST_g, self._w = _sphere_grid()
        self._Y_grid_flat = Y_modes(LAT_g, LST_g).reshape(-1, 16)   # (n_grid, 16)

        # Solver segment is identical for every source: normalize once.
        sv_K = self._norm_K(sv_rho400, sv_TJ70, sv_lm)
        self._sources: dict[str, dict] = {
            src: self._pack_source(sv_t, sv_TJ70, sv_lm, sv_K, ft_t, *f)
            for src, f in fits.items()}

    # ── Source packer ─────────────────────────────────────────────────────

    def _norm_K(self, rho400, TJ70, lm, chunk: int = 2048) -> NDArray:
        """Per-timestep amplitude K so the cos-lat-weighted (lat, LST) mean of
        K*J70(400, TJ70*(1+sum lm*Y)) equals rho400, both at constant
        geodetic height 400 km.  Vectorized in chunks."""
        n_t = len(rho400)
        K = np.full(n_t, np.nan)
        if n_t == 0:
            return K
        lm = np.asarray(lm, float).reshape(-1, 16)
        w = self._w.ravel()
        wsum = float(w.sum())
        Yg = self._Y_grid_flat                                  # (n_grid, 16)
        ok = (np.isfinite(TJ70) & np.isfinite(rho400) & (rho400 > 0)
              & np.all(np.isfinite(lm), axis=1))
        idx = np.where(ok)[0]
        for i0 in range(0, idx.size, chunk):
            ii = idx[i0:i0 + chunk]
            T_g = TJ70[ii, None] * (1.0 + lm[ii] @ Yg.T)         # (n, n_grid)
            r_g = self.j70(np.full(T_g.shape, 400.0), T_g)
            m = (r_g * w).sum(axis=1) / wsum
            good = m > 0
            K[ii[good]] = rho400[ii[good]] / m[good]
        return K

    def _pack_source(self, sv_t, sv_TJ70, sv_lm, sv_K,
                     ft_t, ft_rho, ft_TJ70, ft_lm) -> dict:
        """Solver reanalysis up to its edge, then fit.

        Fit rows at or before the solver edge are dropped (the fit block
        also carries hindcast rows over the solver window; using them
        would let scenarios alter the past).
        """
        fk = ft_t > self._solver_edge
        ft_t, ft_rho, ft_TJ70, ft_lm = ft_t[fk], ft_rho[fk], ft_TJ70[fk], ft_lm[fk]
        t = np.concatenate([sv_t, ft_t])
        TJ70 = np.concatenate([sv_TJ70, ft_TJ70])
        lm = np.vstack([sv_lm, ft_lm])
        K = np.concatenate([sv_K, self._norm_K(ft_rho, ft_TJ70, ft_lm)])
        u_t, u_idx = np.unique(t, return_index=True)      # sorted, first of duplicates
        return dict(t=u_t, TJ70_bg=TJ70[u_idx], TJ70_lm=lm[u_idx], K=K[u_idx])

    # ── Public API ────────────────────────────────────────────────────────

    def rho(
        self,
        t: TimeLike | Sequence[TimeLike] | NDArray,
        lat_deg: ArrayLike,
        lon_deg: ArrayLike,
        alt_km: ArrayLike,
        source: str = 'baseline',
    ) -> Union[float, NDArray[np.float64]]:
        """Evaluate density at ``(t, lat_deg, lon_deg, alt_km)``.

        Parameters
        ----------
        t : ISO-8601 str, ``datetime``, ``np.datetime64``, POSIX-days
            float, or 1-D sequence thereof.  UTC.
        lat_deg : array_like, WGS-84 geodetic latitude, degrees.
        lon_deg : array_like, degrees east.
        alt_km : array_like, geodetic height above the WGS-84 ellipsoid,
            km.  Density is evaluated at exactly this height and geodetic
            latitude (J70's native coordinates; the products'
            ``altitude_km`` grid uses the same reference).  NaN outside the
            J70 table (100-1000 km).
        source : str, one of ``'baseline'`` (default) or a name in
            :data:`SCENARIOS`.  ``'baseline'`` uses the main fit
            rho / TJ70_lm; scenario names swap the fit segment
            (solver segment remains reanalysis in every case).  Only
            scenarios present in the file are available.

        Returns
        -------
        float or ndarray
            Density in kg / m^3.  Scalar if every input is scalar.  NaN
            for times outside the loaded coverage (see :attr:`coverage`).
        """
        if source not in self._sources:
            raise ValueError(f'source={source!r} not available; this file has '
                             f'{sorted(self._sources)}')

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
        try:
            t_arr, lat, lon, alt = np.broadcast_arrays(
                np.asarray(t_arr, object), lat, lon, alt)
        except ValueError as e:
            raise ValueError(
                f'Could not broadcast inputs: t={len(t_arr)} '
                f'lat={lat.shape} lon={lon.shape} alt={alt.shape}') from e
        t_days = _vec_to_days(np.ravel(t_arr).tolist())
        lat_f = np.ravel(lat).astype(float)
        lon_f = np.ravel(lon).astype(float)
        alt_f = np.ravel(alt).astype(float)       # geodetic height, used as-is
        lst_h = _apparent_lst_h(t_days, lon_f)

        s = self._sources[source]
        TJ70_bg = _interp1d(t_days, s['t'], s['TJ70_bg'])
        K = _interp1d(t_days, s['t'], s['K'])
        # Interp the 16 spatial coefficients per-mode.
        TJ70_lm = np.column_stack([_interp1d(t_days, s['t'], s['TJ70_lm'][:, k])
                                    for k in range(16)])
        # Per-point synthesis: TJ70_pt = TJ70_bg * (1 + sum_k lm_k * Y_k).
        Y = Y_modes(lat_f, lst_h)                # (N, 16)
        frac = np.einsum('nk,nk->n', TJ70_lm, Y)
        TJ70_pt = TJ70_bg * (1.0 + frac)
        rho_out = K * self.j70(alt_f, TJ70_pt)

        if scalar_in:
            return float(rho_out[0])
        return rho_out.reshape(np.shape(np.asarray(t_arr)))

    def rho_from_state(
        self,
        t: TimeLike | Sequence[TimeLike] | NDArray,
        r_km: ArrayLike,
        frame: str = 'EarthMJ2000Eq',
        source: str = 'baseline',
    ) -> Union[float, NDArray[np.float64]]:
        """Same as :meth:`rho`, but takes a position vector in km.

        Convenience for propagators like GMAT that carry the spacecraft
        state as an ``(x, y, z)`` vector rather than as geodetic
        ``(lat, lon, alt)``.  Handles the frame conversion internally.

        Parameters
        ----------
        t : time-like or 1-D sequence
            UTC time(s), one per state.
        r_km : array_like, shape ``(..., 3)``
            Position vector(s) in km, one per time.  ``r_km[..., 0]``,
            ``r_km[..., 1]``, ``r_km[..., 2]`` are the x, y, z components.
        frame : {'EarthMJ2000Eq', 'EarthFixed'}
            Coordinate system of ``r_km``.  ``'EarthMJ2000Eq'`` is
            GMAT's default inertial frame (Earth-centered J2000 mean
            equator); the package rotates it to Earth-fixed via GMST
            before computing WGS-84 geodetic latitude / longitude /
            height, which are passed unchanged to :meth:`rho`.
            ``'EarthFixed'`` is already Earth-body-fixed (ECEF), no
            rotation applied.
        source : str
            Same as :meth:`rho`.

        Returns
        -------
        float or ndarray
            Density in kg / m^3, one value per input state.
        """
        r = np.asarray(r_km, float)
        if r.shape[-1] != 3:
            raise ValueError(f'r_km last axis must be size 3 (got {r.shape})')
        if frame not in ('EarthMJ2000Eq', 'EarthFixed'):
            raise ValueError(
                f"frame must be 'EarthMJ2000Eq' or 'EarthFixed' (got {frame!r})")
        flat_r = r.reshape(-1, 3)
        n = flat_r.shape[0]
        t_arr = _as_time_seq(t)
        if len(t_arr) == 1:
            t_arr = t_arr * n
        elif len(t_arr) != n:
            raise ValueError(f't and r_km lengths mismatch '
                             f'(t={len(t_arr)} vs r_km={n})')
        r_ecef = (_eci_to_ecef(flat_r, _vec_to_days(t_arr))
                  if frame == 'EarthMJ2000Eq' else flat_r)
        lat_deg, lon_deg, alt_km = _ecef_to_geodetic(r_ecef)
        out = self.rho(t_arr, lat_deg, lon_deg, alt_km, source=source)
        if r.ndim == 1:
            return float(out[0])
        return np.asarray(out).reshape(r.shape[:-1])

    # ── Loaders ───────────────────────────────────────────────────────────

    @classmethod
    def from_files(
        cls,
        rho_realtime: PathLike | Mapping[str, Any],
        rho_full: PathLike | Mapping[str, Any] | None = None,
        j70_lut: PathLike | None = None,
        rho_forecast: PathLike | Mapping[str, Any] | None = None,
    ) -> "DragTrackDensity":
        """Build from local path(s), HTTP(S) URL(s) or already-parsed
        dict(s); paths and URLs may be mixed."""
        def load(src):
            if src is None:
                return None
            return dict(src) if isinstance(src, Mapping) else _load_json(src)
        j = J70(j70_lut) if j70_lut is not None else None
        return cls(rho_realtime=load(rho_realtime), rho_full=load(rho_full), j70=j,
                   rho_forecast=load(rho_forecast))

    @classmethod
    def from_url(
        cls,
        rho_realtime_url: str,
        rho_full_url: str | None = None,
        j70_lut: PathLike | None = None,
        rho_forecast_url: str | None = None,
    ) -> "DragTrackDensity":
        """Fetch JSON products over HTTP(S) (same as :meth:`from_files`)."""
        return cls.from_files(rho_realtime_url, rho_full_url or None, j70_lut,
                              rho_forecast_url or None)

    # ── Metadata ──────────────────────────────────────────────────────────

    @property
    def meta(self) -> dict:
        """The ``meta`` block from the ``rho_realtime`` JSON."""
        return dict(self._meta)

    @property
    def coverage(self) -> dict:
        """UTC span covered by the loaded products, as ISO strings.

        ``{'solver_start', 'solver_edge', 'end', 'end_by_source'}``.
        ``end`` is the last time the default ``source='baseline'`` returns
        a finite density; ``end_by_source`` maps every source name
        (``'baseline'`` and each scenario) to its own last finite time --
        scenarios typically run several days past the baseline end.
        Queries outside ``[solver_start, end_by_source[source]]`` return
        NaN."""
        def _iso(d):
            if not np.isfinite(d):
                return None
            return (_EPOCH_UTC + timedelta(days=float(d))).strftime('%Y-%m-%dT%H:%M:%SZ')
        s = self._sources['baseline']
        ends = {name: _iso(_source_end(src))
                for name, src in self._sources.items()}
        return {'solver_start': _iso(s['t'][0]),
                'solver_edge': _iso(self._solver_edge),
                'end': ends['baseline'],
                'end_by_source': ends}

    @property
    def altitude_km(self) -> NDArray[np.float64]:
        """The products' altitude grid, km (geodetic).  ``rho()`` accepts any
        height within the J70 table (100-1000 km)."""
        return self._alt_grid.copy()


# ─── Helpers ──────────────────────────────────────────────────────────────

def _splice_forecast(rho_realtime: Mapping[str, Any], rho_forecast: Mapping[str, Any],
                     alts: NDArray, i400: int, ft_t: NDArray, fits: dict):
    """Realtime fit rows before the forecast's first time, then the forecast
    rows, per source.  Every source in ``fits`` must be in the forecast."""
    fc = rho_forecast.get('fit') or {}
    if not fc.get('time_utc'):
        raise KeyError("not a DragTrack rho_forecast JSON (no fit.time_utc)")
    if not np.array_equal(np.asarray(rho_forecast.get('altitude_km', []), float), alts):
        raise ValueError("rho_forecast and rho_realtime altitude_km grids differ")
    rid_rt = (rho_realtime.get('meta') or {}).get('run_id')
    rid_fc = (rho_forecast.get('meta') or {}).get('run_id')
    if rid_rt and rid_fc and rid_rt != rid_fc:
        warnings.warn(f"rho_forecast run_id {rid_fc} differs from rho_realtime "
                      f"run_id {rid_rt}; use files from the same run", stacklevel=3)
    fc_t = _vec_to_days(fc['time_utc'])
    keep = ft_t < fc_t[0]
    out = {}
    for src, (rho400, tj70, lm) in fits.items():
        k = '' if src == 'baseline' else f'_{src}'
        rkey = 'rho_fit' if src == 'baseline' else f'rho_{src}'
        if rkey not in fc or f'TJ70_K{k}' not in fc:
            raise KeyError(f"rho_forecast has no fit.{rkey} / fit.TJ70_K{k}")
        out[src] = (np.concatenate([rho400[keep], _as_2d(fc[rkey])[:, i400]]),
                    np.concatenate([tj70[keep], _as_1d(fc[f'TJ70_K{k}'])]),
                    np.vstack([lm[keep], _spatial_lm(rho_forecast, 'fit', fc['time_utc'],
                                                     f'TJ70_lm{k}')]))
    return np.concatenate([ft_t[keep], fc_t]), out


def _source_end(s: dict) -> float:
    """Last time (POSIX days) at which :meth:`DragTrackDensity.rho` is
    finite for packed source ``s``: every interpolated quantity (K,
    TJ70_bg, each TJ70_lm mode) goes NaN past its own last finite sample
    (see :func:`_interp1d`), so the end is the earliest of those."""
    t = s['t']
    cols = [s['K'], s['TJ70_bg']] + [s['TJ70_lm'][:, k] for k in range(16)]
    last = []
    for y in cols:
        ok = np.isfinite(y)
        if not np.any(ok):
            return np.nan
        last.append(t[ok][-1])
    return float(min(last))


def _interp1d(x_new: NDArray, x_ref: NDArray, y_ref: NDArray) -> NDArray:
    """Linear interpolation over the finite reference samples.  NaN
    outside the reference span -- no edge-holding / extrapolation."""
    ok = np.isfinite(y_ref)
    if not np.any(ok):
        return np.full_like(x_new, np.nan, dtype=float)
    return np.interp(x_new, x_ref[ok], y_ref[ok], left=np.nan, right=np.nan)


# Height convention: density is evaluated at geodetic height above the
# WGS-84 ellipsoid with geodetic latitude (J70's native coordinates).  The
# products' altitude_km grid / 400 km reference are geodetic heights too.
_WGS84_A_KM = 6378.137
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)   # eccentricity squared


def _gmst_rad(t_days: NDArray[np.float64]) -> NDArray[np.float64]:
    """Greenwich Mean Sidereal Time in radians (IAU-1982 series).

    ``t_days`` is UT1 (approximated by UTC — drag lookup doesn't need
    the sub-second UT1-UTC correction) in days since 1970-01-01 UTC.
    """
    # Days since J2000.0 (2000-01-01 12:00 UTC = 10957.5 POSIX days).  T
    # includes the time of day, so the 876600 h·T term carries the daily
    # rotation.
    d = np.asarray(t_days, float) - 10957.5
    T = d / 36525.0
    gmst_s = (67310.54841
              + (876600.0 * 3600.0 + 8640184.812866) * T
              + 0.093104 * T ** 2
              - 6.2e-6 * T ** 3)
    return np.deg2rad((gmst_s / 240.0) % 360.0)


def _sun_ra_rad(t_days: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apparent solar right ascension (rad), low-precision almanac series
    (~0.01 deg); the same formula the DragTrack pipeline uses."""
    n = np.asarray(t_days, float) - 10957.5              # days since J2000.0
    L = np.deg2rad(280.460 + 0.9856474 * n)              # mean longitude
    g = np.deg2rad(357.528 + 0.9856003 * n)              # mean anomaly
    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)
    eps = np.deg2rad(23.439 - 0.0000004 * n)             # obliquity
    return np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))


def _apparent_lst_h(t_days: NDArray[np.float64],
                    lon_deg: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apparent local solar time (h) from the point's right ascension
    (GMST + longitude) and the Sun's: 12 h where the Sun is overhead.
    Differs from UT + lon/15 by the equation of time (up to ~16 min)."""
    ra = _gmst_rad(t_days) + np.deg2rad(np.asarray(lon_deg, float))
    return (12.0 + np.degrees(ra - _sun_ra_rad(t_days)) / 15.0) % 24.0


def _eci_to_ecef(r_eci_km: NDArray[np.float64],
                  t_days: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate an inertial (EarthMJ2000Eq) position vector into
    Earth-fixed (ECEF) using GMST.  Precession/nutation not applied:
    in 2026 precession since J2000 is ~0.37 deg (~45 km of along-track
    position at LEO, mostly a ~1.5 min local-solar-time shift).  Pass
    ``frame='EarthFixed'`` if you need better geodetic accuracy."""
    theta = _gmst_rad(t_days)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = r_eci_km[..., 0], r_eci_km[..., 1], r_eci_km[..., 2]
    x_e = c * x + s * y
    y_e = -s * x + c * y
    z_e = z
    return np.stack([x_e, y_e, z_e], axis=-1)


def _ecef_to_geodetic(r_ecef_km: NDArray[np.float64]):
    """WGS84 ECEF (km) → geodetic (lat_deg, lon_deg, alt_km), Bowring's
    iteration (5 steps)."""
    x = np.asarray(r_ecef_km[..., 0], float)
    y = np.asarray(r_ecef_km[..., 1], float)
    z = np.asarray(r_ecef_km[..., 2], float)
    a = _WGS84_A_KM
    e2 = _WGS84_E2
    p = np.sqrt(x ** 2 + y ** 2)
    # Iterative Bowring for latitude.
    lat = np.arctan2(z, p * (1.0 - e2))
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1.0 - e2 * sin_lat ** 2)
        lat = np.arctan2(z + e2 * N * sin_lat, p)
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        alt = np.where(np.abs(sin_lat) < 0.5,
                        p / np.cos(lat) - N,
                        z / np.where(sin_lat == 0, np.nan, sin_lat)
                        - N * (1.0 - e2))
    lon = np.arctan2(y, x)
    return np.rad2deg(lat), np.rad2deg(lon), alt


_URL_TIMEOUT_S = 120


def _load_json(src: PathLike) -> dict:
    """Parse a product JSON from a local path or an HTTP(S) URL
    (``.gz`` compressed or plain)."""
    s = str(src)
    if s.startswith(('http://', 'https://')):
        with urlopen(s, timeout=_URL_TIMEOUT_S) as r:
            raw = r.read()
        if s.endswith('.gz'):
            raw = gzip.decompress(raw)
        return json.loads(raw)
    p = Path(src)
    if p.suffix == '.gz':
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            return json.load(f)
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def _is_scalar_time(t: Any) -> bool:
    if isinstance(t, (str, datetime, np.datetime64)):
        return True
    if isinstance(t, (int, float, np.floating, np.integer)):
        return True
    return False


def _as_time_seq(t):
    if _is_scalar_time(t):
        return [t]
    return list(t)
