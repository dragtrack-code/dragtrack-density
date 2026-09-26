"""Point atmospheric density from DragTrack JSON products.

Public entry point: :class:`DragTrackDensity` and the constant
:data:`SCENARIOS`.  See the top-level README for usage and coordinate
conventions.
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple, Union
from urllib.request import urlopen

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import REG_NAMES, Y_modes
from .j70 import J70

PathLike = Union[str, Path]
TimeLike = Union[str, datetime, float, int, np.datetime64]

SCENARIOS: Tuple[str, ...] = (
    'solar_only', 'quiet', 'moderate', 'intense', 'extreme', 'median',
)

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


# ─── Time helpers ─────────────────────────────────────────────────────────

def _parse_time(t: TimeLike) -> datetime:
    """Parse ``t`` into a timezone-aware UTC ``datetime``."""
    if isinstance(t, datetime):
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    if isinstance(t, str):
        return datetime.fromisoformat(t.replace('Z', '+00:00'))
    if isinstance(t, np.datetime64):
        secs = (t - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        return datetime.fromtimestamp(float(secs), tz=timezone.utc)
    if isinstance(t, (float, int, np.floating, np.integer)):
        return _EPOCH_UTC + (datetime.fromtimestamp(float(t) * 86400.0,
                                                    tz=timezone.utc) - _EPOCH_UTC)
    raise TypeError(f'Cannot parse time {t!r}')


def _to_days(t: TimeLike) -> float:
    """Days since 1970-01-01 UTC (fractional)."""
    return (_parse_time(t) - _EPOCH_UTC).total_seconds() / 86400.0


def _vec_time_features(times: Iterable[TimeLike]) -> Tuple[NDArray, NDArray]:
    """Return arrays ``(t_days, ut_hours)`` for a sequence of times."""
    dts = [_parse_time(t) for t in times]
    n = len(dts)
    t_days = np.empty(n, float)
    ut_h = np.empty(n, float)
    for i, d in enumerate(dts):
        t_days[i] = (d - _EPOCH_UTC).total_seconds() / 86400.0
        ut_h[i] = d.hour + d.minute / 60.0 + d.second / 3600.0
    return t_days, ut_h


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


# ─── Internal helpers ────────────────────────────────────────────────────

def _sphere_grid(n_lat: int = 36, n_lst: int = 24):
    """Canonical (lat, lst) grid + cos-lat area weights for sphere means."""
    lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lst_edges = np.linspace(0.0, 24.0, n_lst + 1)
    lst_c = 0.5 * (lst_edges[:-1] + lst_edges[1:])
    LST_g, LAT_g = np.meshgrid(lst_c, lat_c, indexing='xy')
    w_lat = np.cos(np.deg2rad(lat_c))
    w = np.broadcast_to(w_lat[:, None], LAT_g.shape).astype(float)
    return LAT_g, LST_g, w


# ─── Main class ───────────────────────────────────────────────────────────

class DragTrackDensity:
    """Point atmospheric density ``rho(t, lat, lon, alt)``.

    Construct one of these from a DragTrack realtime JSON (and optionally
    a forecast JSON), then call :meth:`rho` with any mix of scalar and
    array-like coordinates.  See the README for coordinate conventions
    and the list of scenarios.

    Parameters
    ----------
    rho_realtime : dict
        Parsed realtime JSON (see :meth:`from_files` / :meth:`from_url`
        for the convenience loaders that handle file paths and URLs).
    rho_forecast : dict, optional
        Parsed forecast JSON.  Extends the reachable time range by up
        to ~45 days.
    rho_full : dict, optional
        Parsed historical-archive JSON (``dragtrack_rho_full_*``).  Adds
        solver history from 2014 onward for times before the realtime
        file's ~27-day window.
    j70 : J70, optional
        Preloaded Jacchia-70 lookup table.  Loaded from the packaged
        asset if omitted.
    """

    SCENARIOS: Tuple[str, ...] = SCENARIOS

    def __init__(
        self,
        rho_realtime: Mapping[str, Any],
        rho_forecast: Mapping[str, Any] | None = None,
        j70: J70 | None = None,
        rho_full: Mapping[str, Any] | None = None,
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
        sv_lm = _as_2d((rho_realtime.get('spatial', {}).get('solver', {})
                        .get('TJ70_lm')))
        if sv_lm.shape != (sv_t.size, 16):
            raise ValueError(
                "This file doesn't look like a current DragTrack realtime "
                "JSON — the spatial coefficients are missing or wrong-shaped. "
                "Try downloading the latest.")

        # Scenario TJ70 regression uses the realtime (27 d) solver window,
        # matching the pipeline recipe.  Keep those before prepending history.
        rt_sv_rho400, rt_sv_TJ70 = sv_rho400, sv_TJ70

        # ── Optional historical archive (rho_full): prepend solver rows
        # strictly before the realtime solver window.
        if rho_full is not None:
            hs = rho_full['solver']
            h_t = np.array([_to_days(t) for t in hs['time_utc']], float)
            h_rho400 = _as_2d(hs['rho'])[:, i400]
            h_TJ70 = _as_1d(hs['TJ70_K'])
            h_lm = _as_2d((rho_full.get('spatial', {}).get('solver', {})
                           .get('TJ70_lm')))
            if h_lm.shape != (h_t.size, 16):
                raise ValueError("rho_full JSON: spatial.solver.TJ70_lm "
                                 "missing or wrong-shaped")
            keep = h_t < (sv_t.min() if sv_t.size else np.inf)
            sv_t = np.concatenate([h_t[keep], sv_t])
            sv_rho400 = np.concatenate([h_rho400[keep], sv_rho400])
            sv_TJ70 = np.concatenate([h_TJ70[keep], sv_TJ70])
            sv_lm = np.vstack([h_lm[keep], sv_lm])

        # Solver edge: last solver (reanalysis) time.  Fit / forecast rows
        # are only used AFTER it -- before the edge every source is the
        # same reanalysis, so scenarios cannot change the past.
        self._solver_edge = float(np.nanmax(sv_t)) if sv_t.size else -np.inf

        # ── Extract fit block (rho_realtime.fit) ────────────────────────
        ft = rho_realtime.get('fit') or {}
        ft_t = np.array([_to_days(t) for t in ft.get('time_utc', [])], float)
        ft_rho_main = _as_2d(ft.get('rho_fit'))
        ft_TJ70_main = _as_1d(ft.get('TJ70_K'))
        sp_ft = (rho_realtime.get('spatial') or {}).get('fit') or {}
        ft_lm_main = _as_2d(sp_ft.get('TJ70_lm'))
        # Per-scenario overrides in the fit block.
        ft_scen: dict[str, dict] = {}
        for k in SCENARIOS:
            r_sc = ft.get(f'rho_{k}')
            lm_sc = sp_ft.get(f'TJ70_lm_{k}')
            if r_sc is None:
                continue
            r_sc_arr = _as_2d(r_sc)
            lm_sc_arr = _as_2d(lm_sc) if lm_sc is not None else ft_lm_main
            # Reconstruct scenario TJ70_bg using log10-linear regression
            # against solver rho400 (same recipe as rho_snapshot):
            #   TJ70_sc = interp_from_solver(rho400_sc)
            # Solver truth: (sv_rho400, sv_TJ70).  Log-linear fit.
            _TJ70_sc = _regress_tinf(rt_sv_rho400, rt_sv_TJ70, r_sc_arr[:, i400])
            ft_scen[k] = dict(rho400=r_sc_arr[:, i400], TJ70=_TJ70_sc,
                              TJ70_lm=lm_sc_arr)

        # ── Extract forecast block (optional) ───────────────────────────
        if rho_forecast is not None:
            fc = rho_forecast.get('fit') or {}
            fc_t = np.array([_to_days(t) for t in fc.get('time_utc', [])], float)
            fc_rho_main = _as_2d(fc.get('rho_fit'))
            fc_TJ70_main = _as_1d(fc.get('TJ70_K'))
            sp_fc = (rho_forecast.get('spatial') or {}).get('fit') or {}
            fc_lm_main = _as_2d(sp_fc.get('TJ70_lm'))
            fc_scen: dict[str, dict] = {}
            for k in SCENARIOS:
                r_sc = fc.get(f'rho_{k}')
                lm_sc = sp_fc.get(f'TJ70_lm_{k}')
                if r_sc is None:
                    continue
                r_sc_arr = _as_2d(r_sc)
                lm_sc_arr = _as_2d(lm_sc) if lm_sc is not None else fc_lm_main
                _TJ70_sc = _regress_tinf(rt_sv_rho400, rt_sv_TJ70, r_sc_arr[:, i400])
                fc_scen[k] = dict(rho400=r_sc_arr[:, i400], TJ70=_TJ70_sc,
                                  TJ70_lm=lm_sc_arr)
        else:
            fc_t = np.zeros(0, float)
            fc_rho_main = np.zeros((0, alts.size))
            fc_TJ70_main = np.zeros(0, float)
            fc_lm_main = np.zeros((0, 16), float)
            fc_scen = {}

        # ── Merge per source: solver → fit → forecast ───────────────────
        LAT_g, LST_g, w = _sphere_grid()
        self._LAT_g, self._LST_g, self._w = LAT_g, LST_g, w
        self._Y_grid = Y_modes(LAT_g, LST_g)                  # (n_lat, n_lst, 16)
        self._Y_grid_flat = self._Y_grid.reshape(-1, 16)       # (n_grid, 16)

        # Solver segment is identical for every source: normalize once.
        sv_K = self._norm_K(sv_rho400, sv_TJ70, sv_lm)
        self._sources: dict[str, dict] = {}
        for src in ('baseline',) + SCENARIOS:
            if src == 'baseline' or src not in ft_scen:
                ft_r = ft_rho_main[:, i400] if ft_rho_main.size else np.zeros(0)
                ft_T, ft_l = ft_TJ70_main, ft_lm_main
            else:
                ft_r = ft_scen[src]['rho400']; ft_T = ft_scen[src]['TJ70']
                ft_l = ft_scen[src]['TJ70_lm']
            if src == 'baseline' or src not in fc_scen:
                fc_r = fc_rho_main[:, i400] if fc_rho_main.size else np.zeros(0)
                fc_T, fc_l = fc_TJ70_main, fc_lm_main
            else:
                fc_r = fc_scen[src]['rho400']; fc_T = fc_scen[src]['TJ70']
                fc_l = fc_scen[src]['TJ70_lm']
            self._sources[src] = self._pack_source(
                sv_t, sv_rho400, sv_TJ70, sv_lm, sv_K,
                ft_t, ft_r, ft_T, ft_l,
                fc_t, fc_r, fc_T, fc_l)

    # ── Source packer ─────────────────────────────────────────────────────

    def _norm_K(self, rho400, TJ70, lm, chunk: int = 2048) -> NDArray:
        """Per-timestep amplitude K so the sphere-average of
        K*J70(400, TJ70*(1+sum lm*Y)) equals rho400.  Vectorized in chunks."""
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

    def _pack_source(self, sv_t, sv_rho, sv_TJ70, sv_lm, sv_K,
                     ft_t, ft_rho, ft_TJ70, ft_lm,
                     fc_t, fc_rho, fc_TJ70, fc_lm) -> dict:
        """Solver reanalysis up to its edge, then fit, then forecast.

        Fit / forecast rows at or before the solver edge are dropped (the
        fit block also carries hindcast rows over the solver window; using
        them would let scenarios alter the past).  Forecast rows at or
        before the last fit row are dropped too.
        """
        edge = self._solver_edge
        ft_lm = np.asarray(ft_lm, float).reshape(-1, 16)
        fc_lm = np.asarray(fc_lm, float).reshape(-1, 16)
        fk = ft_t > edge
        ft_t, ft_rho, ft_TJ70 = ft_t[fk], ft_rho[fk], ft_TJ70[fk]
        ft_lm = ft_lm[fk] if ft_lm.shape[0] == fk.size else np.zeros((0, 16))
        last = max(edge, float(ft_t.max()) if ft_t.size else edge)
        ck = fc_t > last
        fc_t, fc_rho, fc_TJ70 = fc_t[ck], fc_rho[ck], fc_TJ70[ck]
        fc_lm = fc_lm[ck] if fc_lm.shape[0] == ck.size else np.zeros((0, 16))

        t = np.concatenate([sv_t, ft_t, fc_t])
        rho = np.concatenate([sv_rho, ft_rho, fc_rho])
        TJ70 = np.concatenate([sv_TJ70, ft_TJ70, fc_TJ70])
        lm = np.vstack([sv_lm, ft_lm, fc_lm])
        K = np.concatenate([sv_K,
                            self._norm_K(ft_rho, ft_TJ70, ft_lm),
                            self._norm_K(fc_rho, fc_TJ70, fc_lm)])
        order = np.argsort(t, kind='stable')
        t, rho, TJ70, lm, K = t[order], rho[order], TJ70[order], lm[order], K[order]
        u_t, u_idx = np.unique(t, return_index=True)
        return dict(t=u_t, rho400=rho[u_idx], TJ70_bg=TJ70[u_idx],
                    TJ70_lm=lm[u_idx], K=K[u_idx])

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
        lat_deg, lon_deg : array_like, degrees.
        alt_km : array_like, WGS84 geodetic altitude, km.  Converted
            internally to height above the 6371.010 km sphere that the
            DragTrack products use as their altitude reference.
        source : str, one of ``'baseline'`` (default) or a name in
            :data:`SCENARIOS`.  ``'baseline'`` uses main fit + forecast
            rho / TJ70_lm; scenario names swap the fit + forecast segments
            (solver segment remains reanalysis in every case).

        Returns
        -------
        float or ndarray
            Density in kg / m^3.  Scalar if every input is scalar.  NaN
            for times outside the loaded coverage (see :attr:`coverage`).
        """
        if source not in self._sources:
            raise ValueError(f'unknown source={source!r}; '
                             f'expected baseline or one of {SCENARIOS}')

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
        t_days, ut_h = _vec_time_features(np.ravel(t_arr).tolist())
        lat_f = np.ravel(lat).astype(float)
        lon_f = np.ravel(lon).astype(float)
        alt_f = _geodetic_to_sphere_height(lat_f, np.ravel(alt).astype(float))
        lst_h = (ut_h + lon_f / 15.0) % 24.0

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
            before computing latitude / longitude / altitude.
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
        flat_r = r.reshape(-1, 3)
        t_arr = _as_time_seq(t)
        t_days, ut_h = _vec_time_features(list(t_arr))
        if flat_r.shape[0] != len(t_days):
            if len(t_days) == 1:
                t_days = np.full(flat_r.shape[0], t_days[0])
                ut_h = np.full(flat_r.shape[0], ut_h[0])
            else:
                raise ValueError(
                    f't and r_km lengths mismatch (t={len(t_days)} vs '
                    f'r_km={flat_r.shape[0]})')
        # Rotate to ECEF if inertial.
        if frame == 'EarthMJ2000Eq':
            r_ecef = _eci_to_ecef(flat_r, t_days)
        elif frame == 'EarthFixed':
            r_ecef = flat_r
        else:
            raise ValueError(
                f"frame must be 'EarthMJ2000Eq' or 'EarthFixed' (got {frame!r})")
        # ECEF → WGS84 geodetic.
        lat_deg, lon_deg, alt_km = _ecef_to_geodetic(r_ecef)
        # Call the scalar rho path via the array shape we've been building.
        # We already have t_days available; delegate to .rho().
        out = self.rho(
            [t_arr[0]] * flat_r.shape[0]
            if len(t_arr) == 1 else list(t_arr),
            lat_deg, lon_deg, alt_km, source=source,
        )
        if flat_r.shape[0] == 1 and np.ndim(r_km) == 1:
            return float(out[0]) if hasattr(out, '__len__') else float(out)
        return np.asarray(out).reshape(r.shape[:-1])

    # ── Loaders ───────────────────────────────────────────────────────────

    @classmethod
    def from_files(
        cls,
        rho_realtime: PathLike | Mapping[str, Any],
        rho_forecast: PathLike | Mapping[str, Any] | None = None,
        j70_lut: PathLike | None = None,
        rho_full: PathLike | Mapping[str, Any] | None = None,
    ) -> "DragTrackDensity":
        """Build from filesystem path(s) or already-parsed dict(s)."""
        rt = _load_json(rho_realtime) if not isinstance(rho_realtime, Mapping) \
             else dict(rho_realtime)
        fc = None
        if rho_forecast is not None:
            fc = _load_json(rho_forecast) if not isinstance(rho_forecast, Mapping) \
                 else dict(rho_forecast)
        fl = None
        if rho_full is not None:
            fl = _load_json(rho_full) if not isinstance(rho_full, Mapping) \
                 else dict(rho_full)
        j = J70(j70_lut) if j70_lut is not None else None
        return cls(rho_realtime=rt, rho_forecast=fc, j70=j, rho_full=fl)

    @classmethod
    def from_url(
        cls,
        rho_realtime_url: str,
        rho_forecast_url: str | None = None,
        j70_lut: PathLike | None = None,
        rho_full_url: str | None = None,
    ) -> "DragTrackDensity":
        """Fetch JSON products over HTTP(S)."""
        with urlopen(rho_realtime_url) as r:
            rt = json.load(r)
        fc = None
        if rho_forecast_url:
            with urlopen(rho_forecast_url) as r:
                fc = json.load(r)
        fl = None
        if rho_full_url:
            with urlopen(rho_full_url) as r:
                fl = json.load(r)
        j = J70(j70_lut) if j70_lut is not None else None
        return cls(rho_realtime=rt, rho_forecast=fc, j70=j, rho_full=fl)

    # ── Metadata ──────────────────────────────────────────────────────────

    @property
    def meta(self) -> dict:
        """The ``meta`` block from the ``rho_realtime`` JSON."""
        return dict(self._meta)

    @property
    def coverage(self) -> dict:
        """UTC span covered by the loaded products:
        ``{'solver_start', 'solver_edge', 'end'}`` as ISO strings.  Queries
        outside ``[solver_start, end]`` return NaN."""
        def _iso(d):
            return (_EPOCH_UTC + timedelta(days=float(d))).strftime('%Y-%m-%dT%H:%M:%SZ')
        s = self._sources['baseline']
        return {'solver_start': _iso(s['t'][0]),
                'solver_edge': _iso(self._solver_edge),
                'end': _iso(s['t'][-1])}

    @property
    def altitude_km(self) -> NDArray[np.float64]:
        """Native altitude grid, km (informational — ``rho()`` accepts any alt)."""
        return self._alt_grid.copy()


# ─── Helpers ──────────────────────────────────────────────────────────────

def _regress_tinf(rho_truth: NDArray, TJ70_truth: NDArray,
                   rho_target: NDArray) -> NDArray:
    """Log-linear regression of TJ70 on log10(rho_400) — used internally
    to reconstruct per-scenario background temperature."""
    ok = (np.isfinite(rho_truth) & np.isfinite(TJ70_truth)
          & (rho_truth > 0))
    if int(ok.sum()) < 3:
        return np.full_like(rho_target, np.nan)
    k, b = np.polyfit(np.log10(rho_truth[ok]), TJ70_truth[ok], 1)
    return k * np.log10(np.maximum(rho_target, 1e-30)) + b


def _interp1d(x_new: NDArray, x_ref: NDArray, y_ref: NDArray) -> NDArray:
    """Linear interpolation over the finite reference samples.  NaN
    outside the reference span -- no edge-holding / extrapolation."""
    ok = np.isfinite(y_ref)
    if not np.any(ok):
        return np.full_like(x_new, np.nan, dtype=float)
    return np.interp(x_new, x_ref[ok], y_ref[ok], left=np.nan, right=np.nan)


# Altitude reference of the DragTrack products: height above a sphere of
# this radius (the TLE/solver mean-Earth radius), NOT WGS84 geodetic.
_PRODUCT_SPHERE_R_KM = 6371.010


def _geodetic_to_sphere_height(lat_deg: NDArray, alt_km: NDArray) -> NDArray:
    """WGS84 geodetic altitude -> height above the 6371.010 km sphere
    (the products' altitude reference).  +7 km at the equator, -14 km at
    the poles for the same geodetic altitude."""
    la = np.deg2rad(np.asarray(lat_deg, float))
    s_, c_ = np.sin(la), np.cos(la)
    N = _WGS84_A_KM / np.sqrt(1.0 - _WGS84_E2 * s_ * s_)
    rho_xy = (N + alt_km) * c_
    z = (N * (1.0 - _WGS84_E2) + alt_km) * s_
    return np.hypot(rho_xy, z) - _PRODUCT_SPHERE_R_KM


_WGS84_A_KM = 6378.137
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)   # eccentricity squared


def _gmst_rad(t_days: NDArray[np.float64]) -> NDArray[np.float64]:
    """Greenwich Mean Sidereal Time in radians (IAU-1982 series).

    ``t_days`` is UT1 (approximated by UTC — drag lookup doesn't need
    the sub-second UT1-UTC correction) in days since 1970-01-01 UTC.
    """
    # Days since J2000.0 = 2000-01-01 12:00 UTC = 10957.5 in POSIX-days.
    # T is the full UT1 epoch (time of day included), so the
    # 876600 h·T term already carries the Earth's daily rotation — do NOT
    # add the UT hours again (v0.4.0 did, giving errors up to 180°).
    d = np.asarray(t_days, float) - 10957.5
    T = d / 36525.0
    gmst_s = (67310.54841
              + (876600.0 * 3600.0 + 8640184.812866) * T
              + 0.093104 * T ** 2
              - 6.2e-6 * T ** 3)
    return np.deg2rad((gmst_s / 240.0) % 360.0)


def _eci_to_ecef(r_eci_km: NDArray[np.float64],
                  t_days: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate an inertial (EarthMJ2000Eq) position vector into
    Earth-fixed (ECEF) using GMST.  Precession/nutation not applied —
    accurate to <1 km for LEO drag-lookup purposes."""
    theta = _gmst_rad(t_days)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = r_eci_km[..., 0], r_eci_km[..., 1], r_eci_km[..., 2]
    x_e = c * x + s * y
    y_e = -s * x + c * y
    z_e = z
    return np.stack([x_e, y_e, z_e], axis=-1)


def _ecef_to_geodetic(r_ecef_km: NDArray[np.float64]):
    """WGS84 ECEF (km) → geodetic (lat_deg, lon_deg, alt_km).  Closed
    form via Bowring (1985)."""
    x = np.asarray(r_ecef_km[..., 0], float)
    y = np.asarray(r_ecef_km[..., 1], float)
    z = np.asarray(r_ecef_km[..., 2], float)
    a = _WGS84_A_KM
    e2 = _WGS84_E2
    b = a * np.sqrt(1.0 - e2)
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


def _load_json(fp: PathLike) -> dict:
    p = Path(fp)
    if p.suffix.endswith('.gz'):
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
