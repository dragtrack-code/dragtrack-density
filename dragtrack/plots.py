"""Plots of the DragTrack density products (optional: ``pip install 'dragtrack[plots]'``).

>>> from dragtrack.plots import atmosphere_figure, save_png
>>> fig = atmosphere_figure('dragtrack_rho_realtime.json', 'dragtrack_rho_forecast.json', alt_km=400)
>>> save_png(fig, 'rho.png')

The figure is the DragTrack website's Atmosphere view at one altitude: the global-average density vs time (solver
values with their uncertainty, the realtime fit and its scenarios, and the forecast if given), the solver's
data-cadence strip, the latitude-averaged density / its global mean vs local solar time and the local-time-averaged
one vs latitude (from the products' spatial coefficients, with J70 for the vertical), and the S and G drivers.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np

from .basis import Y_modes
from .j70 import J70

PathLike = Union[str, Path]

# Cadence strip colors (solver bin length at each timestamp).
CADENCE_3H = "#4caf50"         # 3 h subdaily bin
CADENCE_DAILY = "#9e9e9e"      # daily solve value (also fills 3 h slots without a bin)
CADENCE_KEY = (f"cadence  <span style='color:{CADENCE_3H}'>■</span> 3 h"
               f"  <span style='color:{CADENCE_DAILY}'>■</span> daily")

# Jacchia-1970 diurnal pattern, used for the latitude / local-time panels only when a file has no spatial
# coefficients (TJ70_lm).  Amplitude R as in the DragTrack producer.
_J70_M, _J70_N = 2.5, 3.0
_J70_BETA, _J70_GAMMA, _J70_P = np.deg2rad(-37.0), np.deg2rad(+43.0), np.deg2rad(+6.0)
J70_R = 0.424

_J70_CACHE = None


def _plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:  # pragma: no cover
        raise ImportError("dragtrack.plots needs plotly: pip install 'dragtrack[plots]'") from e
    return go, make_subplots


def _j70() -> J70:
    global _J70_CACHE
    if _J70_CACHE is None:
        _J70_CACHE = J70()
    return _J70_CACHE


def solar_dec_deg(doy):
    """Solar declination (deg) as a function of day of year."""
    return 23.45 * np.sin(np.deg2rad(360.0 * (doy - 81) / 365.25))


def tinf_ratio(lat_deg, lst_hr, delta_deg, R=J70_R):
    """Jacchia-1970 T_inf / T_c at (lat, lst) for solar declination delta."""
    phi = np.deg2rad(np.asarray(lat_deg, float))
    delta = np.deg2rad(delta_deg)
    H = np.deg2rad((np.asarray(lst_hr, float) - 12.0) * 15.0)
    eta = 0.5 * np.abs(phi - delta)
    theta = 0.5 * np.abs(phi + delta)
    tau = H + _J70_BETA + _J70_P * np.sin(H + _J70_GAMMA)
    tau = np.mod(tau + np.pi, 2 * np.pi) - np.pi
    return (1.0 + R * np.sin(theta) ** _J70_M
            + R * (np.cos(eta) ** _J70_M - np.sin(theta) ** _J70_M) * np.cos(tau / 2.0) ** _J70_N)


def cadence_colors(bin_days) -> list:
    """Cadence-strip colour per solver value: 3 h subdaily bin or daily solve."""
    bh = np.array([float(v) if v is not None else np.nan for v in bin_days], dtype=float) * 24.0
    return np.where(bh <= 6.0, CADENCE_3H, CADENCE_DAILY).tolist()


def _log_interp_at(row, alt_km, alts):
    """Log-linear interp of rho at alt_km using alt grid, mask nan/nonpos."""
    row = np.asarray(row, dtype=float)
    ok = np.isfinite(row) & (row > 0)
    if ok.sum() < 2:
        return np.nan
    return float(np.exp(np.interp(alt_km, alts[ok], np.log(row[ok]))))


def _splice_block(rt: dict, fc: dict, t0: str) -> dict:
    """Realtime rows before t0, then the forecast rows.  Every time-indexed realtime key gets the same length;
    keys missing from the forecast are padded with None (row width taken from the realtime array)."""
    t_rt = rt.get("time_utc") or []
    keep = sum(1 for t in t_rt if t < t0)
    n_fc = len(fc.get("time_utc") or [])
    out = {}
    for k, rv in rt.items():
        if not (isinstance(rv, list) and len(rv) == len(t_rt)):
            out[k] = rv
            continue
        fv = fc.get(k)
        if isinstance(fv, list) and len(fv) == n_fc:
            out[k] = rv[:keep] + fv
        else:
            row = next((r for r in rv if isinstance(r, list)), None)
            pad = [None] * len(row) if row is not None else None
            out[k] = rv[:keep] + [pad] * n_fc
    return out


def splice_forecast(rho_realtime: Mapping[str, Any], rho_forecast: Mapping[str, Any]) -> dict:
    """rho_realtime with the rho_forecast rows spliced onto fit and spatial.fit: the realtime fit rows before the
    forecast's first time, then the forecast rows."""
    d = dict(rho_realtime)
    fc_fit = rho_forecast.get("fit") or {}
    if not fc_fit.get("time_utc"):
        return d
    t0 = fc_fit["time_utc"][0]
    d["fit"] = _splice_block(d.get("fit") or {}, fc_fit, t0)
    sp_rt = (d.get("spatial") or {}).get("fit")
    sp_fc = (rho_forecast.get("spatial") or {}).get("fit")
    if sp_rt and sp_fc:
        d["spatial"] = {**d["spatial"], "fit": _splice_block(sp_rt, sp_fc, t0)}
    return d


def _load(x) -> dict:
    """A product dict as is; a path or an http(s) URL loaded as JSON."""
    if isinstance(x, Mapping):
        return dict(x)
    from .model import _load_json
    return _load_json(x)


def atmosphere_figure(rho_realtime, rho_forecast=None, alt_km: float = 400.0, now: datetime | None = None):
    """Plotly figure of a rho_realtime file (dict, path or URL), with a rho_forecast (dict, path or URL) spliced on if
    given, at ``alt_km`` (geodetic; log-interpolated across the product's altitude grid).  ``now`` (UTC, default
    the current time) is marked on every panel."""
    d = _load(rho_realtime)
    if rho_forecast is not None:
        d = splice_forecast(d, _load(rho_forecast))
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    return _build(d, float(alt_km), now)


def save_png(fig, path: PathLike, width: int = 1300, height: int | None = None, scale: float = 1.3) -> Path:
    """Write ``fig`` as a PNG (needs kaleido: ``pip install 'dragtrack[plots]'``); returns the path."""
    path = Path(path)
    fig.write_image(str(path), format="png", width=width, height=height or fig.layout.height, scale=scale)
    return path


def _spatial_hm_grid(Tc_target, rho400_target, t_target_dt,
                      alt_sel, j70, LAT_g, LST_g, cos_w,
                      TJ70_lm_target=None,
                      hold_at_edge: bool = False):
    """ρ/⟨ρ⟩ heatmap from TJ70_lm SH-16 spatial coefficients.

    Consumer of the new schema (v0.8.0+):
        TJ70_lm_target: (N, 16) fractional SH coefficients per epoch (from
        spatial.{solver,fit}[.<scenario>].TJ70_lm).  Order matches the
        provenance mode_order: Y00, Y10, Y11c, Y11s, Y20, Y21c, Y21s, Y22c,
        Y22s, Y30, Y31c, Y31s, Y32c, Y32s, Y33c, Y33s.
    Synthesis per epoch:
        Tinf(lat,lst) = T_c(t) · (1 + Σ_k TJ70_lm[t,k]·Y_k(lat,lst))
        K             = ρ_bg / ⟨J70(400, Tinf)⟩
        ρ_alt         = K · J70(alt, Tinf)
    """
    N = len(t_target_dt)
    j70_rho = j70
    Tc = np.asarray(Tc_target, float); rho400 = np.asarray(rho400_target, float)
    TJ70_lm = None
    if TJ70_lm_target is not None:
        TJ70_lm = np.asarray(TJ70_lm_target, float)
        if TJ70_lm.ndim != 2 or TJ70_lm.shape[1] != 16:
            TJ70_lm = None
    lat_avg = np.full((LAT_g.shape[1], N), np.nan)   # (n_lst, N)
    lst_avg = np.full((LAT_g.shape[0], N), np.nan)   # (n_lat, N)
    # SH-16 basis on the heatmap grid (the producer's own basis).
    _Y_STACK = np.moveaxis(Y_modes(LAT_g, LST_g), -1, 0)  # (16, nlat, nlst)
    for i in range(N):
        if not (np.isfinite(Tc[i]) and np.isfinite(rho400[i]) and rho400[i] > 0):
            continue
        # Tinf(lat,lst) synthesis: T_c · (1 + Σ β_k · Y_k)
        if TJ70_lm is not None and i < len(TJ70_lm) and np.isfinite(TJ70_lm[i]).all():
            frac = (TJ70_lm[i][:, None, None] * _Y_STACK).sum(axis=0)
            Tinf = Tc[i] * (1.0 + frac)
        else:
            # Fallback when TJ70_lm is missing: full-amplitude J70 clim pattern
            doy = t_target_dt[i].timetuple().tm_yday
            delta = solar_dec_deg(doy)
            ratio = tinf_ratio(LAT_g, LST_g, delta)
            w_map = cos_w[:, None] * np.ones_like(ratio)
            r_mean = float((ratio * w_map).sum() / w_map.sum())
            Tinf = Tc[i] * ratio / r_mean
        # K normalization so global-mean rho400 = rho400_target[i].
        rho400_grid = j70_rho(np.full_like(Tinf, 400.0), Tinf)
        w_map = cos_w[:, None] * np.ones_like(rho400_grid)
        mean_400 = float((rho400_grid * w_map).sum() / w_map.sum())
        if not (np.isfinite(mean_400) and mean_400 > 0): continue
        K = rho400[i] / mean_400
        rho_alt = K * j70_rho(np.full_like(Tinf, float(alt_sel)), Tinf)
        rho_by_lst = (rho_alt * cos_w[:, None]).sum(axis=0)
        mean_g = float(rho_by_lst.mean())
        if not (np.isfinite(mean_g) and mean_g > 0): continue
        lat_avg[:, i] = rho_by_lst / mean_g
        lst_avg[:, i] = rho_alt.mean(axis=1) / mean_g
    return lat_avg, lst_avg


def _spatial_heatmap_inputs(d: dict, n_lst: int = 24, n_lat: int = 36):
    """Static setup for the J70-clim heatmap: LST/lat axes + grid + cos(lat)."""
    lst_edges = np.linspace(0.0, 24.0, n_lst + 1)
    lst_c = 0.5 * (lst_edges[:-1] + lst_edges[1:])
    lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    LST_g, LAT_g = np.meshgrid(lst_c, lat_c, indexing="xy")
    cos_w = np.cos(np.deg2rad(LAT_g[:, 0]))
    cos_w = cos_w / cos_w.sum()
    return dict(lst_c=lst_c, lat_c=lat_c, LAT_g=LAT_g, LST_g=LST_g, cos_w=cos_w)


def _build(d: dict, alt_sel: float, now: datetime):
    """The figure for a rho_realtime dict `d` (forecast already spliced in, if any)."""
    go, make_subplots = _plotly()
    j70 = _j70()
    _has_spatial = bool(d.get("spatial"))
    alts = np.array(d["altitude_km"], dtype=float)
    alt_label = f"{alt_sel:.0f}"

    edge = datetime.fromisoformat(d["meta"]["tle_edge_utc"].replace("Z", "+00:00")).replace(tzinfo=None)
    _ref_iso = d["meta"].get("ref_time_utc", d["meta"]["tle_edge_utc"])
    ref_time_marker = datetime.fromisoformat(_ref_iso.replace("Z", "+00:00")).replace(tzinfo=None)
    t_solver = np.array([datetime.fromisoformat(t.replace("Z", "+00:00")).replace(tzinfo=None)
                         for t in d["solver"]["time_utc"]])
    # Log-interp rho at arbitrary alt_sel from each timestep's alt grid
    rho_solver_400 = np.array([_log_interp_at(row, alt_sel, alts)
                                for row in d["solver"]["rho"]], dtype=float)

    t_fit = np.array([datetime.fromisoformat(t.replace("Z", "+00:00")).replace(tzinfo=None)
                      for t in d["fit"]["time_utc"]])

    def _col_interp(key, alt_km):
        return np.array([np.nan if row is None else _log_interp_at(row, alt_km, alts)
                          for row in d["fit"][key]], dtype=float)

    rho_fit_400   = _col_interp("rho_fit",         alt_sel)
    unc_fit_400   = _col_interp("uncertainty_fit", alt_sel)
    S_fit         = np.array([np.nan if v is None else v for v in d["fit"]["S"]], dtype=float)
    G_fit         = np.array([np.nan if v is None else v for v in d["fit"]["G"]], dtype=float)

    SCEN_MAP = [
        ("solar_only", "Solar Only", "springgreen"),
        ("quiet",      "Quiet",      "yellowgreen"),
        ("median",     "Median",     "gold"),
        ("moderate",   "Storm",      "darkorange"),
        ("extreme",    "Severe",     "deeppink"),
    ]
    scen_rho = {}
    SCEN = []
    for key, lbl, col in SCEN_MAP:
        scen_rho[lbl] = _col_interp(f"rho_{key}", alt_sel)
        SCEN.append((lbl, col, None))

    mask_hind = t_fit <= edge
    mask_fwd  = t_fit >  edge

    # Row-index mapping: enables 2-row heatmap insert between density and cadence.
    if _has_spatial:
        _row = {"rho": 1, "hm_lat": 2, "hm_lst": 3, "cad": 4, "S": 5, "G": 6}
        _row_heights = [3.5, 1.4, 1.4, 0.10, 0.9, 1.1]
    else:
        _row = {"rho": 1, "hm_lat": None, "hm_lst": None, "cad": 2, "S": 3, "G": 4}
        _row_heights = [3.5, 0.10, 0.9, 1.1]
    fig = make_subplots(rows=len(_row_heights), cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=_row_heights)

    unc_solver_400 = np.array([_log_interp_at(row, alt_sel, alts)
                                 for row in d["solver"]["uncertainty"]], dtype=float)
    band_lo_h = rho_solver_400 * (1 - unc_solver_400)
    band_hi_h = rho_solver_400 * (1 + unc_solver_400)
    fig.add_trace(go.Scatter(x=t_solver, y=band_lo_h, mode="lines",
                             line=dict(color="deepskyblue", width=0),
                             showlegend=False, hoverinfo="skip"), row=_row["rho"], col=1)
    fig.add_trace(go.Scatter(x=t_solver, y=band_hi_h, mode="lines",
                             line=dict(color="deepskyblue", width=0),
                             fill="tonexty", fillcolor="rgba(0,180,255,0.45)",
                             showlegend=False, hoverinfo="skip"), row=_row["rho"], col=1)

    fig.add_trace(go.Scatter(x=t_solver, y=rho_solver_400, mode="lines",
                             line=dict(color="deepskyblue", width=2),
                             showlegend=False,
                             hovertemplate="%{x|%Y-%m-%d %H:%M}<br>ρ=%{y:.3e}<extra>solver</extra>"),
                  row=_row["rho"], col=1)

    t_truth_edge = t_solver[-1]
    y_seed = float(np.interp(
        (t_truth_edge - t_fit[0]).total_seconds(),
        np.array([(t - t_fit[0]).total_seconds() for t in t_fit]),
        rho_fit_400,
    ))
    ext_mask = t_fit > t_truth_edge
    t_ext = np.concatenate([[t_truth_edge], t_fit[ext_mask]])
    y_ext = np.concatenate([[y_seed],       rho_fit_400[ext_mask]])
    sig_ext_arr = np.concatenate([
        [unc_fit_400[int(np.argmin(np.abs(
            np.array([(t - t_truth_edge).total_seconds() for t in t_fit]))))]],
        unc_fit_400[ext_mask],
    ])
    fig.add_trace(go.Scatter(x=t_ext, y=y_ext * (1 - sig_ext_arr), mode="lines",
                             line=dict(color="white", width=0),
                             showlegend=False, hoverinfo="skip"), row=_row["rho"], col=1)
    fig.add_trace(go.Scatter(x=t_ext, y=y_ext * (1 + sig_ext_arr), mode="lines",
                             line=dict(color="white", width=0),
                             fill="tonexty", fillcolor="rgba(255,255,255,0.35)",
                             showlegend=False, hoverinfo="skip"), row=_row["rho"], col=1)
    fig.add_trace(go.Scatter(x=t_ext, y=y_ext, mode="lines",
                             line=dict(color="white", width=2, dash="dash"),
                             showlegend=False,
                             hovertemplate="%{x|%Y-%m-%d %H:%M}<br>ρ=%{y:.3e}<extra>extrap</extra>"),
                  row=_row["rho"], col=1)

    fit_win_mask = mask_hind & np.isfinite(rho_fit_400)
    fig.add_trace(go.Scatter(
        x=t_fit, y=np.where(fit_win_mask, rho_fit_400, np.nan),
        mode="lines", line=dict(color="white", width=1.5, dash="dot"),
        showlegend=False,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>ρ_fit=%{y:.3e}<extra>fit-window</extra>"),
        row=_row["rho"], col=1)

    # WCAG 1.4.1 redundant encoding: hindcast portion stays solid (all
    # scenarios overlap here anyway); forecast portion uses a per-scenario
    # dash pattern in addition to color, so users without color perception
    # can still distinguish the 5 forecast bands where they diverge.
    _SCEN_DASH = {'Solar Only': 'solid',    'Quiet':  'dot',
                  'Median':     'dash',     'Storm':  'longdash',
                  'Severe':     'dashdot'}
    sc_endpoints = []
    for lbl, col, _ in SCEN:
        y_solid = np.where(mask_hind, scen_rho[lbl], np.nan)
        y_dash  = np.where(mask_fwd,  scen_rho[lbl], np.nan)
        _fd = _SCEN_DASH.get(lbl, 'dash')
        fig.add_trace(go.Scatter(x=t_fit, y=y_solid, mode="lines",
                                 line=dict(color=col, width=2),
                                 showlegend=False,
                                 hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>ρ=%{{y:.3e}}<extra>{lbl}</extra>"),
                      row=_row["rho"], col=1)
        fig.add_trace(go.Scatter(x=t_fit, y=y_dash, mode="lines",
                                 line=dict(color=col, width=2.4, dash=_fd),
                                 showlegend=False,
                                 hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>ρ=%{{y:.3e}}<extra>{lbl}</extra>"),
                      row=_row["rho"], col=1)
        fin = np.isfinite(scen_rho[lbl])
        if fin.any():
            last = int(np.where(fin)[0][-1])
            sc_endpoints.append((lbl, col, t_fit[last], float(scen_rho[lbl][last])))

    fig.add_annotation(xref="x domain", yref="y domain", xanchor="left", yanchor="bottom",
                       showarrow=False, x=0.01, y=0.03, text="— Atm Solve",
                       font=dict(color="deepskyblue", size=14, family="Arial"))
    fig.add_annotation(xref="x domain", yref="y domain", xanchor="left", yanchor="bottom",
                       showarrow=False, x=0.01, y=0.10, text="-- Atm Fit Extrap",
                       font=dict(color="white", size=14, family="Arial"))

    _all_y = [v for _,_,_,v in sc_endpoints if v > 0]
    _all_y += [v for v in rho_fit_400 if np.isfinite(v) and v > 0]
    if _all_y:
        y_min = np.log10(min(_all_y)); y_max = np.log10(max(_all_y))
        rng = max(y_max - y_min, 0.1)
        sc_sorted = sorted(sc_endpoints, key=lambda e: -e[3])
        MIN_GAP = 0.10; PAD = 0.03
        prev = None; placed = []
        for lbl, col, xe, ye in sc_sorted:
            yfrac = (np.log10(ye) - y_min) / rng
            yfrac = float(np.clip(yfrac, PAD, 1 - PAD))
            if prev is not None and prev - yfrac < MIN_GAP:
                yfrac = max(prev - MIN_GAP, PAD)
            prev = yfrac
            placed.append((lbl, col, ye, yfrac))
        prev = None
        for i in range(len(placed) - 1, -1, -1):
            lbl, col, ye, yfrac = placed[i]
            if prev is not None and yfrac - prev < MIN_GAP:
                yfrac = min(prev + MIN_GAP, 1 - PAD)
            placed[i] = (lbl, col, ye, yfrac); prev = yfrac
        for lbl, col, ye, yfrac in placed:
            fig.add_annotation(x=1.025, y=yfrac, xref="paper", yref="y domain",
                               text=f"<b>{lbl}</b><br>ρ={ye:.2e} kg/m³",
                               showarrow=False, xanchor="left", yanchor="middle",
                               align="left", font=dict(color=col, size=11, family="Arial"))

    S_mW = S_fit * 1000.0
    fig.add_trace(go.Scatter(x=t_fit[mask_hind], y=S_mW[mask_hind], mode="lines",
                             line=dict(color="mediumpurple", width=2),
                             showlegend=False,
                             hovertemplate="%{x|%Y-%m-%d %H:%M}<br>S=%{y:.3f} mW/m²<extra></extra>"),
                  row=_row["S"], col=1)
    fig.add_trace(go.Scatter(x=t_fit[mask_fwd], y=S_mW[mask_fwd], mode="lines",
                             line=dict(color="mediumpurple", width=2, dash="dash"),
                             showlegend=False,
                             hovertemplate="%{x|%Y-%m-%d %H:%M}<br>S=%{y:.3f} mW/m²<extra></extra>"),
                  row=_row["S"], col=1)

    # G — plot every non-null value the JSON provides. The file is
    # the source of truth for validity; no plotter-side display cap.
    #   solid where G_fit finite AND t <= tle_edge
    #   dashed where G_fit finite AND t > tle_edge
    g_valid = np.isfinite(G_fit)
    g_solid_mask = g_valid & (t_fit <= edge)
    g_dash_mask  = g_valid & (t_fit >  edge)
    fig.add_trace(go.Scatter(x=t_fit, y=np.where(g_solid_mask, G_fit, np.nan),
                             mode="lines", line=dict(color="mediumpurple", width=2),
                             showlegend=False,
                             hovertemplate="%{x|%Y-%m-%d %H:%M}<br>G=%{y:.2f} nT<extra></extra>"),
                  row=_row["G"], col=1)
    if g_dash_mask.any():
        fig.add_trace(go.Scatter(x=t_fit, y=np.where(g_dash_mask, G_fit, np.nan),
                                 mode="lines",
                                 line=dict(color="mediumpurple", width=2, dash="dash"),
                                 showlegend=False,
                                 hovertemplate="%{x|%Y-%m-%d %H:%M}<br>G=%{y:.2f} nT<extra></extra>"),
                      row=_row["G"], col=1)  # dashed forecast tail

    # "Now" = when the figure is drawn (UTC), after the product's reference time.
    now_marker = now
    for r in range(1, len(_row_heights) + 1):
        fig.add_shape(type="line", x0=ref_time_marker, x1=ref_time_marker,
                      y0=0, y1=1, yref="y domain",
                      line=dict(color="#fbbf24", dash="dash", width=1.2),
                      row=r, col=1)
        fig.add_shape(type="line", x0=now_marker, x1=now_marker,
                      y0=0, y1=1, yref="y domain",
                      line=dict(color="#4ade80", dash="dot", width=1.4),
                      row=r, col=1)

    # ── Heatmap rows: J70-clim recipe (T_c, ρ_bg_400 per region) ─────
    if _has_spatial:
        _sh = _spatial_heatmap_inputs(d)
        lst_c, lat_c = _sh["lst_c"], _sh["lat_c"]
        LAT_g, LST_g, cos_w_hm = _sh["LAT_g"], _sh["LST_g"], _sh["cos_w"]
        _i400 = int(np.argmin(np.abs(alts - 400.0)))

        def _row_col(rows, ci):
            return np.array([np.nan if r is None or r[ci] is None else float(r[ci])
                              for r in rows], float)
        def _f1d(v):
            return np.array([np.nan if x is None else float(x) for x in v], float)
        sv = d["solver"]
        ft = d["fit"]
        sp = (d.get("spatial") or {})
        _sp_sv = sp.get("solver") or {}
        _sp_ft = sp.get("fit") or {}
        rho400_sv = _row_col(sv["rho"], _i400)
        TJ70_sv   = _f1d(sv.get("TJ70_K", [np.nan]*len(sv["time_utc"])))
        rho400_ft = _row_col(ft["rho_fit"], _i400)
        rho400_med = _row_col(ft.get("rho_median") or ft["rho_fit"], _i400)
        TJ70_ft   = _f1d(ft.get("TJ70_K", [np.nan]*len(ft["time_utc"])))
        # median scenario's own TJ70 (older files lack it: fall back to TJ70_K)
        TJ70_med  = _f1d(ft.get("TJ70_K_median") or ft.get("TJ70_K", [np.nan]*len(ft["time_utc"])))
        def _f2d(v):
            if v is None: return None
            return np.array([[np.nan if x is None else float(x) for x in r] for r in v], float)
        TJ70_lm_sv = _f2d(_sp_sv.get("TJ70_lm"))
        TJ70_lm_ft = _f2d(_sp_ft.get("TJ70_lm"))
        TJ70_lm_med = _f2d(_sp_ft.get("TJ70_lm_median"))

        # ---- Region A: solver time base (historic reanalysis) --------------
        if len(t_solver):
            A_lat, A_lst = _spatial_hm_grid(
                TJ70_sv, rho400_sv, t_solver, alt_sel, j70,
                LAT_g, LST_g, cos_w_hm,
                TJ70_lm_target=TJ70_lm_sv)
            t_A = t_solver
        else:
            t_A = np.array([], dtype=object); A_lat = None; A_lst = None

        # ---- Region B: fit rows past solver_edge with finite rho_fit ------
        solver_edge_dt = t_solver[-1] if len(t_solver) else t_fit[0]
        _B_mask = (t_fit > solver_edge_dt) & np.isfinite(rho400_ft)
        if _B_mask.any():
            t_B = t_fit[_B_mask]
            B_lat, B_lst = _spatial_hm_grid(
                TJ70_ft[_B_mask], rho400_ft[_B_mask], t_B, alt_sel, j70,
                LAT_g, LST_g, cos_w_hm,
                TJ70_lm_target=(TJ70_lm_ft[_B_mask] if TJ70_lm_ft is not None else None))
            _last_B_idx = int(np.where(_B_mask)[0][-1])
            fit_edge_dt = t_fit[_last_B_idx]
        else:
            t_B = np.array([], dtype=object); B_lat = None; B_lst = None
            fit_edge_dt = solver_edge_dt

        # ---- Region C: fit rows past fit_edge (uses median scenario) ------
        _C_mask = (t_fit > fit_edge_dt) & np.isfinite(rho400_med)
        if _C_mask.any():
            t_C = t_fit[_C_mask]
            C_lat, C_lst = _spatial_hm_grid(
                TJ70_med[_C_mask], rho400_med[_C_mask], t_C, alt_sel, j70,
                LAT_g, LST_g, cos_w_hm,
                TJ70_lm_target=(TJ70_lm_med[_C_mask] if TJ70_lm_med is not None else None))
        else:
            t_C = np.array([], dtype=object); C_lat = None; C_lst = None

        # Color scales: one per panel, autoscaled to what is shown and
        # symmetric around 1 in log(ρ/⟨ρ⟩) (half and double equally far from
        # white).  Half-range = HM_AUTO_PCT[panel] percentile of |log10 ratio|
        # over every displayed cell of that panel (solver, fit, forecast), so
        # the most extreme cells saturate: 98 for day/night; 90 for latitude,
        # whose extremes are the polar rows that would wash out the rest.
        # Never below HM_MIN_HALF, so a quiet view does not stretch small
        # differences into strong colour.
        HM_AUTO_PCT, HM_MIN_HALF = {"lat": 98.0, "lst": 90.0}, float(np.log10(1.02))

        def _auto_half(arrs, pct):
            parts = [np.abs(np.log10(a[np.isfinite(a) & (a > 0)])).ravel() for a in arrs if a is not None]
            v = np.concatenate(parts) if parts else np.array([])
            return max(float(np.percentile(v, pct)), HM_MIN_HALF) if v.size else 0.1

        _CS = "RdBu_r"
        _hr = {"lat": _auto_half([A_lat, B_lat, C_lat], HM_AUTO_PCT["lat"]),
               "lst": _auto_half([A_lst, B_lst, C_lst], HM_AUTO_PCT["lst"])}
        _NICE_UP = [1.01, 1.02, 1.03, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0]

        def _ticks(h):
            """Up to 3 round ratios above 1, evenly spread in log within ±h,
            and their reciprocals below 1 (symmetric ticks)."""
            up = [r for r in _NICE_UP if np.log10(r) <= h + 1e-9]
            if len(up) > 3:
                pick = []
                for f in (1 / 3, 2 / 3, 1.0):
                    r = min(up, key=lambda x: abs(np.log10(x) - f * h))
                    if r not in pick:
                        pick.append(r)
                up = sorted(pick)
            return [1.0 / r for r in reversed(up)] + [1.0] + up

        def _cbar(panel):
            """Colorbar beside its own subplot, ticks labeled in ρ/⟨ρ⟩."""
            dom = fig.layout[f"yaxis{_row['hm_' + panel]}"].domain
            h = _hr[panel]
            ticks = _ticks(h)
            return dict(title=dict(text="ρ/⟨ρ⟩", side="right", font=dict(size=11)),
                        x=1.02, xanchor="left", y=0.5 * (float(dom[0]) + float(dom[1])), yanchor="middle",
                        len=float(dom[1]) - float(dom[0]), lenmode="fraction", thickness=14,
                        tickvals=[float(np.log10(r)) for r in ticks],
                        ticktext=[f"{r:.3g}" if r >= 1 else f"{r:.2f}" for r in ticks], tickfont=dict(size=10))

        y_lst_sin = np.sin(np.deg2rad(lat_c))   # equal-area lat display

        # Helper to add one (LAT-panel, LST-panel) heatmap pair with given
        # opacity; colorbars only on the region-A traces.  z = log10(ratio);
        # hover shows the ratio itself (customdata).
        def _add_hm_pair(t_axis, arr_lat, arr_lst, opacity, colorbar):
            if arr_lat is None or arr_lst is None or len(t_axis) == 0:
                return
            with np.errstate(divide="ignore", invalid="ignore"):
                z_lat = np.log10(arr_lat); z_lst = np.log10(arr_lst)
            fig.add_trace(go.Heatmap(
                x=t_axis, y=lst_c, z=z_lat, zsmooth=False,
                colorscale=_CS, zmin=-_hr["lat"], zmax=_hr["lat"], zauto=False,
                opacity=opacity, showscale=colorbar,
                colorbar=_cbar("lat") if colorbar else None,
                customdata=arr_lat,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>LST=%{y:.1f} h<br>ρ/⟨ρ⟩=%{customdata:.3f}<extra></extra>",
            ), row=_row["hm_lat"], col=1)
            fig.add_trace(go.Heatmap(
                x=t_axis, y=y_lst_sin, z=z_lst, zsmooth=False,
                colorscale=_CS, zmin=-_hr["lst"], zmax=_hr["lst"], zauto=False,
                opacity=opacity, showscale=colorbar,
                colorbar=_cbar("lst") if colorbar else None,
                customdata=np.dstack([arr_lst, np.tile(lat_c[:, None], (1, len(t_axis)))]),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Lat=%{customdata[1]:.1f}°<br>ρ/⟨ρ⟩=%{customdata[0]:.3f}<extra></extra>",
            ), row=_row["hm_lst"], col=1)

        _add_hm_pair(t_A, A_lat, A_lst, opacity=1.00, colorbar=True)
        _add_hm_pair(t_B, B_lat, B_lst, opacity=0.80, colorbar=False)
        _add_hm_pair(t_C, C_lat, C_lst, opacity=0.60, colorbar=False)

    # ── Row 2: cadence strip (subtle indicator of per-timestamp bin length)
    _bin_days = d["solver"].get("bin_days")
    if _bin_days is not None:
        _bh = np.array([float(v) if v is not None else np.nan for v in _bin_days],
                       dtype=float) * 24.0
        _cad_colors = cadence_colors(_bin_days)
        fig.add_trace(go.Scattergl(
            x=t_solver, y=np.full(len(t_solver), 0.5),
            mode="markers",
            marker=dict(
                size=8, symbol="square",
                color=_cad_colors,
                line=dict(width=0),
            ),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>bin=%{customdata:.1f} h<extra></extra>",
            customdata=_bh,
            showlegend=False,
        ), row=_row["cad"], col=1)
        fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False,
                         zeroline=False, row=_row["cad"], col=1)
        fig.add_annotation(xref="x domain", yref=f"y{_row['cad']} domain",
                           xanchor="left", yanchor="middle",
                           showarrow=False, x=0.005, y=0.5,
                           text=CADENCE_KEY,
                           font=dict(color="rgba(255,255,255,0.55)", size=10,
                                     family="Arial"))

    fig.add_annotation(x=ref_time_marker, y=1.02, yref="y domain", xref="x",
                       text=ref_time_marker.strftime("%Y-%m-%d %H:%M UTC") + " ",
                       showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(color="#fbbf24", size=11, family="Arial"))
    fig.add_annotation(x=now_marker, y=1.02, yref="y domain", xref="x",
                       text=" Now", showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(color="#4ade80", size=12, family="Arial"))

    def _ptitle(text, yref):
        fig.add_annotation(xref="x domain", yref=yref, xanchor="left", yanchor="top",
                           showarrow=False, x=0.005, y=0.97, text=f"<b>{text}</b>",
                           font=dict(color="white", size=16, family="Arial"))
    _ptitle(f"Global-Average Density at {alt_label} km", "y domain")
    if _has_spatial:
        _ptitle("Latitude-Averaged ρ/⟨ρ⟩ vs LST",
                f"y{_row['hm_lat']} domain")
        _ptitle("LST-Averaged ρ/⟨ρ⟩ vs Latitude",
                f"y{_row['hm_lst']} domain")
    _ptitle("Solar Energy",       f"y{_row['S']} domain")
    _ptitle("Geomagnetic Energy", f"y{_row['G']} domain")

    ax = dict(mirror=True, showline=True, linecolor="rgba(255,255,255,0.5)",
              linewidth=1, ticks="inside", ticklen=6, tickwidth=1,
              tickcolor="rgba(255,255,255,0.7)",
              showgrid=True, gridcolor="rgba(255,255,255,0.12)")

    all_y_pos = list(rho_fit_400[np.isfinite(rho_fit_400) & (rho_fit_400 > 0)])
    all_y_pos += list(rho_solver_400[np.isfinite(rho_solver_400) & (rho_solver_400 > 0)])
    for lbl in scen_rho:
        v = scen_rho[lbl]
        all_y_pos += list(v[np.isfinite(v) & (v > 0)])
    if all_y_pos:
        y_min = float(np.log10(min(all_y_pos)))
        y_max = float(np.log10(max(all_y_pos)))
        pad = 0.15
        y1_range = [y_min - pad, y_max + pad]
    else:
        y1_range = None

    # Force x range to include full scenario extent (auto-range sometimes clips)
    x_range = [t_solver[0], max(t_fit[-1], now_marker + timedelta(hours=6))]
    for _r in range(1, len(_row_heights) + 1):
        fig.update_xaxes(tickformat="%Y-%m-%d", range=x_range, **ax, row=_r, col=1)
    fig.update_yaxes(title="ρ (kg/m³)", type="log", exponentformat="power", dtick=0.25,
                     range=y1_range, **ax, row=_row["rho"], col=1)
    fig.update_yaxes(title="S (mW/m²)", **ax, row=_row["S"], col=1)
    fig.update_yaxes(title="G (nT)",    **ax, row=_row["G"], col=1)
    if _has_spatial:
        fig.update_yaxes(title="LST (h)", range=[0, 24], dtick=6,
                         **ax, row=_row["hm_lat"], col=1)
        # Equal-area lat: y = sin(lat_rad); labels are degrees at sin positions.
        _lat_ticks_deg = np.array([-90, -60, -30, 0, 30, 60, 90], dtype=float)
        _lat_ticks_sin = np.sin(np.deg2rad(_lat_ticks_deg))
        fig.update_yaxes(title="Lat (°, equal-area)",
                         range=[-1, 1],
                         tickmode="array",
                         tickvals=_lat_ticks_sin.tolist(),
                         ticktext=[f"{int(d)}" for d in _lat_ticks_deg],
                         **ax, row=_row["hm_lst"], col=1)

    _hdr_text = (f"DragTrack   |   generated_utc: {d['meta']['generated_utc']}"
                 f"   |   tle_edge: {d['meta']['tle_edge_utc']}")
    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Arial", color="white"),
        height=900,
        margin=dict(l=70, r=180, t=40, b=50),
        showlegend=False,
        title=dict(text=_hdr_text, x=0.005, xanchor="left",
                   font=dict(color="#888", size=10, family="monospace")),
    )
    return fig
