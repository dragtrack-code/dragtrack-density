"""Tests for the dragtrack package."""
from __future__ import annotations

import json

import numpy as np
import pytest

import dragtrack
from dragtrack import DragTrackDensity, SCENARIOS
from dragtrack.basis import Y_modes, REG_NAMES
from dragtrack.j70 import J70, calc_Tx

# Reasonable envelope for LEO thermospheric densities (kg/m^3).
_RHO_LO = 1e-16
_RHO_HI = 1e-6


def _posix_days(dt):
    """Days since 1970-01-01 UTC; naive datetimes are UTC."""
    from datetime import timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 86400.0


def test_module_metadata():
    assert dragtrack.__version__
    assert isinstance(REG_NAMES, tuple)
    assert len(REG_NAMES) == 16
    assert REG_NAMES[0] == 'Y00'
    assert 'baseline' not in SCENARIOS
    assert set(SCENARIOS) == {'solar_only', 'quiet', 'moderate',
                              'intense', 'extreme', 'median'}


def test_basis_shape_and_finite():
    lat = np.linspace(-80, 80, 17)
    lst = np.linspace(0, 24, 17, endpoint=False)
    Y = Y_modes(lat, lst)
    assert Y.shape == (17, 16)
    assert np.all(np.isfinite(Y))
    # Y00 identically 1.
    np.testing.assert_allclose(Y[:, 0], 1.0)
    # Y10 == sin(lat).
    np.testing.assert_allclose(Y[:, 1], np.sin(np.deg2rad(lat)))


def test_basis_scalar_input():
    Y = Y_modes(45.0, 12.0)
    assert Y.shape == (16,)


def test_j70_load_and_call():
    j = J70()
    r = j(400.0, 850.0)
    assert float(r) > 0 and np.isfinite(float(r))


def test_j70_vectorized():
    j = J70()
    alt = np.array([300., 400., 500., 600.])
    T = np.full_like(alt, 900.)
    r = j(alt, T)
    assert r.shape == alt.shape
    assert np.all(np.diff(r) < 0)     # rho decreases with alt at fixed TJ70
    assert np.all(np.isfinite(r) & (r > 0))
    T_lo = j(np.full(4, 400.0), np.array([600., 900., 1200., 1500.]))
    assert np.all(np.diff(T_lo) > 0)  # rho at 400 increases with TJ70


def test_j70_nan_outside_table():
    j = J70()
    assert np.isnan(j(50.0, 850.0)) and np.isnan(j(j.z[-1] + 1.0, 850.0))
    assert np.isfinite(j(j.z[0], 850.0)) and np.isfinite(j(j.z[-1], 850.0))


def test_calc_Tx_shape_and_range():
    T = np.array([500., 900., 1500., 2500.])
    Tx = calc_Tx(T)
    assert Tx.shape == T.shape
    assert np.all(Tx > 200) and np.all(Tx < 700)


# ── Density-from-synthetic-dict tests ────────────────────────────────────

def test_density_from_synthetic_dict(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    # Scalar path.
    r = m.rho('2026-09-13T12:00:00Z', 45.0, 0.0, 450.0)
    assert isinstance(r, float)
    assert _RHO_LO < r < _RHO_HI
    # Vector lat.
    lat = np.linspace(-80, 80, 9)
    r2 = m.rho('2026-09-13T12:00:00Z', lat, np.zeros(9), np.full(9, 450.0))
    assert r2.shape == (9,)
    assert np.all(np.isfinite(r2))
    # Vector alt — rho decreases with altitude.
    alt = np.array([300., 400., 500.])
    r3 = m.rho('2026-09-13T12:00:00Z',
               np.full(3, 45.0), np.zeros(3), alt)
    assert r3.shape == (3,)
    assert r3[0] > r3[1] > r3[2]


def test_density_meta(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    assert m.meta.get('schema_version') == '0.9.0-test'
    assert m.meta.get('tle_edge_utc') == '2026-09-13T00:00:00Z'


def test_density_multi_time(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    tps = ['2026-09-13T00:00:00Z', '2026-09-13T12:00:00Z',
           '2026-09-13T21:00:00Z']
    r = m.rho(tps, 45.0, 0.0, 450.0)
    assert r.shape == (3,)
    assert np.all(np.isfinite(r))


def test_density_altitude_400_required(rho_realtime_dict):
    bad = json.loads(json.dumps(rho_realtime_dict))
    bad['altitude_km'] = [250.0, 300.0]
    bad['solver']['rho'] = [row[:2] for row in bad['solver']['rho']]
    with pytest.raises(ValueError, match='400'):
        DragTrackDensity.from_files(bad)


def test_density_missing_solver_raises(rho_realtime_dict):
    bad = json.loads(json.dumps(rho_realtime_dict))
    del bad['solver']
    with pytest.raises(KeyError, match='solver'):
        DragTrackDensity.from_files(bad)


def test_density_missing_spatial_solver_lm_raises(rho_realtime_dict):
    bad = json.loads(json.dumps(rho_realtime_dict))
    del bad['spatial']['solver']['TJ70_lm']
    with pytest.raises(ValueError, match='spatial.solver.TJ70_lm'):
        DragTrackDensity.from_files(bad)


# ── Scenarios ─────────────────────────────────────────────────────────────

def test_scenario_selection(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    # baseline works (inside the fit block, past the solver edge).
    r0 = m.rho('2026-09-14T06:00:00Z', 0.0, 0.0, 400.0, source='baseline')
    assert _RHO_LO < r0 < _RHO_HI
    # every named scenario is accepted.
    for name in SCENARIOS:
        r = m.rho('2026-09-14T06:00:00Z', 0.0, 0.0, 400.0, source=name)
        assert np.isfinite(r) and _RHO_LO < r < _RHO_HI


def test_unknown_source_raises(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    with pytest.raises(ValueError, match='not available'):
        m.rho('2026-09-13T12:00:00Z', 0.0, 0.0, 400.0, source='oops')


# ── Coordinate helpers + rho_from_state ─────────────────────────────────

from dragtrack.model import (_ecef_to_geodetic, _eci_to_ecef, _gmst_rad,
                              _WGS84_A_KM, _WGS84_E2)


def _geodetic_to_ecef(lat_deg, lon_deg, alt_km):
    """Forward WGS84 geodetic → ECEF (test helper only)."""
    lat_r = np.deg2rad(lat_deg); lon_r = np.deg2rad(lon_deg)
    s, c = np.sin(lat_r), np.cos(lat_r)
    N = _WGS84_A_KM / np.sqrt(1 - _WGS84_E2 * s ** 2)
    x = (N + alt_km) * c * np.cos(lon_r)
    y = (N + alt_km) * c * np.sin(lon_r)
    z = (N * (1 - _WGS84_E2) + alt_km) * s
    return np.array([x, y, z])


def test_ecef_to_geodetic_equator():
    lat, lon, alt = _ecef_to_geodetic(np.array([_WGS84_A_KM + 400.0, 0.0, 0.0]))
    assert abs(lat) < 1e-9
    assert abs(lon) < 1e-9
    assert abs(alt - 400.0) < 1e-6


def test_ecef_geodetic_round_trip():
    # Non-trivial point far from equator / pole singularities.
    lat0, lon0, alt0 = 45.123, -120.456, 412.7
    r = _geodetic_to_ecef(lat0, lon0, alt0)
    lat, lon, alt = _ecef_to_geodetic(r)
    assert abs(lat - lat0) < 1e-8
    assert abs(lon - lon0) < 1e-8
    assert abs(alt - alt0) < 1e-6


def test_gmst_matches_sgp4_reference():
    # Independent reference: sgp4's IAU-82 gstime, at several times of day.
    sgp4_prop = pytest.importorskip('sgp4.propagation')
    from datetime import datetime, timezone
    for hh in (0, 3, 6, 9, 12, 15, 18, 21):
        dt = datetime(2026, 9, 24, hh, 17, tzinfo=timezone.utc)
        t_days = dt.timestamp() / 86400.0
        ref = sgp4_prop.gstime(t_days + 2440587.5)
        got = float(_gmst_rad(np.array([t_days]))[0])
        err = (got - ref + np.pi) % (2 * np.pi) - np.pi
        assert abs(err) < 1e-6, f'{hh:02d}UT GMST err {np.degrees(err):.3f} deg'


def test_eci_ecef_invertible():
    # Rotate an ECEF vector out to ECI via forward GMST, then back with
    # the package's ECI→ECEF; must recover the original to numeric noise.
    from datetime import datetime
    t_md = np.array([_posix_days(datetime(2026, 6, 15, 12, 0, 0))])
    r_ecef = np.array([[6778.0, 0.0, 0.0]])
    theta = _gmst_rad(t_md)[0]
    c, s = float(np.cos(theta)), float(np.sin(theta))
    r_eci = np.array([[c * r_ecef[0, 0] - s * r_ecef[0, 1],
                        s * r_ecef[0, 0] + c * r_ecef[0, 1],
                        r_ecef[0, 2]]])
    r_ecef_back = _eci_to_ecef(r_eci, t_md)
    assert np.abs(r_ecef_back - r_ecef).max() < 1e-9


def test_rho_from_state_earthfixed_matches_direct(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    lat, lon, alt = 45.0, -120.0, 400.0
    t = '2026-09-13T12:00:00Z'
    r_direct = m.rho(t, lat, lon, alt)
    r_ecef = _geodetic_to_ecef(lat, lon, alt)
    r_state = m.rho_from_state(t, r_ecef, frame='EarthFixed')
    assert abs(r_state / r_direct - 1.0) < 1e-8


def test_rho_from_state_inertial_matches_direct(rho_realtime_dict):
    from datetime import datetime
    m = DragTrackDensity.from_files(rho_realtime_dict)
    lat, lon, alt = 45.0, -120.0, 400.0
    sgp4_prop = pytest.importorskip('sgp4.propagation')
    # Off-noon on purpose (a UT double-count would vanish at 12 UT).
    t_iso = '2026-09-13T05:40:00Z'
    t_md = np.array([_posix_days(datetime(2026, 9, 13, 5, 40, 0))])
    r_direct = m.rho(t_iso, lat, lon, alt)
    r_ecef = _geodetic_to_ecef(lat, lon, alt)
    # ECEF -> ECI via inverse rotation using an INDEPENDENT GMST (sgp4).
    theta = float(sgp4_prop.gstime(float(t_md[0]) + 2440587.5))
    c, s = np.cos(theta), np.sin(theta)
    r_eci = np.array([c * r_ecef[0] - s * r_ecef[1],
                       s * r_ecef[0] + c * r_ecef[1], r_ecef[2]])
    r_state = m.rho_from_state(t_iso, r_eci, frame='EarthMJ2000Eq')
    assert abs(r_state / r_direct - 1.0) < 1e-8


def test_rho_from_state_bad_frame_raises(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    r_ecef = _geodetic_to_ecef(0.0, 0.0, 400.0)
    with pytest.raises(ValueError, match='frame must be'):
        m.rho_from_state('2026-09-13T12:00:00Z', r_ecef, frame='wat')


def test_rho_from_state_wrong_shape_raises(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    with pytest.raises(ValueError, match='size 3'):
        m.rho_from_state('2026-09-13T12:00:00Z', np.array([1.0, 2.0]))


def test_rho_from_state_vs_sgp4():
    """Cross-check ECI→geodetic against sgp4's own conversion."""
    pytest.importorskip('sgp4')
    from sgp4.api import Satrec, WGS84
    from sgp4 import conveniences as _conv
    from datetime import datetime, timezone
    # A generic ISS-ish TLE.
    l1 = '1 25544U 98067A   26256.50000000  .00000000  00000-0  00000-0 0  9997'
    l2 = '2 25544  51.6400   0.0000 0001000   0.0000   0.0000 15.50000000000019'
    sat = Satrec.twoline2rv(l1, l2, WGS84)
    # Pick a time within the TLE's validity.
    dt = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)
    jd, fr = _conv.jday_datetime(dt)
    e, r_eci, _ = sat.sgp4(jd, fr)
    if e:
        pytest.skip(f'sgp4 error {e}')
    r_eci = np.asarray(r_eci, float).reshape(1, 3)   # km, TEME (approx J2000 for LEO drag)
    t_md = np.array([_posix_days(dt.replace(tzinfo=None))])
    r_ecef = _eci_to_ecef(r_eci, t_md)
    lat_ours, lon_ours, alt_ours = _ecef_to_geodetic(r_ecef[0])
    # sgp4 has no direct geodetic helper; use a loose spherical-Earth band.
    R_earth = 6378.137
    r_mag = float(np.linalg.norm(r_eci))
    alt_expected = r_mag - R_earth       # spherical Earth approx
    assert abs(alt_ours - alt_expected) < 25.0, \
        f'alt mismatch: {alt_ours=} vs sph-Earth {alt_expected=}'
    assert -90 <= lat_ours <= 90
    assert -180 <= lon_ours <= 180
    assert 200 < alt_ours < 800


# ── Real-JSON round-trip (optional) ──────────────────────────────────────

def test_real_json_roundtrip(real_rho_realtime):
    m = DragTrackDensity.from_files(str(real_rho_realtime))
    r = m.rho(m.coverage['solver_edge'], 0.0, 0.0, 400.0)
    assert isinstance(r, float)
    assert _RHO_LO < r < _RHO_HI
    assert m.meta


# ── coverage, solver precedence, history, altitude reference ────

def _set_fit(d, times, rho_fit, **scen_rho):
    """Rebuild d's fit + spatial.fit on `times`: every scenario gets its three
    arrays (rho from scen_rho, else rho_fit), TJ70 850 K, flat pattern."""
    blank = [[0.0] * 16 for _ in times]
    ft = {'time_utc': times, 'rho_fit': rho_fit, 'TJ70_K': [850.0] * len(times)}
    sp = {'time_utc': times, 'TJ70_lm': blank}
    for k in SCENARIOS:
        ft[f'rho_{k}'] = scen_rho.get(k, rho_fit)
        ft[f'TJ70_K_{k}'] = [850.0] * len(times)
        sp[f'TJ70_lm_{k}'] = blank
    d['fit'], d['spatial']['fit'] = ft, sp
    return d


def _with_hindcast_scenario(rt):
    """Copy of the synthetic realtime whose fit block also spans the solver
    window (like real files) with a 'extreme' scenario 3x baseline."""
    import copy
    d = copy.deepcopy(rt)
    hind = [f'2026-09-13T{h:02d}:00:00Z' for h in range(1, 24, 3)]
    rows = [d['solver']['rho'][0]] * len(hind) + d['fit']['rho_fit']
    return _set_fit(d, hind + d['fit']['time_utc'], rows,
                    extreme=[[3 * x for x in r] for r in rows])


def test_scenarios_do_not_change_the_past(rho_realtime_dict):
    m = DragTrackDensity.from_files(_with_hindcast_scenario(rho_realtime_dict))
    t_past = '2026-09-13T10:00:00Z'          # inside solver window
    b = m.rho(t_past, 0.0, 0.0, 400.0)
    x = m.rho(t_past, 0.0, 0.0, 400.0, source='extreme')
    assert abs(x / b - 1.0) < 1e-12
    t_future = '2026-09-14T06:00:00Z'        # after solver edge
    b2 = m.rho(t_future, 0.0, 0.0, 400.0)
    x2 = m.rho(t_future, 0.0, 0.0, 400.0, source='extreme')
    assert x2 > 2.5 * b2


def test_nan_outside_coverage(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    assert np.isnan(m.rho('2020-01-01T00:00:00Z', 0.0, 0.0, 400.0))
    assert np.isnan(m.rho('2027-01-01T00:00:00Z', 0.0, 0.0, 400.0))
    cov = m.coverage
    assert cov['solver_start'] == '2026-09-13T00:00:00Z'
    assert cov['solver_edge'] == '2026-09-13T21:00:00Z'


def test_rho_full_extends_history(rho_realtime_dict):
    import copy
    full = copy.deepcopy(rho_realtime_dict)
    hist = [f'2026-06-{d:02d}T12:00:00Z' for d in range(1, 11)]
    full['solver']['time_utc'] = hist
    full['solver']['rho'] = [full['solver']['rho'][0]] * len(hist)
    full['solver']['TJ70_K'] = [850.0] * len(hist)
    full['spatial'] = {'solver': {'time_utc': hist,
                                  'TJ70_lm': [[0.0] * 16 for _ in hist]}}
    t = '2026-06-05T00:00:00Z'
    assert np.isnan(DragTrackDensity.from_files(rho_realtime_dict).rho(t, 0, 0, 400))
    m = DragTrackDensity.from_files(rho_realtime_dict, rho_full=full)
    assert np.isfinite(m.rho(t, 0.0, 0.0, 400.0))
    assert m.coverage['solver_start'] == '2026-06-01T12:00:00Z'


# ── J70 Tx form, geodetic height, LUT units ────────────────────

def test_calc_Tx_is_j70_form():
    T = 1000.0
    expect = 444.3807 + 0.02385 * T - 392.8292 * np.exp(-0.0021357 * T)
    assert abs(float(calc_Tx(T)) - expect) < 1e-9
    assert abs(float(calc_Tx(T)) - 421.8130) < 1e-3


def test_j70_units_kg_m3():
    # ~1e-12 kg/m^3 at 400 km, 850 K (the raw LUT file is g/cm^3).
    assert 1e-13 < float(J70()(400.0, 850.0)) < 1e-11


def test_rho_evaluated_at_geodetic_height(rho_realtime_dict):
    """rho(lat, alt) == K * J70(alt, TJ70_pt) with alt the geodetic height
    used unchanged at every latitude (no sphere-height conversion)."""
    import copy
    from dragtrack.basis import Y_modes
    from dragtrack.model import _apparent_lst_h, _to_days
    # Blank pattern: result must be latitude-independent at fixed geodetic
    # height.
    m0 = DragTrackDensity.from_files(rho_realtime_dict)
    t = '2026-09-13T12:00:00Z'          # exactly on a solver sample
    r = [m0.rho(t, lat, 0.0, 450.0) for lat in (0.0, 45.0, 80.0, 90.0)]
    np.testing.assert_allclose(r, r[0], rtol=1e-12)
    # Non-trivial pattern: compare against the core J70 call at alt_km.
    d = copy.deepcopy(rho_realtime_dict)
    lm = [0.0] * 16
    lm[1], lm[2], lm[4] = 0.05, 0.10, -0.03       # Y10, Y11c, Y20
    d['spatial']['solver']['TJ70_lm'] = [lm] * len(d['solver']['time_utc'])
    m = DragTrackDensity.from_files(d)
    s = m._sources['baseline']
    i = 4                                         # 12:00 row
    for lat in (0.0, 45.0, 80.0, -89.0):
        for alt in (300.0, 450.0):
            got = m.rho(t, lat, 30.0, alt)
            lst = float(_apparent_lst_h(np.array([_to_days(t)]), np.array([30.0]))[0])
            T_pt = s['TJ70_bg'][i] * (1.0 + Y_modes(lat, lst) @ s['TJ70_lm'][i])
            want = s['K'][i] * float(m.j70(alt, T_pt))
            assert abs(got / want - 1.0) < 1e-12, (lat, alt, got, want)


# ── coverage['end'] is the baseline end, per-source ends reported ──

def test_coverage_end_matches_baseline(rho_realtime_dict):
    """Scenarios run past the baseline rho_fit horizon (real files: ~5 d vs
    ~2 d).  coverage['end'] must be the baseline end; rho() there is finite
    and NaN 1 h later, while the scenario keeps going."""
    import copy
    from datetime import datetime, timedelta
    d = copy.deepcopy(rho_realtime_dict)
    ft = d['fit']
    row = ft['rho_fit'][0]
    extra = [f'2026-09-14T{h:02d}:00:00Z' for h in range(12, 24, 3)]
    _set_fit(d, ft['time_utc'] + extra,
             ft['rho_fit'] + [[None] * len(row)] * len(extra),
             extreme=[[3 * x for x in r] for r in ft['rho_fit']]
             + [[3 * x for x in row]] * len(extra))
    m = DragTrackDensity.from_files(d)
    cov = m.coverage
    assert cov['end'] == '2026-09-14T09:00:00Z'
    assert cov['end_by_source']['baseline'] == cov['end']
    assert cov['end_by_source']['extreme'] == '2026-09-14T21:00:00Z'
    assert set(cov['end_by_source']) == {'baseline', *SCENARIOS}
    t_end = datetime.fromisoformat(cov['end'].replace('Z', '+00:00'))
    assert np.isfinite(m.rho(t_end, 0.0, 0.0, 400.0))
    assert np.isnan(m.rho(t_end + timedelta(hours=1), 0.0, 0.0, 400.0))
    assert np.isfinite(m.rho(t_end + timedelta(hours=1), 0.0, 0.0, 400.0,
                             source='extreme'))


# ── every time form is UTC ──

def test_time_forms_are_utc(rho_realtime_dict):
    """Naive strings / datetimes are UTC; offset-aware inputs are converted."""
    from datetime import datetime, timedelta, timezone
    m = DragTrackDensity.from_files(rho_realtime_dict)
    ref = m.rho('2026-09-13T12:00:00Z', 45.0, 30.0, 450.0)
    assert np.isfinite(ref)
    same = [
        '2026-09-13T12:00:00',
        '2026-09-13T14:00:00+02:00',
        datetime(2026, 9, 13, 12),
        datetime(2026, 9, 13, 7, tzinfo=timezone(timedelta(hours=-5))),
    ]
    for t in same:
        assert m.rho(t, 45.0, 30.0, 450.0) == pytest.approx(ref, rel=1e-12), t


# ── apparent local solar time ──

def test_apparent_lst_equation_of_time():
    """Apparent minus mean solar time = equation of time (almanac values:
    ~+16.4 min early Nov, ~-14.2 min mid Feb, ~0 mid Apr / mid Jun)."""
    from datetime import datetime
    from dragtrack.model import _apparent_lst_h
    for date, eot_min in (((2026, 11, 3), 16.4), ((2026, 2, 11), -14.2),
                          ((2026, 4, 15), 0.0), ((2026, 6, 13), 0.0)):
        t = np.array([_posix_days(datetime(*date, 12))])
        for lon in (-120.0, 0.0, 75.0):
            mean = (12.0 + lon / 15.0) % 24.0
            got = float(_apparent_lst_h(t, np.array([lon]))[0])
            d_min = ((got - mean + 12.0) % 24.0 - 12.0) * 60.0
            assert abs(d_min - eot_min) < 0.7, (date, lon, d_min)


# ── load_sats ──

def test_load_sats():
    from dragtrack import load_sats
    d = {'meta': {}, 'time_utc_daily': ['2026-09-01', '2026-09-02', '2026-09-03'],
         'sats': [{'norad_id': 25544, 'name': 'ISS', 'inc_deg': 51.6,
                   'bc_inv_m2_per_kg': [0.01, None, 0.012],
                   'rho': [1e-12, 2e-12, None]},
                  {'norad_id': 7, 'name': None, 'inc_deg': 98.0,
                   'bc_inv_m2_per_kg': [None, None, None], 'rho': [None] * 3}]}
    sats = load_sats(d)
    assert set(sats) == {25544, 7}
    s = sats[25544]
    assert s['name'] == 'ISS' and s['inc_deg'] == 51.6
    assert s['date'].dtype == np.dtype('datetime64[D]')
    assert str(s['date'][1]) == '2026-09-02'
    np.testing.assert_array_equal(np.isnan(s['bc_inv_m2_per_kg']), [False, True, False])
    assert s['bc_inv_m2_per_kg'][2] == 0.012
    assert np.all(np.isnan(sats[7]['rho']))
    with pytest.raises(KeyError):
        load_sats({'meta': {}})
    bad = dict(d, sats=[dict(d['sats'][0], rho=[1e-12])])
    with pytest.raises(ValueError):
        load_sats(bad)


# ── File consistency checks ──

def test_scenario_needs_published_tj70(rho_realtime_dict):
    bad = json.loads(json.dumps(rho_realtime_dict))
    del bad['fit']['TJ70_K_extreme']
    with pytest.raises(KeyError, match='predates'):
        DragTrackDensity.from_files(bad)


def test_scenario_missing_pattern_raises(rho_realtime_dict):
    bad = json.loads(json.dumps(rho_realtime_dict))
    del bad['spatial']['fit']['TJ70_lm_extreme']
    with pytest.raises(ValueError, match='TJ70_lm_extreme'):
        DragTrackDensity.from_files(bad)


def test_absent_scenario_not_available(rho_realtime_dict):
    d = json.loads(json.dumps(rho_realtime_dict))
    del d['fit']['rho_extreme']
    m = DragTrackDensity.from_files(d)
    with pytest.raises(ValueError, match='not available'):
        m.rho('2026-09-14T06:00:00Z', 0.0, 0.0, 400.0, source='extreme')


def test_spatial_times_must_match(rho_realtime_dict):
    bad = json.loads(json.dumps(rho_realtime_dict))
    t = bad['spatial']['solver']['time_utc']
    bad['spatial']['solver']['time_utc'] = t[1:] + t[:1]
    with pytest.raises(ValueError, match='time_utc'):
        DragTrackDensity.from_files(bad)


def test_rho_full_altitude_grid_must_match(rho_realtime_dict):
    full = json.loads(json.dumps(rho_realtime_dict))
    full['altitude_km'] = [a + 1.0 for a in full['altitude_km']]
    with pytest.raises(ValueError, match='altitude_km'):
        DragTrackDensity.from_files(rho_realtime_dict, full)
