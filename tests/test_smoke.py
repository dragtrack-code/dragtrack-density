"""Smoke tests for the dragtrack package (v0.2+ schema, SH-16 spatial)."""
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
    assert np.all(np.diff(r) < 0)     # rho decreases with alt at fixed Tinf
    assert np.all(np.isfinite(r) & (r > 0))
    T_lo = j(np.full(4, 400.0), np.array([600., 900., 1200., 1500.]))
    assert np.all(np.diff(T_lo) > 0)  # rho at 400 increases with Tinf


def test_j70_clips_out_of_range_alt():
    j = J70()
    r_lo = j(50.0, 850.0)                 # below LUT floor → clip
    r_at = j(j.z[0], 850.0)
    np.testing.assert_allclose(r_lo, r_at)


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
    assert m.meta.get('schema_version') == '0.8.0-test'
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
    del bad['spatial']['solver']['Tinf_lm']
    with pytest.raises(ValueError, match='current DragTrack realtime'):
        DragTrackDensity.from_files(bad)


# ── Scenarios ─────────────────────────────────────────────────────────────

def test_scenario_selection(rho_realtime_dict, rho_forecast_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict, rho_forecast_dict)
    # baseline works.
    r0 = m.rho('2026-09-14T15:00:00Z', 0.0, 0.0, 400.0, source='baseline')
    assert _RHO_LO < r0 < _RHO_HI
    # every named scenario is accepted.
    for name in SCENARIOS:
        r = m.rho('2026-09-14T15:00:00Z', 0.0, 0.0, 400.0, source=name)
        assert np.isfinite(r) and _RHO_LO < r < _RHO_HI


def test_unknown_source_raises(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    with pytest.raises(ValueError, match='unknown source'):
        m.rho('2026-09-13T12:00:00Z', 0.0, 0.0, 400.0, source='oops')


# ── Forecast splice ──────────────────────────────────────────────────────

def test_forecast_splice(rho_realtime_dict, rho_forecast_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict, rho_forecast_dict)
    # Query inside the forecast window returns a finite rho.
    r = m.rho('2026-09-14T18:00:00Z', 0.0, 0.0, 400.0)
    assert np.isfinite(r) and _RHO_LO < r < _RHO_HI


# ── Real-JSON round-trip (optional) ──────────────────────────────────────

def test_real_json_roundtrip(real_rho_realtime, real_rho_forecast):
    m = DragTrackDensity.from_files(str(real_rho_realtime),
                                    rho_forecast=(str(real_rho_forecast)
                                                    if real_rho_forecast else None))
    r = m.rho('2026-09-20T12:00:00Z', 0.0, 0.0, 400.0)
    assert isinstance(r, float)
    assert _RHO_LO < r < _RHO_HI
    assert m.meta
