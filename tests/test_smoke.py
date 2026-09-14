"""Smoke tests for the dragtrack package.

Covered:
- module import and version
- Y_modes basis shape and finiteness
- J70 lookup in normal + clip regimes
- DragTrackDensity built from synthetic rho_realtime dict
- Scalar vs vector rho() shapes
- Optional: real rho_realtime file (skipped if unavailable)
- Optional: TLE propagation (skipped if sgp4 missing)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import dragtrack
from dragtrack import DragTrackDensity, J70, Y_modes, REG_NAMES, calc_Tx

# Reasonable envelope for LEO thermospheric densities (kg/m^3).
_RHO_LO = 1e-16
_RHO_HI = 1e-6


def test_module_metadata():
    assert dragtrack.__version__
    assert isinstance(REG_NAMES, tuple)
    assert len(REG_NAMES) == 15


def test_basis_shape_and_finite():
    lat = np.linspace(-80, 80, 17)
    lst = np.linspace(0, 24, 17, endpoint=False)
    Y = Y_modes(lat, lst)
    assert Y.shape == (17, 15)
    assert np.all(np.isfinite(Y))
    # Y10 should equal sin(lat).
    np.testing.assert_allclose(Y[:, 0], np.sin(np.deg2rad(lat)))


def test_basis_scalar_input():
    Y = Y_modes(45.0, 12.0)
    assert Y.shape == (1, 15)


def test_j70_load_and_call():
    j = J70()
    r = j(400.0, 850.0)
    # J70 LUT stores a shape function (kg/m^3 units but not absolutely
    # calibrated — the pipeline scales it by J70_scale).  We only check
    # finiteness + positivity here.
    assert np.isscalar(r) or r.shape == ()
    assert float(r) > 0 and np.isfinite(float(r))


def test_j70_vectorized():
    j = J70()
    alt = np.array([300., 400., 500., 600.])
    T = np.full_like(alt, 900.)
    r = j(alt, T)
    assert r.shape == alt.shape
    # rho decreases monotonically with altitude at fixed Tinf.
    assert np.all(np.diff(r) < 0)
    assert np.all(np.isfinite(r) & (r > 0))
    # rho at 400 km should increase with Tinf.
    T_lo = j(np.full(4, 400.0), np.array([600., 900., 1200., 1500.]))
    assert np.all(np.diff(T_lo) > 0)


def test_j70_clips_out_of_range_alt():
    j = J70()
    # 50 km is below the LUT floor; should clip, not raise.
    r_lo = j(50.0, 850.0)
    r_at = j(j.z[0], 850.0)
    np.testing.assert_allclose(r_lo, r_at)


def test_calc_Tx_shape_and_range():
    T = np.array([500., 900., 1500., 2500.])
    Tx = calc_Tx(T)
    assert Tx.shape == T.shape
    assert np.all(Tx > 200) and np.all(Tx < 700)


def test_density_from_synthetic_dict(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    # Scalar path.
    r = m.rho('2026-09-13T12:00:00Z', 45.0, 0.0, 450.0)
    assert isinstance(r, float)
    assert _RHO_LO < r < _RHO_HI
    # Vector-lat path.
    lat = np.linspace(-80, 80, 9)
    r2 = m.rho('2026-09-13T12:00:00Z', lat, np.zeros(9), np.full(9, 450.0))
    assert r2.shape == (9,)
    assert np.all(np.isfinite(r2))
    # Vector-alt path.
    alt = np.array([300., 400., 500.])
    r3 = m.rho('2026-09-13T12:00:00Z',
               np.full(3, 45.0), np.zeros(3), alt)
    assert r3.shape == (3,)
    assert r3[0] > r3[1] > r3[2]     # rho decreases with alt


def test_density_meta_and_provenance(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    assert m.schema_version == '0.8.0-test'
    assert m.tle_edge_utc == '2026-09-13T00:00:00Z'
    prov = m.provenance()
    assert prov['n_solver_bins'] == len(rho_realtime_dict['solver']['time_utc'])
    assert 'recipe' in prov and 'rho_400' in prov['recipe']


def test_density_multi_time(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    tps = ['2026-09-13T00:00:00Z', '2026-09-13T12:00:00Z',
           '2026-09-13T21:00:00Z']
    r = m.rho(tps, 45.0, 0.0, 450.0)
    assert r.shape == (3,)
    assert np.all(np.isfinite(r))


def test_missing_T_J70_keys_raises(rho_realtime_dict):
    bad = json.loads(json.dumps(rho_realtime_dict))  # deep copy
    del bad['solver']['T_J70_m']
    with pytest.raises(KeyError, match='T_J70_m'):
        DragTrackDensity.from_files(bad)


def test_altitude_400_required(rho_realtime_dict):
    bad = json.loads(json.dumps(rho_realtime_dict))
    bad['altitude_km'] = [250.0, 300.0]
    bad['solver']['rho'] = [row[:2] for row in bad['solver']['rho']]
    with pytest.raises(ValueError, match='400'):
        DragTrackDensity.from_files(bad)


# ── Real-JSON round-trip (optional) ─────────────────────────────────

def test_real_json_roundtrip(real_rho_realtime):
    m = DragTrackDensity.from_files(str(real_rho_realtime))
    r = m.rho('2026-09-13T12:00:00Z', 45.0, 0.0, 450.0)
    assert isinstance(r, float)
    assert _RHO_LO < r < _RHO_HI
    # Provenance must expose schema version + TLE edge.
    assert m.schema_version
    assert m.tle_edge_utc


# ── Optional SGP4 test ──────────────────────────────────────────────

def _sgp4_available():
    try:
        import sgp4  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _sgp4_available(),
                    reason='sgp4 not installed')
def test_tle_propagation_returns_sensible_altitude(rho_realtime_dict):
    m = DragTrackDensity.from_files(rho_realtime_dict)
    # A generic ISS-like TLE — bracket around 400-450 km orbit.
    l1 = '1 25544U 98067A   26256.50000000  .00000000  00000-0  00000-0 0  9997'
    l2 = '2 25544  51.6400   0.0000 0001000   0.0000   0.0000 15.50000000000019'
    tps = ['2026-09-13T00:00:00Z', '2026-09-13T06:00:00Z',
           '2026-09-13T12:00:00Z']
    t, lat, lon, alt, rho = m.rho_along_tle(l1, l2, tps)
    assert np.all(np.isfinite(alt))
    assert np.all((alt > 300) & (alt < 700))
    assert np.all((lat > -90) & (lat < 90))
    assert np.all(np.isfinite(rho))
    assert np.all((rho > _RHO_LO) & (rho < _RHO_HI))
