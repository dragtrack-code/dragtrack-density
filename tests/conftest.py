"""pytest fixtures for dragtrack tests.

Fixtures build small synthetic ``dragtrack_rho_realtime.json`` /
``dragtrack_rho_forecast.json`` dicts that match the v0.8+ SH-16 schema,
so the test-suite runs without any external file dependencies.  If the
env var ``DRAGTRACK_RHO_REALTIME`` points at a real file, the
``real_rho_realtime`` fixture opts in.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pytest


def _blank_lm(n: int) -> list[list[float]]:
    """SH-16 all-zero coefficients (pure global T_c, no spatial variation)."""
    return [[0.0] * 16 for _ in range(n)]


@pytest.fixture(scope='session')
def rho_realtime_dict() -> dict:
    """Small synthetic rho_realtime dict spanning ~1 day at 3-hr cadence."""
    times = [f'2026-09-13T{h:02d}:00:00Z' for h in range(0, 24, 3)]
    n = len(times)
    alts = [250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0]
    rho_row = [4.3e-11, 1.3e-11, 4.5e-12, 1.7e-12,
               6.7e-13, 2.8e-13, 1.2e-13, 5.6e-14]
    rho = [rho_row for _ in range(n)]
    # Fit block: same rho as solver so tests get a monotone timeseries.
    fit_times = [f'2026-09-14T{h:02d}:00:00Z' for h in range(0, 12, 3)]
    nf = len(fit_times)
    fit_rho = [rho_row for _ in range(nf)]
    return {
        'meta': {
            'schema_version': '0.8.0-test',
            'generated_utc': '2026-09-14T00:00:00Z',
            'tle_edge_utc': '2026-09-13T00:00:00Z',
            'product_latency_hours': 12,
            'window_days': 27,
        },
        'altitude_km': alts,
        'solver': {
            'time_utc': times,
            'source': 'test',
            'rho': rho,
            'uncertainty': [[0.1] * len(alts)] * n,
            'J70_Tinf_K': [850.0] * n,
            'J70_scale': [1.0] * n,
            'bin_days': [0.125] * n,
        },
        'fit': {
            'time_utc': fit_times,
            'row_type': ['forecast'] * nf,
            'rho_fit': fit_rho,
            'J70_Tinf_K': [850.0] * nf,
            'S': [0.0] * nf,
            'G': [10.0] * nf,
            'params': {'a': 1e-9, 'b': 1e-14, 'c': -1e-12, 't0': 0.5, 'rmspe': 0.10},
        },
        'spatial': {
            'solver': {
                'time_utc': times,
                'Tinf_lm':  _blank_lm(n),
            },
            'fit': {
                'time_utc': fit_times,
                'Tinf_lm':  _blank_lm(nf),
            },
            'provenance': {
                'recipe':       'j70_clim_R0.25_A_res_Y11',
                'convention':   'Tinf(lat,lst,t) = J70_Tinf_K[t] · (1 + Σ_k Tinf_lm[t,k]·Y_k(lat,lst))',
                'mode_order':   ['Y00','Y10','Y11c','Y11s','Y20','Y21c','Y21s','Y22c','Y22s',
                                 'Y30','Y31c','Y31s','Y32c','Y32s','Y33c','Y33s'],
            },
        },
    }


@pytest.fixture(scope='session')
def rho_forecast_dict() -> dict:
    """Small synthetic rho_forecast dict — a fit-only block extending realtime."""
    times = [f'2026-09-14T{h:02d}:00:00Z' for h in range(12, 24, 3)]
    n = len(times)
    alts = [250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0]
    rho_row = [4.3e-11, 1.3e-11, 4.5e-12, 1.7e-12,
               6.7e-13, 2.8e-13, 1.2e-13, 5.6e-14]
    rho = [rho_row for _ in range(n)]
    return {
        'meta': {'schema_version': '0.8.0-test',
                 'tle_edge_utc': '2026-09-13T00:00:00Z'},
        'altitude_km': alts,
        'fit': {
            'time_utc': times,
            'row_type': ['forecast'] * n,
            'rho_fit':  rho,
            'uncertainty_fit': [[0.15] * len(alts)] * n,
            'J70_Tinf_K': [860.0] * n,
            'S': [0.0] * n,
            'G': [15.0] * n,
            'params': {'a': 1e-9, 'b': 1e-14, 'c': -1e-12, 't0': 0.5, 'rmspe': 0.10},
        },
        'spatial': {
            'fit': {
                'time_utc': times,
                'Tinf_lm':  _blank_lm(n),
            },
            'provenance': {'recipe': 'j70_clim_R0.25_A_res_Y11'},
        },
    }


@pytest.fixture(scope='session')
def real_rho_realtime() -> Path:
    """Newest real rho_realtime JSON, or skip.

    Order of precedence: ``DRAGTRACK_RHO_REALTIME`` env var,
    then the well-known dev-machine reports dir.
    """
    env = os.environ.get('DRAGTRACK_RHO_REALTIME')
    if env and Path(env).exists():
        return Path(env)
    reports = Path('C:/Users/dange/Documents/dragtrack_ops_data/reports/rho_realtime')
    if reports.exists():
        fps = sorted(reports.glob('dragtrack_rho_realtime_*.json'))
        if fps:
            return fps[-1]
    pytest.skip('No real rho_realtime JSON available '
                '(set DRAGTRACK_RHO_REALTIME to enable)')


@pytest.fixture(scope='session')
def real_rho_forecast() -> Path:
    """Newest real rho_forecast JSON, or ``None`` if unavailable."""
    env = os.environ.get('DRAGTRACK_RHO_FORECAST')
    if env and Path(env).exists():
        return Path(env)
    reports = Path('C:/Users/dange/Documents/dragtrack_ops_data/reports/rho_forecast')
    if reports.exists():
        fps = sorted(reports.glob('dragtrack_rho_forecast_*.json'))
        if fps:
            return fps[-1]
    return None
