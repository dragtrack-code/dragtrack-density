"""pytest fixtures for dragtrack tests.

Fixtures build small synthetic ``dragtrack_rho_realtime.json`` /
dicts that match the v0.9+ SH-16 schema
(``TJ70_K`` / ``TJ70_lm`` — renamed from ``J70_Tinf_K`` / ``Tinf_lm``),
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
            'schema_version': '0.9.0-test',
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
            'TJ70_K': [850.0] * n,
            'J70_scale': [1.0] * n,
            'bin_days': [0.125] * n,
        },
        'fit': {
            'time_utc': fit_times,
            'row_type': ['forecast'] * nf,
            'rho_fit': fit_rho,
            'TJ70_K': [850.0] * nf,
            'S': [0.0] * nf,
            'G': [10.0] * nf,
            'params': {'a': 1e-9, 'b': 1e-14, 'c': -1e-12, 't0': 0.5, 'rmspe': 0.10},
        },
        'spatial': {
            'solver': {
                'time_utc': times,
                'TJ70_lm':  _blank_lm(n),
            },
            'fit': {
                'time_utc': fit_times,
                'TJ70_lm':  _blank_lm(nf),
            },
            'provenance': {
                'recipe':       'j70_clim_R0.25_A_res_Y11',
                'convention':   'TJ70(lat,lst,t) = TJ70_K[t] · (1 + Σ_k TJ70_lm[t,k]·Y_k(lat,lst))',
                'mode_order':   ['Y00','Y10','Y11c','Y11s','Y20','Y21c','Y21s','Y22c','Y22s',
                                 'Y30','Y31c','Y31s','Y32c','Y32s','Y33c','Y33s'],
            },
        },
    }



@pytest.fixture(scope='session')
def real_rho_realtime() -> Path:
    """Newest real rho_realtime JSON matching the current schema (v0.9+), or skip.

    Order of precedence: ``DRAGTRACK_RHO_REALTIME`` env var,
    then the well-known dev-machine reports dir.  Pre-v0.9 snapshots
    (``J70_Tinf_K`` / ``Tinf_lm`` schema) are hard-break incompatible and
    are filtered out here — the reader must fail loudly on them if the
    caller supplies one directly.
    """
    env = os.environ.get('DRAGTRACK_RHO_REALTIME')
    if env and Path(env).exists():
        return Path(env)
    reports = Path('C:/Users/dange/Documents/dragtrack_ops_data/reports/rho_realtime')
    if reports.exists():
        fps = sorted(reports.glob('dragtrack_rho_realtime_v0.9*.json'))
        if fps:
            return fps[-1]
    pytest.skip('No real rho_realtime JSON (v0.9+) available '
                '(set DRAGTRACK_RHO_REALTIME to enable)')
