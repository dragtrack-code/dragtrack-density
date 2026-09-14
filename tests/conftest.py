"""pytest fixtures for dragtrack tests.

The `rho_realtime_dict` fixture builds a small synthetic
`dragtrack_rho_realtime.json`-shaped dict so the test-suite runs without
any external file dependencies.  If the environment variable
``DRAGTRACK_RHO_REALTIME`` points at a real file, tests that opt in via
the `real_rho_realtime` fixture pick it up; otherwise they skip.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='session')
def rho_realtime_dict() -> dict:
    """Small synthetic rho_realtime dict spanning ~1 day at 3-hr cadence."""
    times = [f'2026-09-13T{h:02d}:00:00Z' for h in range(0, 24, 3)]
    n = len(times)
    alts = [250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0]
    # Fake rho column that decreases roughly with altitude
    rho_row = [4.3e-11, 1.3e-11, 4.5e-12, 1.7e-12, 6.7e-13, 2.8e-13, 1.2e-13, 5.6e-14]
    rho = [rho_row for _ in range(n)]
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
            'T_J70_m': [1.04e14] * n,
            'T_J70_b': [759.0] * n,
            'bin_days': [0.125] * n,
        },
    }


@pytest.fixture(scope='session')
def real_rho_realtime() -> Path:
    """Path to a real rho_realtime JSON, or skip.

    Set ``DRAGTRACK_RHO_REALTIME`` env var to opt in.  A well-known dev path
    is also probed automatically.
    """
    env = os.environ.get('DRAGTRACK_RHO_REALTIME')
    if env and Path(env).exists():
        return Path(env)
    dev_candidates = [
        Path('C:/Users/dange/Documents/dragtrack_ops_data/reports/rho_realtime/'
             'dragtrack_rho_realtime_v0.8.0_20260914T021229Z-657180.json'),
    ]
    for p in dev_candidates:
        if p.exists():
            return p
    pytest.skip('No real rho_realtime JSON available '
                '(set DRAGTRACK_RHO_REALTIME to enable)')
