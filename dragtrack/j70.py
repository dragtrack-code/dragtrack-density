"""Jacchia 1970 vertical density lookup table.

The LUT is a precomputed 3D grid ``rho(z, Tinf, Tx)`` (gzip-base64 encoded
inside a JSON envelope; see :data:`dragtrack.data.j70_density_lookup_tx`).
``Tx`` is the temperature-inflection height parameter from Roberts (1971);
at query time we set ``Tx = calc_Tx(Tinf)`` so effectively rho depends on
``(z, Tinf)`` alone through the physically consistent Roberts composition
formula.

Usage
-----
>>> from dragtrack import J70
>>> j = J70()
>>> j(alt_km=450, TJ70_K=850)   # kg / m^3  (TJ70 == Jacchia's Tinf)
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

PathLike = Union[str, Path]


def calc_Tx(Tinf: ArrayLike) -> NDArray[np.float64]:
    """Roberts (1971) mixed-layer inflection temperature ``Tx`` from ``Tinf``.

    Parameters
    ----------
    Tinf : array_like
        Exospheric temperature in Kelvin.

    Returns
    -------
    ndarray
        ``Tx`` in Kelvin, same shape as ``Tinf``.
    """
    Tinf = np.asarray(Tinf, float)
    return 371.6678 + 0.0518806 * Tinf - 294.3505 * np.exp(-0.00216222 * Tinf)


def _decode(a: dict) -> np.ndarray:
    """Decode a ``{b64gz, shape, dtype}`` array envelope to ndarray."""
    raw = gzip.decompress(base64.b64decode(a['b64gz']))
    return np.frombuffer(raw, dtype=a['dtype']).reshape(a['shape']).copy()


class J70:
    """Callable Jacchia-70 lookup: ``rho_kg_m3 = J70()(alt_km, TJ70_K)``.

    ``TJ70_K`` is the exospheric temperature parameter (Jacchia's ``Tinf``,
    same numeric value — renamed for consistency with the dragtrack package).

    Parameters
    ----------
    path : str or Path, optional
        Path to the LUT JSON.  Defaults to the packaged asset
        ``dragtrack/data/j70_density_lookup_tx.json``.

    Attributes
    ----------
    z, T, Tx : ndarray
        Monotonic axis grids (``z`` in km, ``T`` and ``Tx`` in K).
    rho : ndarray, shape ``(len(z), len(T), len(Tx))``
        Density grid in kg/m^3.
    """

    def __init__(self, path: PathLike | None = None):
        if path is None:
            path = Path(__file__).parent / 'data' / 'j70_density_lookup_tx.json'
        self.path = Path(path)
        with open(self.path, 'r') as f:
            d = json.load(f)
        self.z = _decode(d['axes']['z_grid_km'])
        self.T = _decode(d['axes']['Tinf_grid_K'])
        self.Tx = _decode(d['axes']['Tx_grid'])
        self.rho = _decode(d['rho_grid_kg_m3'])
        self._log_rho = np.log10(np.clip(self.rho, 1e-30, None))
        # Guard: axes must be monotonic increasing (required by searchsorted).
        for name, ax in ('z', self.z), ('T', self.T), ('Tx', self.Tx):
            if not np.all(np.diff(ax) > 0):
                raise ValueError(f'J70 LUT axis {name!r} is not monotonic increasing')

    def __call__(self, alt_km: ArrayLike, TJ70_K: ArrayLike) -> NDArray[np.float64]:
        """Interpolate ``rho`` at ``(alt_km, TJ70_K)`` — inputs broadcast.

        ``TJ70_K`` is Jacchia's ``Tinf`` parameter (same numeric value),
        renamed to match the dragtrack package's naming.  Trilinear
        interpolation is performed on ``log10(rho)`` for numerical
        stability across ~10 orders of magnitude.
        """
        a = np.asarray(alt_km, float)
        T = np.asarray(TJ70_K, float)
        a, T = np.broadcast_arrays(a, T)
        a = np.clip(a, self.z[0], self.z[-1])
        T = np.clip(T, self.T[0], self.T[-1])
        Tx = np.clip(calc_Tx(T), self.Tx[0], self.Tx[-1])
        nZ, nT, nX = len(self.z), len(self.T), len(self.Tx)
        iz = np.clip(np.searchsorted(self.z, a, side='right') - 1, 0, nZ - 2)
        iT = np.clip(np.searchsorted(self.T, T, side='right') - 1, 0, nT - 2)
        iX = np.clip(np.searchsorted(self.Tx, Tx, side='right') - 1, 0, nX - 2)
        fz = (a - self.z[iz]) / (self.z[iz + 1] - self.z[iz])
        fT = (T - self.T[iT]) / (self.T[iT + 1] - self.T[iT])
        fX = (Tx - self.Tx[iX]) / (self.Tx[iX + 1] - self.Tx[iX])
        L = self._log_rho
        c000, c100 = L[iz, iT, iX], L[iz + 1, iT, iX]
        c010, c110 = L[iz, iT + 1, iX], L[iz + 1, iT + 1, iX]
        c001, c101 = L[iz, iT, iX + 1], L[iz + 1, iT, iX + 1]
        c011, c111 = L[iz, iT + 1, iX + 1], L[iz + 1, iT + 1, iX + 1]
        c00 = c000 * (1 - fz) + c100 * fz
        c10 = c010 * (1 - fz) + c110 * fz
        c01 = c001 * (1 - fz) + c101 * fz
        c11 = c011 * (1 - fz) + c111 * fz
        c0 = c00 * (1 - fT) + c10 * fT
        c1 = c01 * (1 - fT) + c11 * fT
        return 10.0 ** (c0 * (1 - fX) + c1 * fX)
