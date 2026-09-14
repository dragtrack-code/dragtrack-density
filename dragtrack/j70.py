"""J70 vertical LUT loader (from gzip-base64 JSON).

Provides `rho_j70(alt_km, Tinf_K)` — trilinear interp on (z, Tinf, Tx)
with Tx = calc_Tx(Tinf) (Roberts 1970 composition formula).
"""
from __future__ import annotations
import json, gzip, base64
from pathlib import Path
import numpy as np


def calc_Tx(Tinf):
    """Roberts 1970 mixed-layer temp Tx from exospheric temp Tinf."""
    Tinf = np.asarray(Tinf, float)
    return 371.6678 + 0.0518806 * Tinf - 294.3505 * np.exp(-0.00216222 * Tinf)


def _decode(a: dict) -> np.ndarray:
    raw = gzip.decompress(base64.b64decode(a['b64gz']))
    return np.frombuffer(raw, dtype=a['dtype']).reshape(a['shape']).copy()


class J70:
    """Callable J70 lookup. `rho = J70()(alt_km, Tinf_K)`."""
    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = Path(__file__).parent / 'data' / 'j70_density_lookup_tx.json'
        with open(path, 'r') as f:
            d = json.load(f)
        self.z    = _decode(d['axes']['z_grid_km'])
        self.T    = _decode(d['axes']['Tinf_grid_K'])
        self.Tx   = _decode(d['axes']['Tx_grid'])
        self.rho  = _decode(d['rho_grid_kg_m3'])
        self._log_rho = np.log10(np.clip(self.rho, 1e-30, None))

    def __call__(self, alt_km, Tinf_K):
        a  = np.clip(np.asarray(alt_km, float), self.z[0],  self.z[-1])
        T  = np.clip(np.asarray(Tinf_K, float), self.T[0],  self.T[-1])
        Tx = np.clip(calc_Tx(T),                 self.Tx[0], self.Tx[-1])
        # Trilinear
        nZ, nT, nX = len(self.z), len(self.T), len(self.Tx)
        iz = np.clip(np.searchsorted(self.z,  a,  side='right') - 1, 0, nZ - 2)
        iT = np.clip(np.searchsorted(self.T,  T,  side='right') - 1, 0, nT - 2)
        iX = np.clip(np.searchsorted(self.Tx, Tx, side='right') - 1, 0, nX - 2)
        fz = (a  - self.z[iz])  / (self.z[iz + 1]  - self.z[iz])
        fT = (T  - self.T[iT])  / (self.T[iT + 1]  - self.T[iT])
        fX = (Tx - self.Tx[iX]) / (self.Tx[iX + 1] - self.Tx[iX])
        L = self._log_rho
        c000, c100 = L[iz,     iT,     iX],     L[iz + 1, iT,     iX]
        c010, c110 = L[iz,     iT + 1, iX],     L[iz + 1, iT + 1, iX]
        c001, c101 = L[iz,     iT,     iX + 1], L[iz + 1, iT,     iX + 1]
        c011, c111 = L[iz,     iT + 1, iX + 1], L[iz + 1, iT + 1, iX + 1]
        c00 = c000*(1-fz) + c100*fz
        c10 = c010*(1-fz) + c110*fz
        c01 = c001*(1-fz) + c101*fz
        c11 = c011*(1-fz) + c111*fz
        c0  = c00*(1-fT)  + c10*fT
        c1  = c01*(1-fT)  + c11*fT
        return 10.0**(c0*(1-fX) + c1*fX)
