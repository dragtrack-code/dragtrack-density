"""Command-line entry point for dragtrack.

Examples
--------
Evaluate rho at one point::

    python -m dragtrack rho 2026-09-13T12:00:00Z 45 0 450 \\
        --rho-realtime path/to/dragtrack_rho_realtime.json

Dump provenance of a runtime JSON::

    python -m dragtrack info --rho-realtime path/to/dragtrack_rho_realtime.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import numpy as np

from . import DragTrackDensity, __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m dragtrack',
        description='DragTrack atmospheric density command-line utility.',
    )
    p.add_argument('--version', action='version', version=f'dragtrack {__version__}')
    sub = p.add_subparsers(dest='cmd', required=True)

    p_rho = sub.add_parser('rho', help='Evaluate rho at (t, lat, lon, alt).')
    p_rho.add_argument('time', help='UTC ISO timestamp, e.g. 2026-09-13T12:00:00Z')
    p_rho.add_argument('lat', type=float, help='Geodetic latitude, deg')
    p_rho.add_argument('lon', type=float, help='Longitude, deg')
    p_rho.add_argument('alt', type=float, help='Altitude, km')
    p_rho.add_argument('--rho-realtime', required=True,
                       help='Path to dragtrack_rho_realtime.json')
    p_rho.add_argument('--clim', default=None,
                       help='Optional override climatology JSON path.')
    p_rho.add_argument('--j70', default=None,
                       help='Optional override J70 LUT JSON path.')
    p_rho.add_argument('--json', action='store_true',
                       help='Emit a JSON dict instead of a plain number.')

    p_info = sub.add_parser('info',
                            help='Show provenance / meta from a rho_realtime JSON.')
    p_info.add_argument('--rho-realtime', required=True,
                        help='Path to dragtrack_rho_realtime.json')

    return p


def _cmd_rho(args) -> int:
    m = DragTrackDensity.from_files(args.rho_realtime,
                                    clim=args.clim, j70_lut=args.j70)
    r = m.rho(args.time, args.lat, args.lon, args.alt)
    if args.json:
        out = {
            'time': args.time,
            'lat_deg': args.lat,
            'lon_deg': args.lon,
            'alt_km': args.alt,
            'rho_kg_m3': float(r),
            'schema_version': m.schema_version,
            'tle_edge_utc': m.tle_edge_utc,
        }
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write('\n')
    else:
        sys.stdout.write(f'{float(r):.6e}\n')
    return 0


def _cmd_info(args) -> int:
    m = DragTrackDensity.from_files(args.rho_realtime)
    json.dump(m.provenance(), sys.stdout, indent=2, default=str)
    sys.stdout.write('\n')
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == 'rho':
        return _cmd_rho(args)
    if args.cmd == 'info':
        return _cmd_info(args)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
