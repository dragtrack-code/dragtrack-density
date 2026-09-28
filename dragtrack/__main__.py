"""Command-line entry point for ``dragtrack``.

Examples
--------
Evaluate the density at one point::

    python -m dragtrack rho 2026-09-13T12:00:00Z 45 0 450 \\
        --rho-realtime path/to/dragtrack_rho_realtime.json

Include the 2014+ history and pick a scenario::

    python -m dragtrack rho 2026-09-28T00:00:00Z 0 0 400 \\
        --rho-realtime path/to/dragtrack_rho_realtime.json \\
        --rho-full path/to/dragtrack_rho_full.json \\
        --source extreme

Show what a JSON contains::

    python -m dragtrack info --rho-realtime path/to/dragtrack_rho_realtime.json

Paths may be local files or HTTP(S) URLs.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import DragTrackDensity, SCENARIOS, __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m dragtrack',
        description=('Compute atmospheric density from a DragTrack realtime '
                     'JSON (and optionally the 2014+ history JSON).'),
    )
    p.add_argument('--version', action='version',
                   version=f'dragtrack {__version__}')
    sub = p.add_subparsers(dest='cmd', required=True)

    p_rho = sub.add_parser(
        'rho',
        help='Compute rho at one (time, lat, lon, alt) point.',
        description=('Example:\n'
                     '  python -m dragtrack rho 2026-09-13T12:00:00Z 45 0 450 '
                     '--rho-realtime FILE'),
    )
    p_rho.add_argument('time',
                       help='UTC ISO-8601 timestamp, e.g. 2026-09-13T12:00:00Z')
    p_rho.add_argument('lat', type=float,
                       help='Geodetic latitude in degrees, -90 to 90')
    p_rho.add_argument('lon', type=float,
                       help='Longitude in degrees, -180 to 360')
    p_rho.add_argument('alt', type=float,
                       help='Geodetic altitude in km')
    p_rho.add_argument('--rho-realtime', required=True,
                       help=('Path to (or URL of) dragtrack_rho_realtime_*.json. '
                             'Required.'))
    p_rho.add_argument('--rho-full', default=None,
                       help=('Path to (or URL of) dragtrack_rho_full_*.json. '
                             'Optional; adds solver history from 2014 for times '
                             'before the realtime file\'s window.'))
    p_rho.add_argument('--source', default='baseline',
                       choices=('baseline',) + SCENARIOS,
                       help=('Which estimate to return.  "baseline" is the '
                             'best-current estimate; the others are scenarios '
                             'holding geomagnetic activity at fixed levels.  '
                             'Default: baseline.'))
    p_rho.add_argument('--json', action='store_true',
                       help='Print a full JSON record instead of a bare number.')

    p_info = sub.add_parser(
        'info',
        help='Print the timing fields and time coverage of a realtime JSON.',
    )
    p_info.add_argument('--rho-realtime', required=True,
                        help=('Path to (or URL of) dragtrack_rho_realtime_*.json.'))
    p_info.add_argument('--json', action='store_true',
                        help='Print the raw meta block as JSON.')

    return p


def _cmd_rho(args) -> int:
    m = DragTrackDensity.from_files(args.rho_realtime, args.rho_full)
    r = m.rho(args.time, args.lat, args.lon, args.alt, source=args.source)
    if args.json:
        out = {
            'time': args.time, 'lat_deg': args.lat, 'lon_deg': args.lon,
            'alt_km': args.alt, 'source': args.source,
            'rho_kg_m3': float(r), 'meta': m.meta,
        }
        json.dump(out, sys.stdout, indent=2, default=str)
        sys.stdout.write('\n')
    else:
        sys.stdout.write(f'{float(r):.6e}\n')
    return 0


def _cmd_info(args) -> int:
    m = DragTrackDensity.from_files(args.rho_realtime)
    if args.json:
        json.dump(m.meta, sys.stdout, indent=2, default=str)
        sys.stdout.write('\n')
        return 0
    # Human-readable summary
    meta = m.meta
    fields = ('schema_version', 'generated_utc', 'tle_edge_utc',
              'ref_time_utc', 'product_latency_hours', 'window_days')
    for k in fields:
        if k in meta:
            print(f'{k:26s}  {meta[k]}')
    print(f'{"altitude_km":26s}  {m.altitude_km.tolist()}')
    cov = m.coverage
    for k in ('solver_start', 'solver_edge', 'end'):
        print(f'{k:26s}  {cov[k]}')
    for src, end in cov['end_by_source'].items():
        print(f'{"end[" + src + "]":26s}  {end}')
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
