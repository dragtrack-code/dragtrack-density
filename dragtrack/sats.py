"""Per-satellite daily drag data from the ``leo_sats`` / ``stable_sats`` products."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from .model import PathLike, _load_json


def load_sats(src: PathLike | Mapping[str, Any]) -> Dict[int, dict]:
    """Load a ``leo_sats`` or ``stable_sats`` JSON (path, URL, ``.gz`` or dict).

    Returns ``{norad_id: record}``.  Each record has the product's
    per-satellite fields under their product names: scalars (``name``,
    ``inc_deg``, ``raan_deg``, ``rmspe``) as-is, daily lists as float arrays
    with null -> NaN, and ``date``, the UTC day of each daily value
    (``datetime64[D]``).  Units and definitions are in the file's ``meta``.
    """
    d = src if isinstance(src, Mapping) else _load_json(src)
    if 'sats' not in d or 'time_utc_daily' not in d:
        raise KeyError("not a DragTrack leo_sats / stable_sats JSON "
                       "(needs 'sats' and 'time_utc_daily')")
    date = np.array(d['time_utc_daily'], dtype='datetime64[D]')
    out: Dict[int, dict] = {}
    for s in d['sats']:
        rec: dict = {'date': date}
        for k, v in s.items():
            if isinstance(v, list):
                if len(v) != date.size:
                    raise ValueError(f"NORAD {s.get('norad_id')}: {k!r} has "
                                     f"{len(v)} values for {date.size} days")
                rec[k] = np.array([np.nan if x is None else float(x) for x in v], float)
            else:
                rec[k] = v
        out[int(s['norad_id'])] = rec
    return out
