"""dragtrack.plots: the figure builds from a product dict, the forecast splice, the CLI, and the PNG (when kaleido is
installed).  Skipped without the ``plots`` extra."""
from __future__ import annotations

import copy
from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("plotly")

from dragtrack.plots import atmosphere_figure, save_png, splice_forecast  # noqa: E402


@pytest.fixture()
def rt(rho_realtime_dict):
    d = copy.deepcopy(rho_realtime_dict)
    nf = len(d["fit"]["time_utc"])
    d["fit"]["uncertainty_fit"] = [[0.1] * len(d["altitude_km"])] * nf
    return d


@pytest.fixture()
def fc(rt):
    """A forecast starting 2026-09-14T06:00 (half way through the realtime fit), 6 rows at 3 h."""
    times = [f"2026-09-14T{h:02d}:00:00Z" for h in range(6, 24, 3)]
    rows = [[r * 1.1 for r in rt["fit"]["rho_fit"][0]] for _ in times]
    fit = {k: v for k, v in rt["fit"].items() if not isinstance(v, list)}
    fit.update(time_utc=times, rho_fit=rows, uncertainty_fit=[[0.2] * len(rt["altitude_km"])] * len(times),
               S=[0.0] * len(times), G=[12.0] * len(times), TJ70_K=[860.0] * len(times))
    return {"meta": dict(rt["meta"]), "altitude_km": rt["altitude_km"], "fit": fit,
            "spatial": {"fit": {"time_utc": times, "TJ70_lm": [[0.0] * 16 for _ in times]}}}


def test_figure_builds_with_spatial_rows(rt):
    fig = atmosphere_figure(rt, alt_km=400.0, now=datetime(2026, 9, 14, 3))
    types = [t.type for t in fig.data]
    assert "heatmap" in types                      # latitude / local-time panels
    rho = next(t for t in fig.data if t.hovertemplate and "solver" in t.hovertemplate)
    assert np.allclose(np.asarray(rho.y, float), 1.7e-12)


def test_altitude_is_interpolated(rt):
    fig = atmosphere_figure(rt, alt_km=425.0, now=datetime(2026, 9, 14, 3))
    rho = next(t for t in fig.data if t.hovertemplate and "solver" in t.hovertemplate)
    y = float(np.asarray(rho.y, float)[0])
    assert 6.7e-13 < y < 1.7e-12
    assert np.isclose(y, np.exp(0.5 * (np.log(1.7e-12) + np.log(6.7e-13))))


def test_splice_forecast(rt, fc):
    d = splice_forecast(rt, fc)
    t = d["fit"]["time_utc"]
    assert t[:2] == rt["fit"]["time_utc"][:2] and t[2:] == fc["fit"]["time_utc"]
    assert all(len(v) == len(t) for v in d["fit"].values() if isinstance(v, list))
    assert len(d["spatial"]["fit"]["TJ70_lm"]) == len(t)
    fig = atmosphere_figure(rt, fc, now=datetime(2026, 9, 14, 3))
    assert len(fig.data) > 0


def test_png_and_cli(rt, fc, tmp_path):
    pytest.importorskip("kaleido")
    import json

    from dragtrack.__main__ import main
    out = save_png(atmosphere_figure(rt, fc, now=datetime(2026, 9, 14, 3)), tmp_path / "a.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    (tmp_path / "rt.json").write_text(json.dumps(rt)); (tmp_path / "fc.json").write_text(json.dumps(fc))
    assert main(["plot", "--rho-realtime", str(tmp_path / "rt.json"), "--rho-forecast", str(tmp_path / "fc.json"),
                 "--alt", "400", "-o", str(tmp_path / "b.png")]) == 0
    assert (tmp_path / "b.png").stat().st_size > 1000


def test_real_file(real_rho_realtime):
    fig = atmosphere_figure(real_rho_realtime, alt_km=400.0)
    assert sum(t.type == "heatmap" for t in fig.data) >= 2
