"""Wheel-boundary collision model (path_planning_wheel_boundary.py) on a synthetic map. No model, no GPU, no ROS.
  /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/navigation_handoff/tests
"""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import coordinate_transform as CT  # noqa: E402
import map_export as ME  # noqa: E402
import path_planning as PP  # noqa: E402
import path_planning_wheel_boundary as WB  # noqa: E402

VALUES = {"free": 254, "occupied": 0, "unknown": 205}


def synthetic():
    """Workspace 200 x 120 cm: yellow FREE band y 40..100, NOT_DRIVABLE elsewhere, a STATIC obstacle y 100..120 at x 120..200."""
    frame = CT.MapFrame(np.eye(3), [0, 120])
    grid = CT.MapGrid(frame, [-10, -10, 210, 130], 0.02)
    xf, yf = grid.cell_center_floor(*np.mgrid[0:grid.rows, 0:grid.cols])
    inside = (xf > 0) & (xf < 200) & (yf > 0) & (yf < 120)
    layers = np.where(inside, ME.NOT_DRIVABLE, ME.UNKNOWN).astype(np.uint8)
    layers[inside & (yf > 40) & (yf < 100)] = ME.FREE
    layers[(xf > 120) & (xf < 200) & (yf > 100) & (yf < 120)] = ME.STATIC_OBSTACLE
    pgm = np.where(layers == ME.FREE, 254, np.where(layers == ME.UNKNOWN, 205, 0)).astype(np.uint8)
    body_ok = ~np.isin(layers, [ME.STATIC_OBSTACLE, ME.NO_GO, ME.UNKNOWN])
    pm_w = PP.PgmMap(pgm, layers, VALUES, grid, frame, 4, 10)
    pm_b = PP.PgmMap(np.where(body_ok, 254, 0).astype(np.uint8), layers, VALUES, grid, frame, 4, 10)
    return WB.DualMap(pm_w, pm_b, (27.2, 24.7), (35, 24))


def test_wheels_inside_yellow_body_may_overhang_but_not_hit_obstacles():
    dm = synthetic()
    # heading -90 (facing towards +y_floor): wheels 27.2 cm along y, body 35 cm along y
    ok, _ = dm.body_free([60, 100 - 13.6 - 0.5], 35, 24, -90)          # wheels just inside the band, body overhangs NOT_DRIVABLE by ~4 cm
    assert ok
    ok, _ = dm.body_free([60, 100 - 13.6 + 1.0], 35, 24, -90)          # a wheel crosses the yellow edge
    assert not ok
    ok, _ = dm.body_free([160, 100 - 13.6 - 0.5], 35, 24, -90)         # same pose next to the obstacle: the body overhang hits it
    assert not ok
    m = WB.Metrics(dm).pose(np.array([60, 100 - 13.6 - 0.5]), -90)
    assert m["wheels_outside_yellow_px"] == 0 and m["body_on_blocking_layer_px"] == 0 and 3.0 < m["body_overhang_beyond_yellow_cm"] < 4.5


def test_dual_cspace_matches_dual_body_check():
    dm = synthetic()
    for yaw in (0, 90, 180, 270):
        valid, _ = dm.cspace(35 + 2 * 2.0, 24 + 2 * 2.0, yaw)
        for r, c in zip(*np.nonzero(dm.b.free)):
            assert valid[r, c] == dm.body_free(dm.centre_floor(r, c), 39, 28, yaw)[0], (yaw, r, c)


def test_urdf_wheel_envelope():
    u = WB.read_urdf_wheels(CT.load_config()["wheel_boundary_experiment"]["urdf"])
    assert abs(u["envelope_cm"][0] - 27.2091) < 1e-3 and abs(u["envelope_cm"][1] - 24.7001) < 1e-3
