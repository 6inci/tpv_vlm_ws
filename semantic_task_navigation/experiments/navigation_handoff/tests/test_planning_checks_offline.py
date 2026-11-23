"""Unified planning checks (planning_checks.py) on a synthetic map. No model, no GPU, no ROS.
  /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/navigation_handoff/tests
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import coordinate_transform as CT  # noqa: E402
import map_export as ME  # noqa: E402
import path_planning as PP  # noqa: E402
import path_planning_batch_step4b as BB  # noqa: E402
import path_planning_wheel_boundary as WB  # noqa: E402
import planning_checks as PC  # noqa: E402

VALUES = {"free": 254, "occupied": 0, "unknown": 205}
UCFG = yaml.safe_load((CT.CONFIG_PATH.parent / "unified_planning.yaml").read_text(encoding="utf-8"))
PCFG = CT.load_config()["planning"]
WHEELS, BODY = (27.2, 24.7), (35, 24)
NO_ZONES = type("NoZones", (), {"zone_at": staticmethod(lambda p: None)})()     # stand-in for survey_photos_step4a.Ctx (zone names only)


def synthetic():
    """Same map as test_wheel_boundary_offline: 200 x 120 cm, yellow FREE band y 40..100, NOT_DRIVABLE elsewhere, STATIC y 100..120 at x 120..200.
    Identity floor transform, so 'image px' = floor cm."""
    frame = CT.MapFrame(np.eye(3), [0, 120])
    grid = CT.MapGrid(frame, [-10, -10, 210, 130], 0.02)
    xf, yf = grid.cell_center_floor(*np.mgrid[0:grid.rows, 0:grid.cols])
    inside = (xf > 0) & (xf < 200) & (yf > 0) & (yf < 120)
    layers = np.where(inside, ME.NOT_DRIVABLE, ME.UNKNOWN).astype(np.uint8)
    layers[inside & (yf > 40) & (yf < 100)] = ME.FREE
    layers[(xf > 120) & (xf < 200) & (yf > 100) & (yf < 120)] = ME.STATIC_OBSTACLE
    pgm = np.where(layers == ME.FREE, 254, np.where(layers == ME.UNKNOWN, 205, 0)).astype(np.uint8)
    return pgm, layers, grid, frame


def world_of(boxes=()):
    pgm, layers, grid, frame = synthetic()
    obst = PC.obstacle_mask(grid, frame, list(boxes), PCFG["raster_px_per_cm"])
    return PC.build_world(pgm, layers, VALUES, grid, frame, PCFG, UCFG["body_blocking_codes"], WHEELS, BODY, obst), grid, frame


def test_shipped_policy_passes_and_relaxations_are_refused():
    assert PC.check_policy(UCFG)
    for path, value in ((("gripper_baseline", "blocking_layers"), ["OTHER_PER_PHOTO_OBJECT"]), (("gripper_baseline", "max_reach_cm"), 60),
                        (("body_blocking_codes",), ["STATIC_OBSTACLE", "UNKNOWN"]), (("margin_cm",), -1.0)):
        u = copy.deepcopy(UCFG)
        d = u
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
        with pytest.raises(ValueError):
            PC.check_policy(u)


def test_empty_obstacles_reproduce_wheel_model_and_a_pallet_on_free_blocks_the_wheels():
    pgm, layers, grid, frame = synthetic()
    body_ok = ~np.isin(layers, [ME.STATIC_OBSTACLE, ME.NO_GO, ME.UNKNOWN])
    ref = WB.DualMap(PP.PgmMap(pgm, layers, VALUES, grid, frame, 4, 10), PP.PgmMap(np.where(body_ok, 254, 0).astype(np.uint8), layers, VALUES, grid, frame, 4, 10),
                     WHEELS, BODY)                                     # = path_planning_wheel_boundary.main
    w0, _, _ = world_of()
    for yaw in (0, 90):
        assert np.array_equal(w0.cspace(39, 28, yaw)[0], ref.cspace(39, 28, yaw)[0])
    assert w0.body_free([60, 70], 39, 28, 0)[0]
    w1, _, _ = world_of([[70, 60, 80, 80]])                           # a pallet inside the yellow band, next to the body
    assert not w1.body_free([60, 70], 39, 28, 0)[0] and w1.body_free([30, 70], 39, 28, 0)[0]


def test_gripper_baseline_blocking_layers_and_2d_only():
    boxes = [[55, 20, 65, 30], [66, 33, 74, 41]]                       # own pallet; another pallet on the slanted line (80, 52.5) -> (60, 25)
    w, grid, frame = world_of(boxes)
    own, other, i = PC.target_masks(grid, frame, boxes, [60, 25], 4)
    assert i == 0
    g = UCFG["gripper_baseline"]
    ok = PC.gripper_baseline_2d(w, [60, 60], 90, [60, 25], own, other, BODY, g)          # facing -y floor: NOT_DRIVABLE, then the own pallet
    assert not ok["blocked"] and {r["layer"] for r in ok["runs"]} <= {"FREE", "NOT_DRIVABLE", "TARGET_PALLET"} and ok["runs"][-1]["layer"] == "TARGET_PALLET"
    assert ok["max_reach_cm"] is None and "UNVERIFIED" in ok["gripper_3d"]
    hit = PC.gripper_baseline_2d(w, [80, 70], 90, [60, 25], own, other, BODY, g)         # the slanted line crosses the other pallet
    assert hit["blocked"] and hit["blocked_by"] == ["OTHER_PER_PHOTO_OBJECT"]
    st = PC.gripper_baseline_2d(w, [160, 70], -90, [160, 112], own, other, BODY, g)      # facing +y floor into the static zone
    assert st["blocked_by"] == ["STATIC_OBSTACLE"]


def test_baseline_decision_matches_the_batch_rule():
    boxes = [[55, 20, 65, 30], [75, 25, 85, 35], [140, 95, 150, 105]]
    w, grid, frame = world_of(boxes)
    rng = np.random.default_rng(0)
    for _ in range(200):
        tgt_box = boxes[rng.integers(len(boxes))]
        tgt = [(tgt_box[0] + tgt_box[2]) / 2, (tgt_box[1] + tgt_box[3]) / 2]
        own, other, _ = PC.target_masks(grid, frame, boxes, tgt, 4)
        cand = {"centre_floor_cm": [float(rng.uniform(20, 180)), float(rng.uniform(45, 95))], "yaw_map_deg": float(rng.choice([0, 90, 180, 270]))}
        new = PC.gripper_baseline_2d(w, cand["centre_floor_cm"], cand["yaw_map_deg"], tgt, own, other, BODY, UCFG["gripper_baseline"])
        old = BB.gripper_line_hits(w, cand, np.array(tgt), other, BODY)                      # batch decision
        _, runs = BB.gripper_line_runs(w, NO_ZONES, cand, np.array(tgt), other, own, BODY)    # batch diagnostic figure
        assert new["blocked"] == bool(old), (cand, tgt, new["blocked_by"], old)
        assert [(r["layer"], r["blocking"], r["length_cm"]) for r in new["runs"]] == [(r["layer"], r["blocking"], r["length_cm"]) for r in runs]


def test_target_outside_every_box_makes_all_pallets_other():
    boxes = [[55, 20, 65, 30], [75, 25, 85, 35]]
    _, grid, frame = world_of(boxes)
    own, other, i = PC.target_masks(grid, frame, boxes, [100, 25], 4)
    assert i is None and not own.any() and other.sum() == PC.obstacle_mask(grid, frame, boxes, 4).sum()


def test_path_b_is_diagnostic_when_path_a_is_not_valid():
    w, grid, frame = world_of()
    own, other, _ = PC.target_masks(grid, frame, [], None, 4)
    gripper = {"offset_x_m": None, "offset_y_m": None, "offset_y_m_assumed": 0.0}
    res = PC.plan_task(w, np.array([40.0, 70.0]), 0.0, np.array([160.0, 112.0]), np.array([60.0, 30.0]),
                       {"A": (own, other), "B": (own, other)}, BODY, 2.0, PCFG, gripper, UCFG["gripper_baseline"], 1.0)
    assert res["A"]["status"] == "NOT_FOUND" and "GRIPPER_LINE_BLOCKED" in [c["status"] for c in res["A"]["candidates"]]
    assert res["B"]["status"] == "CONDITIONAL" and res["B"]["role"] == "DIAGNOSTIC_ONLY" and "robot start" in res["B"]["basis"]
    assert res["B"]["desc"]["checks"]["orthogonal_4_connected"] and res["B"]["desc"]["collision_checks_passed"]
