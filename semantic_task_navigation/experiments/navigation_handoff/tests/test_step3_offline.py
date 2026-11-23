"""STEP 3 offline checks on synthetic PGM maps (no model, no GPU, no ROS).
  /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/navigation_handoff/tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import coordinate_transform as CT  # noqa: E402
import map_export as ME  # noqa: E402
import object_positions as OP  # noqa: E402
import path_planning as PP  # noqa: E402
from semantic_task_nav.navigation import planner as P  # noqa: E402

VALUES = {"free": 254, "occupied": 0, "unknown": 205}
PCFG = {"turn_penalty_cm": 20, "planner": "SAFETY_COST_ASTAR", "safety_weight": 1.0, "start_escape_max_cm": 10, "rotation_step_deg": 2,
        "sweep_step_cm": 0.5}


def make_map(free_rect_cm, W=200, H=120, obstacles=()):
    """Workspace W x H cm, free rectangle (x0, y0, x1, y1) in floor cm, optional occupied rectangles inside it."""
    frame = CT.MapFrame(np.eye(3), [0, H])
    grid = CT.MapGrid(frame, [-10, -10, W + 10, H + 10], 0.02)
    xf, yf = grid.cell_center_floor(*np.mgrid[0:grid.rows, 0:grid.cols])
    rects = free_rect_cm if isinstance(free_rect_cm, list) else [free_rect_cm]
    free = np.zeros(xf.shape, bool)
    for x0, y0, x1, y1 in rects:
        free |= (xf > x0) & (xf < x1) & (yf > y0) & (yf < y1)
    for a, b, c, d in obstacles:
        free &= ~((xf > a) & (xf < c) & (yf > b) & (yf < d))
    pgm = np.where(free, 254, 0).astype(np.uint8)
    layers = np.where(free, ME.FREE, ME.NOT_DRIVABLE).astype(np.uint8)
    return PP.PgmMap(pgm, layers, VALUES, grid, frame, 4, 10)


def test_turn_penalty_removes_staircase_that_v02_4neighbours_makes():
    pm = make_map((0, 0, 200, 120))
    view = pm.view(10, 10, 0)
    s, g = (50, 15), (12, 90)                                             # floor (21, 91) -> (171, 15) cm, both well inside
    assert view.valid[s] and view.valid[g]
    with PP.four_neighbours():
        base = P.astar(view, s, g, "SAFETY_COST_ASTAR", 1.0)
    new = PP.astar_orthogonal(view, s, g, "SAFETY_COST_ASTAR", 1.0, 20 * pm.s)
    assert base["status"] == new["status"] == "PATH_FOUND"
    assert P.NEIGHBORS != PP.NEIGHBORS4                                   # the v0.2 table is restored
    assert PP.is_orthogonal(new["cells"]) and PP.is_orthogonal(base["cells"])
    assert len(PP.runs_of(new["cells"])) - 1 <= 2 < len(PP.runs_of(base["cells"])) - 1
    assert len(new["cells"]) == len(base["cells"])                        # same Manhattan length, fewer turns


def test_turn_indices_keep_all_raw_cells():
    cells = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 3]]
    assert PP.turn_indices(cells) == [0, 2, 4, 5]
    assert PP.is_orthogonal(cells) and not PP.is_orthogonal([[0, 0], [1, 1]])


def test_cspace_matches_independent_body_check():
    pm = make_map((20, 20, 180, 100), obstacles=[(90, 20, 110, 60)])
    for yaw in (0, 90, 180, 270):                                         # every free cell: erosion C-space == rasterised body test
        valid, _ = pm.cspace(42, 31, yaw)
        for r, c in zip(*np.nonzero(pm.free)):
            assert valid[r, c] == pm.body_free(pm.centre_floor(r, c), 42, 31, yaw)[0], (yaw, r, c)


def test_rotation_check_blocked_in_corridor_free_in_open_space():
    pm = make_map((0, 40, 200, 75))                                       # corridor 35 cm high
    assert not PP.rotation_check(pm, [100, 57.5], 42, 31, 0, 90, 2)["ok"]
    pm2 = make_map((0, 0, 200, 120))
    r = PP.rotation_check(pm2, [100, 60], 42, 31, 0, 90, 2)
    assert r["ok"] and r["direction"] == "ccw"


def test_approach_candidate_faces_target_and_reports_gap():
    pm = make_map((60, 20, 200, 100))                                    # free only right of x = 60
    cands = PP.approach_candidates(pm, [50, 60], (35, 24), 3.5, [0, 90, 180, 270], 150, {"offset_x_m": None, "offset_y_m": None})
    by = {c["yaw_map_deg"]: c for c in cands}
    assert by[180]["status"] == "POSE_FOUND" and by[180]["facing"] == "-x_map"
    assert by[180]["centre_floor_cm"][0] - 21 >= 60 - 1e-6                # margin body inside the free area
    assert abs(by[180]["front_to_target_gap_cm"] - (by[180]["centre_floor_cm"][0] - 50 - 17.5)) < 1e-6
    assert by[180]["gripper_alignment_error_cm"] is None
    assert all(by[y]["status"] == "NO_COLLISION_FREE_POSE_ON_AXIS" for y in (0, 90, 270))


def test_leg_plan_is_orthogonal_and_collision_checked():
    pm = make_map((0, 0, 200, 120), obstacles=[(80, 0, 100, 80)])
    best, tried = PP.plan_leg(pm, "A", np.array([40.0, 30.0]), 0.0, pm.cell_of_floor([160, 30]), 0.0, (35, 24), 3.5, PCFG, False)
    assert best is not None and best["strategy"] == "no_rotation"
    d = PP.describe_leg(pm, best, (35, 24), 3.5, PCFG)
    assert d["checks"]["orthogonal_4_connected"] and d["collision_checks_passed"]
    assert d["n_turns"] <= 4


def test_heading_change_uses_an_open_cell_when_both_ends_are_narrow():
    # horizontal corridor (35 cm high) -> room 100 x 80 cm -> vertical corridor (35 cm wide); turning circle of 42 x 31 cm = 52 cm
    pm = make_map([(0, 60, 101, 95), (100, 40, 200, 120), (140, 0, 175, 41)])
    best, tried = PP.plan_leg(pm, "B", np.array([31.0, 77.0]), 0.0, pm.cell_of_floor([157.5, 25.0]), 90.0, (35, 24), 3.5, {**PCFG, "rotation_cell_candidates": 300}, False)
    status = {t["strategy"]: t["status"] for t in tried}
    assert status["rotate_at_start"] == "ROTATION_BLOCKED_AT_START"
    assert status["rotate_at_goal"] in ("ROTATION_BLOCKED_AT_GOAL", "ENDPOINT_NOT_IN_CSPACE")
    assert best is not None and best["strategy"] == "rotate_at_open_cell"
    d = PP.describe_leg(pm, best, (35, 24), 3.5, PCFG)
    assert d["collision_checks_passed"] and d["checks"]["orthogonal_4_connected"]
    rc = best["rotation_cell"]
    assert 100 < pm.centre_floor(*rc)[0] < 200 and 40 < pm.centre_floor(*rc)[1] < 120   # rotation happens in the room
    poses = PP.pose_list(best, d)
    assert [p["kind"] for p in poses].count("rotate_in_place") == 1 and poses[-1]["yaw_deg"] == 90.0


def test_unconfirmed_positions_are_refused(tmp_path):
    cfg = CT.load_config()
    cfg["object_positions_dir"] = str(tmp_path)
    photo = cfg["photos"][4]
    doc = {"photo": photo["name"], "photo_sha256": photo["sha256"], "status": "CANDIDATE",
           "robot": {"base_link_image_px": [1, 2], "yaw_map_deg": 0}, "cube": {"target_image_px": [1, 2]}, "placement": {"target_image_px": [1, 2]}}
    (tmp_path / f"{photo['name']}.yaml").write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):
        OP.load_confirmed(cfg, photo["name"])
    doc["status"] = "USER_CONFIRMED"
    (tmp_path / f"{photo['name']}.yaml").write_text(yaml.safe_dump(doc))
    assert OP.load_confirmed(cfg, photo["name"])[0]["status"] == "USER_CONFIRMED"
