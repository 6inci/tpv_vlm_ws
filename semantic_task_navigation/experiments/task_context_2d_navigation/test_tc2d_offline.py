"""Offline checks for run_task_context.py (synthetic inputs; no model, no GPU).

  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/task_context_2d_navigation/test_tc2d_offline.py
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_task_context as T  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
REL = CFG["relations"]
TC = {"roi_margin_ratio": 0.6, "roi_min_margin_px": 120, "roi_zoom_long_side_px": 1600, "roi_max_zoom": 4.0}


def test_no_answer_in_prompts_rules_or_instructions():
    texts = [CFG["interpretation"]["template"], CFG["interpretation"]["rules_R1"], CFG["roles"]["source_template"], CFG["roles"]["destination_template"]] + [t["instruction"] for t in CFG["tasks"]]
    for t in texts:
        assert not [w for w in ("pallet", "팔레트", "grate", "격자", "1593", "맨홀") if w in t.lower()], t[:60]
    src = inspect.getsource(T.main)
    assert src.index("evaluation_reference.yaml") > src.index("routes.json") > src.index("model_calls.json")


def test_reference_class_and_regions():
    assert T.ref_class("Conveyor belt", REL["reference_classes"]) == "conveyor" and T.ref_class("white truck", REL["reference_classes"]) == "truck"
    assert T.ref_class("loading dock", REL["reference_classes"]) is None
    v = T.search_region("next_to", [100, 100, 200, 200], (1920, 1080), TC, REL)
    assert v["type"] == "vicinity" and v["rect"] == [0, 0, 320, 320]
    c = T.search_region("on", [100, 100, 200, 200], (1920, 1080), TC, REL)
    assert c["type"] == "containment" and T.in_region([120, 120, 150, 150], c, REL) and not T.in_region([180, 180, 260, 260], c, REL)
    assert T.search_region("near", None, (1920, 1080), TC, REL)["type"] == "NO_REFERENCE"


def test_source_ties_are_ambiguous_never_first_pick():
    tie = {"candidates": [{"id": 1, "role": "object_to_move"}, {"id": 2, "role": "other"}, {"id": 3, "role": "object_to_move"}], "object_to_move_id": 1}
    assert T.decide_source(tie, 3) == {"status": "AMBIGUOUS", "numbers": [1, 3], "why": "2 candidates rated object_to_move (tie; no first-pick)"}
    one = {"candidates": [{"id": 2, "role": "object_to_move"}], "object_to_move_id": 2}
    assert T.decide_source(one, 3)["status"] == "SELECTED" and T.decide_source({"candidates": [], "object_to_move_id": None}, 3)["status"] == "NOT_FOUND"


def _d(best, roles):
    return {"candidates": [{"id": k, "placement_role": r} for k, r in roles.items()], "best_destination_id": best}


def test_destination_policy_with_accessibility_and_alternatives():
    roles = {1: "suitable", 2: "suitable", 3: "possible", 4: "unsuitable"}
    assert T.decide_destination(_d(1, roles), {1: True, 2: True, 3: True}, 4)["kind"] == "DIRECT"
    alt = T.decide_destination(_d(1, roles), {1: False, 2: True, 3: True}, 4)
    assert alt["status"] == "SELECTED" and alt["number"] == 2 and alt["kind"] == "ALTERNATIVE"
    assert T.decide_destination(_d(1, {1: "suitable", 2: "suitable", 3: "suitable"}), {1: False, 2: True, 3: True}, 3)["status"] == "AMBIGUOUS"
    poss = T.decide_destination(_d(1, roles), {1: False, 2: False, 3: True}, 4)
    assert poss["number"] == 3 and poss["kind"] == "ALTERNATIVE_POSSIBLE"
    assert T.decide_destination(_d(1, roles), {1: False, 2: False, 3: False, 4: True}, 4)["status"] == "NOT_ACCESSIBLE"     # 'unsuitable' never selected
    assert T.decide_destination(_d(None, {1: "unsuitable"}), {1: True}, 1)["status"] == "NOT_FOUND"


def test_c1_baseline():
    vis = {"VLM_ACCEPTED": 3, "PROBABLE": 2, "PENDING_VERIFICATION": 1, "REJECTED": 0}
    cs = [{"id": "a", "state": "PROBABLE", "source_types": ["x", "y"]}, {"id": "b", "state": "PROBABLE", "source_types": ["x", "y"]}, {"id": "c", "state": "REJECTED", "source_types": ["x"]}]
    assert T.c1_source(cs, vis)["status"] == "AMBIGUOUS" and T.c1_source([{"id": "d", "score": .2}, {"id": "e", "score": .5}], vis)["id"] == "e"


class _R:
    """Tiny raster: 10 x 10 cells, polygons given in cell units."""
    def mask(self, polys):
        m = np.zeros((10, 10), bool)
        for p in polys:
            (x0, y0), (x1, y1) = p[0], p[2]
            m[y0:y1, x0:x1] = True
        return m


def test_occupancy_keeps_empty_forbidden_space_free():
    spec = {"traversable": [{"id": "Z02", "polygon": [[0, 0], [6, 0], [6, 6], [0, 6]]}], "static_obstacles": [{"id": "Z03", "polygon": [[2, 2], [3, 2], [3, 3], [2, 3]]}],
            "no_go": [{"id": "Z01", "polygon": [[7, 0], [9, 0], [9, 3], [7, 3]]}, {"id": "Z07", "polygon": [[7, 5], [9, 5], [9, 8], [7, 8]]}]}
    occ, nav = T.occupancy_and_constraints(spec, _R())
    assert occ[1, 8] == T.OCC["free"] and nav[1, 8] == T.NAV_CODES["FORBIDDEN"]          # pedestrian route: free but forbidden
    assert occ[2, 2] == T.OCC["occupied"] and nav[2, 2] == T.NAV_CODES["OBSTACLE"]
    assert occ[6, 8] == T.OCC["unknown"] and nav[6, 8] == T.NAV_CODES["FORBIDDEN"]        # workcell: equipment not measured
    assert occ[9, 0] == T.OCC["unknown"] and nav[0, 0] == T.NAV_CODES["ALLOWED"]
