"""Offline checks for select_targets.py (synthetic inputs; no model, no GPU).

  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/contextual_target_selection/test_cts_offline.py
"""
import inspect
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import select_targets as S  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def test_prompts_and_rules_carry_no_answer():
    for k in ("destination_zone", "source_zone", "destination_verify"):
        t = CFG["qwen"][k]["template"].lower()
        assert not [w for w in ("pallet", "팔레트", "grate", "격자", "1593", "cone", "콘") if w in t]
    src = inspect.getsource(S.main)
    assert src.index("evaluation_reference") > src.index("routes.json") > src.index("selection.json")
    for fn in (S.zone_members, S.view_rect, S.decide_destination, S.decide_source, S.snap_anchor):
        assert "evaluation_reference" not in inspect.getsource(fn)


def test_snap_anchor():
    objs = [{"object_id": "truck_1", "bbox_xyxy": [100, 0, 200, 100]}, {"object_id": "x", "bbox_xyxy": [500, 500, 600, 600]}]
    assert S.snap_anchor([98, 0, 205, 104], objs, 0.3)[0] == [100, 0, 200, 100]
    assert S.snap_anchor([300, 300, 320, 320], objs, 0.3)[0] == [300, 300, 320, 320] and S.snap_anchor(None, objs, 0.3)[0] is None


def test_zone_members_keep_border_objects_and_separate_regions():
    zone = [0, 0, 100, 100]
    cands = [{"id": "in", "bbox_xyxy": [10, 10, 20, 20]}, {"id": "border", "bbox_xyxy": [90, 90, 120, 110]},       # centre (105, 100) outside, 1/6 inside
             {"id": "half", "bbox_xyxy": [80, 40, 120, 60]}, {"id": "region", "bbox_xyxy": [0, 0, 90, 90]}, {"id": "out", "bbox_xyxy": [300, 300, 310, 310]}]
    inc, reg = S.zone_members(cands, zone, 0.5)
    assert [c["id"] for c in inc] == ["in", "half"] and [c["id"] for c in reg] == ["region"]
    assert S.view_rect(zone, [c["bbox_xyxy"] for c in inc], (1000, 1000)) == [0, 0, 120, 100]


def test_destination_decision_rules():
    z = lambda best, role: {"candidates": [{"id": 1, "placement_role": role}, {"id": 2, "placement_role": "possible"}], "best_destination_id": best}   # noqa: E731
    assert S.decide_destination(z(1, "suitable"), {"placement_role": "possible"}, 2)["status"] == "SELECTED"
    assert S.decide_destination(z(1, "suitable"), {"placement_role": "unsuitable"}, 2)["status"] == "UNVERIFIED"
    assert S.decide_destination(z(1, "possible"), {"placement_role": "suitable"}, 2)["status"] == "UNVERIFIED"
    assert S.decide_destination(z(1, "suitable"), None, 2)["status"] == "UNVERIFIED"
    assert S.decide_destination(z(None, "suitable"), None, 2)["status"] == "AMBIGUOUS"
    assert S.decide_destination({"candidates": [{"id": 1, "placement_role": "unsuitable"}], "best_destination_id": None}, None, 1)["status"] == "NOT_FOUND"
    assert S.decide_destination(z(5, "suitable"), {"placement_role": "suitable"}, 2)["status"] in ("AMBIGUOUS", "NOT_FOUND")
    assert S.decide_destination(None, None, 2)["status"] == "AMBIGUOUS"


def test_source_is_independent_of_destination():
    assert S.decide_source({"candidates": [], "object_to_move_id": 3}, 4) == {"status": "SELECTED", "number": 3, "why": "object_to_move_id"}
    assert S.decide_source({"candidates": [], "object_to_move_id": None}, 4)["status"] == "AMBIGUOUS"
    assert "dst" not in str(inspect.signature(S.decide_source))
