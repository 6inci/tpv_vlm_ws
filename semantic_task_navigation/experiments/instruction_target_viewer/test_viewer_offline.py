#!/usr/bin/env python3
"""Offline checks of the viewer back end (no model call, no GPU):
  /home/binci/miniconda3/envs/stnav_gui/bin/python experiments/instruction_target_viewer/test_viewer_offline.py
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import viewer_pipeline as V  # noqa: E402

GR = {"region_words": ["floor", "area", "zone", "marking", "surface"], "min_image_frac": 0.03}


def test_pure():
    assert V.photo_id("/x/undistorted_25070552942982.png") == "U25070552" and V.photo_id("/x/scene.png") == "scene"
    assert V.chunks(list(range(45)), 20) == [list(range(20)), list(range(20, 40)), list(range(40, 45))] and V.chunks([], 20) == []
    one = {"candidates": [{"id": 1, "role": "object_to_move"}], "object_to_move_id": 1}
    assert V.merge_replies([one], [3], "object_to_move_id") is one                        # one image: the reply itself (task_context behaviour)
    two = V.merge_replies([one, {"candidates": [{"id": 2, "role": "object_to_move"}], "object_to_move_id": 2}], [20, 5], "object_to_move_id")
    assert [c["id"] for c in two["candidates"]] == [1, 22] and two["object_to_move_id"] is None   # two bests -> none kept
    assert V.merge_replies([one, None], [20, 5], "object_to_move_id") is None              # any invalid reply -> invalid
    d = V.TC.decide_source(two, 25)
    assert d["status"] == "AMBIGUOUS" and d["numbers"] == [1, 22]                          # ties are never first-picked
    img = (1920, 1080)
    assert V.goal_kind({"bbox_xyxy": [0, 0, 40, 30], "phrases": ["floor marking", "yellow marked area"]}, None, img, GR)[0] == "GOAL_REGION"
    assert V.goal_kind({"bbox_xyxy": [0, 0, 40, 30], "phrases": ["pallet"]}, ["road surface"], img, GR)[0] == "GOAL_REGION"
    assert V.goal_kind({"bbox_xyxy": [0, 0, 600, 400], "phrases": ["pallet"]}, ["pallet"], img, GR)[0] == "GOAL_REGION"
    assert V.goal_kind({"bbox_xyxy": [0, 0, 40, 30], "phrases": ["pallet", "floor marking"]}, ["grate"], img, GR)[0] == "OBJECT"
    assert V.placement_status("OBJECT", {"placement_role": "suitable", "load_state": "empty"}) == "VLM_SUITABLE_EMPTY"
    assert V.placement_status("OBJECT", {"placement_role": "possible", "load_state": "empty"}) == "PLACEMENT_UNVERIFIED"
    assert V.placement_status("GOAL_REGION", {"placement_role": "suitable", "load_state": "empty"}) == "PLACEMENT_UNVERIFIED"
    rated = {1: {"placement_role": "suitable"}, 2: {"placement_role": "suitable"}, 3: {"placement_role": "possible"}}
    assert V.tied_destinations(rated, {1: {"accessible": True}, 2: {"accessible": True}}) == [1, 2]
    assert V.tied_destinations(rated, {1: {"accessible": False}, 2: {"accessible": False}, 3: {"accessible": True}}) == []
    assert V.first_failure([V.stage("a", "OK", ""), V.stage("b", "REUSED", ""), V.stage("c", "NOT_CHECKED", ""), V.stage("d", "UNCERTAIN", "")]) == "d"


def test_no_answers_in_viewer():
    """No evaluation reference is read, and no box of it is written in the viewer code or config."""
    ref = (V.TC.HERE / "evaluation_reference.yaml").read_text(encoding="utf-8")
    import yaml
    boxes = [o["bbox_xyxy_px"] for o in yaml.safe_load(ref)["E1_objects"].values()]
    for f in ("viewer_pipeline.py", "viewer_app.py", "config.yaml"):
        p = HERE / f
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        assert "evaluation_reference" not in t, f
        for b in boxes:
            assert str(b) not in t and ", ".join(map(str, b)) not in t, (f, b)


def test_draw():
    img = np.zeros((1080, 1920, 3), np.uint8)
    base = {"run_id": "run_001_X", "rules": "R1", "first_failure": None, "instruction": "t", "photo": {"photo_id": "X", "path": "/x/X.png"}, "interpretation": {"parsed": {"source_object": {"name_en": "red cube"}}},
            "references": {"source": {"name": "conveyor", "relation": "next_to", "box": [200, 200, 400, 700], "region": {"type": "vicinity", "rect": [80, 80, 600, 900]}},
                           "destination": {"name": "truck", "relation": "near", "box": [1400, 0, 1700, 200], "region": {"type": "vicinity", "rect": [1300, 0, 1900, 300]}}}}
    sel = {**base, "source": {"status": "SELECTED", "ids": ["c1"], "boxes": [[420, 410, 460, 450]], "ratings": {"c1": {"appearance": ["red cube"]}}, "candidates": [{"id": "c1", "bbox_xyxy": [420, 410, 460, 450]}]},
           "destination": {"status": "SELECTED", "id": "m5", "bbox_xyxy": [1590, 250, 1640, 290], "kind": "DIRECT", "goal_kind": "OBJECT", "placement": "VLM_SUITABLE_EMPTY",
                           "ratings": {"m5": {"appearance": ["support"], "placement_role": "suitable"}}, "candidates": [{"id": "m5", "bbox_xyxy": [1590, 250, 1640, 290]}],
                           "accessibility": {"m5": {"accessible": True, "approach_goal_image_px": [1600, 330]}}}}
    amb = {**base, "source": {"status": "AMBIGUOUS", "ids": ["c1", "c2"], "boxes": [[420, 410, 460, 450], [520, 280, 550, 300]]},
           "destination": {"status": "NOT_ACCESSIBLE", "id": None, "candidates": [{"id": "m9", "bbox_xyxy": [1450, 20, 1650, 140]}],
                           "accessibility": {"m9": {"accessible": False, "gap_to_drivable_cm": 62.0}}}}
    for r in (sel, amb, {**base, "source": {"status": "NOT_FOUND"}, "destination": {"status": "NO_REFERENCE"}}):
        for c in (False, True):
            out = V.draw(img, r, show_candidates=c)
            assert out.shape == img.shape and out.any()
    s = V.summary(sel)
    assert s["판단 상태"].startswith("A SELECTED · B SELECTED") and "m5" in s["B 목적 대상"]


def test_e1_context_without_models():
    """E1: stored references and candidates (no model call); accessibility from the existing Python rule."""
    with tempfile.TemporaryDirectory() as td:
        v = V.Viewer(session_dir=Path(td) / "s", progress=lambda m: None)
        e1 = next(p for p in v.list_photos() if p["photo_id"] == "U25070552")
        ph = v.photo(e1["path"])
        assert v.client is None, "E1 context must not start the model worker"
        assert ph["alignment"]["status"] == "ALIGNED" and ph["refs"]["robot"]["start_img"] and len(ph["dino"]) > 50 and ph["cube"]
        t = time.time()
        a = v.access(ph, ph["dino"][0])
        print("accessibility of", ph["dino"][0]["id"], {k: a[k] for k in ("accessible", "gap_to_drivable_cm", "approach_reachable_from_start")}, f"{time.time() - t:.1f}s")
        assert a["accessible"] in (True, False)
        assert len(v.list_photos()) >= 1


if __name__ == "__main__":
    for f in (test_pure, test_no_answers_in_viewer, test_draw, test_e1_context_without_models):
        f()
        print("ok", f.__name__)
