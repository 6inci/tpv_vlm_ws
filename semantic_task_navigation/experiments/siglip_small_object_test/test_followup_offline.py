"""Offline checks of the follow-up logic (no model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_followup as F  # noqa: E402

CFG = F.FC["exemplar"]
LABELS = ["a red cube", "a small pallet", "a metal grate"]


def test_parse_label():
    assert F.parse_label('{"label": "a metal grate"}', LABELS) == ("a metal grate", None)
    assert F.parse_label('```json\n{"label": "A Small Pallet."}\n```', LABELS) == ("a small pallet", None)
    assert F.parse_label("a pallet", LABELS)[1] == "NO_JSON"
    assert F.parse_label('{"label": "pallet"}', LABELS)[1].startswith("NOT_IN_LIST")


def photo():
    objs = [{"key": "refobj_cube", "kind": "cube", "box": [110, 110, 127, 127]},
            {"key": "refobj_pad_with_cube", "kind": "pad_with_cube", "box": [100, 100, 145, 140]},
            {"key": "refobj_empty_pad_0", "kind": "empty_pad", "box": [300, 100, 345, 140]}]
    dino = [{"id": "m1", "bbox_xyxy": [100, 100, 145, 141]}, {"id": "m2", "bbox_xyxy": [301, 100, 345, 140]},
            {"id": "m3", "bbox_xyxy": [302, 101, 346, 141]},   # duplicate of m2 (suppressed by NMS)
            {"id": "m4", "bbox_xyxy": [600, 600, 650, 650]}, {"id": "m5", "bbox_xyxy": [110, 110, 127, 128]}]
    return {"objects": objs, "dino": dino}


def test_exemplar_recall_nms_and_margin():
    sims = {"m1": 0.7, "m2": 0.9, "m3": 0.89, "m4": 0.6, "m5": 0.5}
    r = F.exemplar_eval(photo(), sims, "pallet", [900, 900, 945, 940], CFG)
    assert r["n_targets"] == 2 and r["recall_at_n"] == 1.0                      # m3 would push m1 out without NMS
    assert r["margin"] == round(0.7 - 0.6, 4)
    sims["m4"] = 0.8
    r = F.exemplar_eval(photo(), sims, "pallet", [900, 900, 945, 940], CFG)
    assert r["recall_at_n"] == 0.5 and r["margin"] < 0


def test_exemplar_same_place_is_left_out():
    sims = {"m1": 0.1, "m2": 0.99, "m3": 0.98, "m4": 0.6, "m5": 0.5}
    r = F.exemplar_eval(photo(), sims, "pallet", [300, 100, 345, 140], CFG)     # exemplar = this very pad position
    assert r["left_out_same_place"] == ["refobj_empty_pad_0"] and r["n_targets"] == 1
    assert all(t["id"] not in ("m2", "m3") for t in r["top"])                    # its boxes are not ranked
    r = F.exemplar_eval(photo(), {"m1": 0.1, "m2": 0.2, "m3": 0.2, "m4": 0.3, "m5": 0.9}, "cube", [0, 0, 17, 17], CFG)
    assert r["top1_hit"] and r["top1_contains_cube_centre"]
