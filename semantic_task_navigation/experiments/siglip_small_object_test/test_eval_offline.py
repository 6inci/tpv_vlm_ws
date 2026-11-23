"""Offline checks of the evaluation logic (no model). Run: $PY -m pytest -q -p no:cacheprovider experiments/siglip_small_object_test"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_test as T  # noqa: E402

EV = T.CFG["evaluation"]


def rank(*labels):
    return {"combined": [(lb, round(0.5 / (i + 1), 3)) for i, lb in enumerate(labels)]}


def test_reference_has_24_objects_on_8_photos():
    refs = T.load_reference()
    assert len(refs) == 8 and len({r["sha256"] for r in refs}) == 8
    kinds = [o["kind"] for r in refs for o in r["objects"]]
    assert (kinds.count("cube"), kinds.count("pad_with_cube"), kinds.count("empty_pad")) == (8, 8, 8)   # 7 from STEP 4-A + 1 addition (photo_6)
    added = [(r["photo"], o["key"]) for r in refs for o in r["objects"] if o["key"].startswith("refobj_added")]
    assert added == [("photo_6", "refobj_added_0")]


def test_object_without_oracle_crop():
    obj = {"key": "refobj_added_0", "kind": "empty_pad", "box": [0, 0, 40, 40]}
    r = T.evaluate_object(obj, [{"id": "m1", "bbox_xyxy": [0, 0, 40, 41], "score": 0.2, "phrases": []}], {"m1": rank("a metal grate")}, None, EV)
    assert r["outcome"] == "NAMED_OTHER" and r["siglip_oracle_top3"] is None and r["oracle_top1_correct"] is None


def test_outcomes():
    obj = {"key": "refobj_empty_pad_0", "kind": "empty_pad", "box": [100, 100, 140, 140]}
    cands = [{"id": "m1", "bbox_xyxy": [101, 101, 141, 141], "score": 0.3, "phrases": ["pallet"]},
             {"id": "m2", "bbox_xyxy": [300, 300, 340, 340], "score": 0.9, "phrases": ["shelf"]}]
    oracle = rank("a metal grate", "a small pallet")
    r = T.evaluate_object(obj, cands, {"m1": rank("a small pallet", "a metal grate"), "m2": rank("a shelf")}, oracle, EV)
    assert r["candidate"] == "m1" and r["detected"] and r["outcome"] == "IDENTIFIED"
    assert not r["oracle_top1_correct"] and r["oracle_topk_correct"]
    r = T.evaluate_object(obj, cands, {"m1": rank("a metal grate", "a wooden pallet"), "m2": rank("a shelf")}, oracle, EV)
    assert r["outcome"] == "NAMED_OTHER" and r["identified_topk"]
    r = T.evaluate_object(obj, cands[1:], {"m2": rank("a small pallet")}, oracle, EV)        # right name on the wrong box is not a hit
    assert r["outcome"] == "NOT_DETECTED" and not r["identified_top1"]


def test_named_targets_marks_unmatched():
    objs = [{"key": "refobj_cube", "kind": "cube", "box": [10, 10, 27, 27]}]
    cands = [{"id": "m1", "bbox_xyxy": [10, 10, 27, 28], "score": 0.2}, {"id": "m2", "bbox_xyxy": [500, 500, 560, 560], "score": 0.2},
             {"id": "m3", "bbox_xyxy": [0, 0, 5, 5], "score": 0.2}]
    out = T.named_targets(cands, {"m1": rank("a red cube"), "m2": rank("a wooden pallet"), "m3": rank("a truck")}, objs, EV)
    assert [(t["id"], t["group"], t["matches_reference"]) for t in out] == [("m1", "cube", "refobj_cube"), ("m2", "pallet", None)]
