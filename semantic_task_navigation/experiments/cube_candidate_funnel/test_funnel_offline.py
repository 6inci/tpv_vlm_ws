"""Offline checks (no GPU, no model). From the project root:
  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/cube_candidate_funnel

(1) the funnel logic on synthetic inputs, including cases where a stage MUST drop the label; (2) design rules (labels never reach candidate
generation, NOT_AVAILABLE is never counted as a drop, no accuracy is reported); (3) E1 reproduction of the stored v0.2 records;
(4) pins of what the latest run recorded (observed behaviour on 8 photos with AI labels, not evidence of accuracy).
"""
import inspect
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import funnel as F  # noqa: E402
from semantic_task_nav.config import load_config  # noqa: E402
from semantic_task_nav.geometry.coordinate_frames import RoiTransform  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
TCFG = load_config()["models"]["target_inspection"]
RULE = CFG["match"]
RED = (0, 0, 220)                                    # BGR, inside the v0.2 red HSV range


def scene(cubes, size=(800, 600)):
    img = np.full((size[1], size[0], 3), 40, np.uint8)
    for x, y in cubes:
        cv2.rectangle(img, (x, y), (x + 18, y + 18), RED, -1)
    return img


def stored(candidates, status):
    return {"n_candidates": len(candidates), "n_verified": len(candidates), "status": status, "candidates": candidates}


def stored_cand(box, verdict, answer):
    return {"candidate_id": "target_cand1", "bbox_xyxy": box, "sources": ["hsv_color_shape", "grounding_dino_roi"], "score": 0.4,
            "verification": {"status": verdict, "answer": {"answer": verdict, "observed_object": answer, "reason": "r"},
                             "call": {"prompt_id": "p", "raw_response": "RAW"}, "crop": {"path": "x.png"}}}


# ------------------------------------------------------------------------------------------------ (1) logic
def test_keeps_label_by_centre_or_iou():
    lab = [100, 100, 118, 118]
    assert F.keeps_label([95, 95, 125, 125], lab, RULE)                  # label centre inside
    assert F.keeps_label([100, 100, 118, 118], lab, RULE)
    assert not F.keeps_label([300, 300, 320, 320], lab, RULE)
    assert not F.keeps_label([120, 100, 140, 118], lab, RULE)            # next to it, centre outside, IoU 0


def test_tightest_candidate_is_chosen():
    lab = [100, 100, 118, 118]
    big, small = {"bbox_xyxy": [50, 50, 200, 200]}, {"bbox_xyxy": [98, 98, 120, 120]}
    assert F.find_label_candidate([big, small], lab, RULE) == 1


def test_cube_outside_roi_is_dropped_at_the_roi_stage_but_found_over_the_whole_image():
    img = scene([(100, 100), (600, 300)])                                # second cube lies outside the ROI
    roi = RoiTransform([0, 0, 400, 600], 1.0)
    p = F.photo_record("T", [600, 300, 618, 318], img, roi, TCFG, RULE)
    assert p["first_dropped_stage"] == "roi_search_area"
    assert p["stages"]["roi_search_area"]["distance_outside_roi_px"] == pytest.approx(209, abs=1)   # label centre x=609, ROI edge x=400
    assert p["stages"]["candidate_generation"]["status"] == F.NOT_REACHED and p["stages"]["qwen_gate"]["status"] == F.NOT_REACHED
    w = F.whole_image_counterfactual(img, [600, 300, 618, 318], roi, RULE, TCFG)
    assert w["label_in_candidates"] and w["n_candidates"] == 2 and w["n_outside_v02_roi"] == 1


def test_cube_inside_roi_without_stored_run_has_no_drop_and_gate_not_available():
    img = scene([(100, 100)])
    roi = RoiTransform([0, 0, 400, 600], 1.0)
    p = F.photo_record("T", [100, 100, 118, 118], img, roi, TCFG, RULE)
    assert p["stages"]["roi_search_area"]["status"] == F.KEPT and p["stages"]["candidate_generation"]["status"] == F.KEPT
    assert p["stages"]["rank_within_verify_limit"]["status"] == F.KEPT
    assert p["stages"]["qwen_gate"]["status"] == F.NOT_AVAILABLE          # not a drop: nothing was run
    assert p["first_dropped_stage"].startswith("NONE")


def test_stored_gate_no_is_recorded_as_the_dropping_stage_with_the_raw_reply():
    img = scene([(100, 100)])
    roi = RoiTransform([0, 0, 400, 600], 1.0)
    ti = stored([stored_cand([100, 100, 118, 118], "no", "round object")], "TARGET_NOT_FOUND")
    p = F.photo_record("T", [100, 100, 118, 118], img, roi, TCFG, RULE, ti)
    assert p["first_dropped_stage"] == "qwen_gate"
    g = p["stages"]["qwen_gate"]["stored"]
    assert g["observed_object"] == "round object" and g["raw_response"] == "RAW"
    assert p["stages"]["v02_decision"]["status"] == F.DROPPED


def test_candidate_beyond_the_verify_limit_is_dropped_at_the_rank_stage():
    img = scene([(100, 100)])
    roi = RoiTransform([0, 0, 400, 600], 1.0)
    decoys = [stored_cand([300 + 40 * i, 400, 318 + 40 * i, 418], "no", "cone") for i in range(TCFG["max_candidates_to_verify"])]
    cube = stored_cand([100, 100, 118, 118], "yes", "cube")
    cube["verification"] = {"status": "not_run", "reason": "beyond max_candidates_to_verify"}
    ti = stored(decoys + [cube], "TARGET_NOT_FOUND")
    p = F.photo_record("T", [100, 100, 118, 118], img, roi, TCFG, RULE, ti)
    assert p["stages"]["rank_within_verify_limit"]["status"] == F.DROPPED
    assert p["first_dropped_stage"] == "rank_within_verify_limit"


# ------------------------------------------------------------------------------------------------ (2) design rules
def test_labels_never_reach_candidate_generation():
    assert "label" not in inspect.signature(F.hsv_roi_candidates).parameters      # generation sees the image and v0.2 parameters only
    img = scene([(100, 100)])
    roi = RoiTransform([0, 0, 400, 600], 1.0)
    a = F.photo_record("T", [100, 100, 118, 118], img, roi, TCFG, RULE)["stages"]["candidate_generation"]["n_hsv_candidates_in_roi"]
    b = F.photo_record("T", [500, 500, 518, 518], img, roi, TCFG, RULE)["stages"]["candidate_generation"]["n_hsv_candidates_in_roi"]
    assert a == b == 1                                                   # the same candidates whatever the label says


def test_config_is_honest_and_images_match():
    assert CFG["label_status"] == "AI_PRELIMINARY_REVIEW"
    assert set(CFG["labels"]) == {i["id"] for i in CFG["images"]}
    for im in CFG["images"]:
        p = Path(CFG["image_dir"]) / im["file"]
        assert F.sha256_file(p) == im["sha256"]
        h, w = cv2.imread(str(p)).shape[:2]
        x1, y1, x2, y2 = CFG["labels"][im["id"]]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


# ------------------------------------------------------------------------------------------------ (3) E1 reproduction
def test_e1_stored_hsv_candidates_and_roi_are_reproduced():
    e1 = (HERE / CFG["stored"]["e1_run"]).resolve()
    rep = (HERE / CFG["stored"]["e1_repeat_run"]).resolve()
    ti = json.loads((e1 / "analysis/target_inspection.json").read_text(encoding="utf-8"))
    ti2 = json.loads((rep / "analysis/target_inspection.json").read_text(encoding="utf-8"))
    det = json.loads((e1 / "analysis/detections.json").read_text(encoding="utf-8"))
    anchor = next(o for o in det["objects"] if o["object_id"] == CFG["roi_anchor_object_id"])
    ref = next(i for i in CFG["images"] if i["id"] == CFG["reference_photo_id"])
    img = cv2.imread(str(Path(CFG["image_dir"]) / ref["file"]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    roi = F.make_roi(anchor["bbox_xyxy"], (img.shape[1], img.shape[0]), TCFG)
    r = F.reproduction_check(img, roi, TCFG, ti, ti2, CFG["reproduce"]["box_tolerance_px"])
    assert r["roi_transform_equal"] and r["hsv_equal"] and r["stored_repeat_run_identical"]


# ------------------------------------------------------------------------------------------------ (4) pins of the latest run
def latest():
    runs = sorted((HERE / CFG["results_dir"]).glob("*_funnel"))
    assert runs, "run funnel.py first"
    return json.loads((runs[-1] / "result.json").read_text(encoding="utf-8"))


def test_latest_run_pins():
    r = latest()
    assert r["provenance"]["v02_code_unchanged"] and r["provenance"]["v02_code_before"].startswith("254cfb31da7f")
    assert r["claims"]["model_calls_in_this_run"] == 0 and r["claims"]["accuracy_or_recall"].startswith("NOT_COMPUTED")
    assert all(p["camera_alignment_vs_E1"]["status"] in ("ALIGNED", "REFERENCE") for p in r["photos"])
    c = r["summary"]["first_dropped_stage_counts"]
    assert c["roi_search_area"] == 4 and c["qwen_gate"] == 1 and sum(c.values()) == 8
    e1 = next(p for p in r["photos"] if p["photo_id"] == "U25070552")
    assert e1["stages"]["qwen_gate"]["stored"]["observed_object"] == "round object"
    assert r["summary"]["whole_image_hsv_label_in_candidates"] == 8
    assert not any(k in r["summary"] for k in ("accuracy", "recall", "success_rate", "iou"))
