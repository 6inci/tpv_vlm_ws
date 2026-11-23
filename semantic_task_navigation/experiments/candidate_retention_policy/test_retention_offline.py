"""Offline checks (no GPU, no model). From the project root:
  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/candidate_retention_policy
(1) policy logic on synthetic input; (2) state rules; (3) pins of the latest run (counts with AI labels on 8 photos, not accuracy)."""
import inspect
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import retention as R  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
TCFG = R.load_config()["models"]["target_inspection"]
ROI = [0, 0, 400, 600]


def scene(cubes):
    img = np.full((600, 800, 3), 40, np.uint8)
    for x, y in cubes:
        cv2.rectangle(img, (x, y), (x + 18, y + 18), (0, 0, 220), -1)
    return img


def test_cube_outside_the_roi_is_kept_and_roi_only_changes_the_order():
    m = R.build_candidates(scene([(600, 300), (100, 100)]), TCFG, ROI)
    assert len(m) == 2 and [c["inside_roi"] for c in m] == [True, False]          # nothing excluded; the ROI candidate is first


def test_budget_records_instead_of_deleting():
    m = R.build_candidates(scene([(40 + 60 * i, 500) for i in range(6)]), TCFG, ROI)
    c = R.apply_budget(m, 4)
    assert len(c) == 6 and sum(x["retention"].startswith("SENT") for x in c) == 4 and sum(x["retention"].startswith("BEYOND") for x in c) == 2


def test_oversize_detector_box_does_not_fuse_distinct_candidates():
    big = {"bbox_xyxy": [80, 80, 380, 180], "score": 0.9, "prompt": None, "source": "grounding_dino_full_image"}     # contains both cubes
    m = R.build_candidates(scene([(100, 100), (300, 100)]), TCFG, ROI, [big])
    small = [c for c in m if not c["oversize_detector_box"]]
    assert len(small) == 2 and all(c["source_types"] == ["hsv_color_shape"] for c in small)
    assert m[-1]["oversize_detector_box"] and len(m) == 3                           # kept as its own record, ordered last


def test_label_is_not_an_input_of_candidate_generation():
    assert "label" not in inspect.signature(R.build_candidates).parameters and "label" not in inspect.signature(R.apply_budget).parameters


def test_state_rules():
    two, one = ["grounding_dino_roi", "hsv_color_shape"], ["hsv_color_shape"]
    no_red = {"status": "no", "answer": {"observed_color": "red", "observed_object": "round object"}}
    no_orange = {"status": "no", "answer": {"observed_color": "orange", "observed_object": "cone"}}
    assert R.state_of({"status": "yes", "answer": {}}, two, "red")[0] == "VLM_ACCEPTED"         # never CONFIRMED
    assert R.state_of(no_red, two, "red")[0] == "PROBABLE" and R.state_of(no_red, two, "red")[1]  # disagreement is recorded
    assert R.state_of(no_red, one, "red")[0] == "REJECTED" and R.state_of(no_orange, two, "red")[0] == "REJECTED"
    assert R.state_of(None, two, "red")[0] == "PENDING_VERIFICATION" and R.state_of({"status": "not_run"}, two, "red")[0] == "PENDING_VERIFICATION"
    assert "CONFIRMED" not in inspect.getsource(R.state_of).replace("never CONFIRMED", "")


def test_photo_decision_never_picks_among_several():
    assert R.photo_decision(["PROBABLE", "PROBABLE", "REJECTED"]) == "AMBIGUOUS"
    assert R.photo_decision(["VLM_ACCEPTED", "PROBABLE", "PENDING_VERIFICATION"]) == "AMBIGUOUS"
    assert R.photo_decision(["PROBABLE", "REJECTED"]) == "PROBABLE" and R.photo_decision(["REJECTED"]) == "TARGET_NOT_FOUND"
    assert R.photo_decision(["PROBABLE", "PENDING_VERIFICATION"]).startswith("INCOMPLETE")


def test_latest_run_pins():
    runs = sorted((HERE / CFG["results_dir"]).glob("*_retention"))
    assert runs, "run retention.py first"
    r = json.loads((runs[-1] / "result.json").read_text(encoding="utf-8"))
    s = r["summary"]
    assert r["claims"]["model_calls_in_this_run"] == 0 and r["claims"]["accuracy_or_recall"].startswith("NOT_COMPUTED")
    assert r["provenance"]["v02_code_before"] == r["provenance"]["v02_code_after"] and r["provenance"]["v02_code_before"].startswith("254cfb31da7f")
    assert s["label_candidate_reaches_verification"] == 8 and s["v02_baseline_not_dropped_before_gate"] == 4
    assert all(p["n_candidates"] == p["n_sent_to_verification"] + p["n_beyond_budget"] for p in r["photos"])
    e1 = next(p for p in r["photos"] if p["photo_id"] == "U25070552")
    assert e1["label"]["rank"] == 1 and e1["label"]["state"] == "PROBABLE" and e1["photo_decision_from_stored_answers"] == "AMBIGUOUS"
    assert s["e1_states"]["VLM_ACCEPTED"] == 0 and s["e1_states"]["PROBABLE"] == 3
    assert all(p["dino_records"] == "NOT_AVAILABLE" and all(c["state"] == "PENDING_VERIFICATION" for c in p["candidates"]) for p in r["photos"] if p["photo_id"] != "U25070552")
    assert not any(k in s for k in ("accuracy", "recall", "success_rate"))
