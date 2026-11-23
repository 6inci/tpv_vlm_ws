"""Offline checks (no GPU, no model). From the project root:
  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q experiments/qwen_semantic_map_diagnostic/test_probe_offline.py
"""
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import semantic_map_probe as P  # noqa: E402
import visualize as VZ  # noqa: E402

CFG, PROMPTS = P.load_all()
PROMPT = P.render_prompt(PROMPTS, CFG["instruction"])


def good():
    return {"scene_summary": "a work floor",
            "objects": [
                {"id": "obj_1", "name": "robot", "bbox_2d": [800, 600, 900, 700], "mobility": "mobile_robot", "mobility_evidence": "wheels",
                 "estimated_function": "carry", "observed_evidence": ["wheels"], "uncertainty": "low", "uncertainty_note": ""},
                {"id": "obj_2", "name": "shelf", "bbox_2d": [100, 100, 200, 300], "mobility": "static_equipment_or_goods",
                 "mobility_evidence": "no wheels", "estimated_function": "store", "observed_evidence": ["shelves"], "uncertainty": "medium",
                 "uncertainty_note": "partly hidden"}],
            "relations": [{"id": "rel_1", "subject": "obj_1", "relation": "near", "object": "area_1", "basis": "observed", "evidence": "x"}],
            "semantic_areas": [{"id": "area_1", "function_name": "handling area", "bbox_2d": [700, 500, 1000, 800], "related_objects": ["obj_1"],
                                "function_evidence": ["marking"], "basis": "inferred", "uncertainty": "medium", "uncertainty_note": "n"}],
            "destination": {"interpreted_purpose": "ship", "decision": "proposed", "proposed_area_id": "area_1", "selection_evidence": "e",
                            "related_objects": ["obj_1"], "related_areas": ["area_1"], "unknowns": []},
            "unverifiable_or_unclear": []}


# ------------------------------------------------------------------ prompt
def test_instruction_is_the_users_sentence_and_nothing_about_this_scene_is_in_the_prompt():
    assert CFG["instruction"] == "빨간 큐브를 출하할 수 있는 적절한 장소로 옮겨줘." and CFG["instruction"] in PROMPT
    body = PROMPT.replace(CFG["instruction"], "").lower()
    for w in ("truck", "pallet", "container", "cone", "dock", "bay", "yard", "parking", "charging", "charger", "barrel", "conveyor",
              "whiteboard", "mecanum", "shelf", "cage", "yellow", "green", "z06", "polygon", "driving map", "traversable"):
        assert w not in body, w
    assert set(re.findall(r"\d+", body)) <= {"0", "1", "2", "10", "25", "40", "1000"}          # coordinate convention + schema ids + limits only


def test_prompt_says_nothing_about_where_or_how_many():
    body = PROMPT.lower()
    assert not re.search(r"\b(left|right|top|bottom|upper|lower|corner|centre|center)\b", body.replace("top-left corner", "").replace("bottom-right corner", ""))
    assert not re.search(r"\b(one|two|three|several|many)\s+(robot|vehicle|object)", body)


def test_frozen_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "FROZEN_FILE", tmp_path / "prompt_frozen.json")
    assert not P.check_frozen(CFG, PROMPTS)[0]
    (tmp_path / "prompt_frozen.json").write_text(json.dumps({"prompt_sha256": "x", "instruction": CFG["instruction"],
                                                             "prompt_version": "sm0", "frozen_at": "t"}))
    assert not P.check_frozen(CFG, PROMPTS)[0]
    (tmp_path / "prompt_frozen.json").write_text(json.dumps({"prompt_sha256": P.sha256_text(PROMPT), "instruction": CFG["instruction"],
                                                             "prompt_version": "sm0", "frozen_at": "t"}))
    assert P.check_frozen(CFG, PROMPTS)[0]


def test_image_selection_was_recorded_before_inference_and_the_original_is_untouched():
    ok, sel = P.selection_ok(CFG)
    assert ok and sel["recorded_BEFORE_any_model_inference"]
    orig = Path(CFG["image"]["source_path"])
    assert hashlib.sha256(orig.read_bytes()).hexdigest() == CFG["image"]["sha256"]
    assert sel["selected"]["used_in_earlier_experiment"] is None and len(sel["candidates"]) == 8


def test_the_probe_calls_only_qwen_and_no_detector_or_verifier():
    src = (HERE / "semantic_map_probe.py").read_text(encoding="utf-8")
    assert '"qwen_generate"' in src and "dino" not in src.lower().replace("grounding dino", "") and "dino_detect" not in src
    for banned in ("import verify", "grounding", "planner", "goal_selector", "sam2", "yolo"):
        assert banned not in src.replace("Grounding DINO", "").replace("# ", "").split('"""')[2].lower(), banned


# ------------------------------------------------------------------ parsing
def test_strict_reply_validates_and_schema_violations_are_listed_not_hidden():
    obj, status, info, errors = P.parse_reply(json.dumps(good()))
    assert status == "STRICT_JSON" and errors == []
    bad = good()
    bad["objects"][0]["mobility"] = "robot-ish"
    bad["objects"][1]["uncertainty"] = "none"
    obj, status, info, errors = P.parse_reply(json.dumps(bad))
    assert status == "STRICT_JSON" and len(errors) == 2                 # the reply is kept as it is; the violations are recorded
    assert obj["objects"][0]["mobility"] == "robot-ish"


def test_truncated_reply_is_salvaged_as_partial_and_labelled():
    text = json.dumps(good())
    cut = text[:text.index('"relations"')] + '"relations": [{"id": "rel_1", "subject": "obj_1", "rela'      # cut inside the 3rd section
    obj, status, info, errors = P.parse_reply(cut)
    assert status == "SALVAGED_PARTIAL" and len(obj["objects"]) == 2 and "relations" not in obj or obj["relations"] == []
    assert "NOT part of the recovered structure" in info["note"]
    assert P.parse_reply("not json at all")[1] == "UNPARSEABLE"
    # a '}' inside a string must not fool the cut finder
    tricky = '{"scene_summary": "has } and ] inside", "objects": [{"id": "obj_1", "name": "a", "bbox_2d": [1,2,3,4], "mobility": "uncertain"'
    obj2, st2, _, _ = P.parse_reply(tricky)
    assert st2 in ("SALVAGED_PARTIAL", "UNPARSEABLE")


def test_usable_extraction_drops_but_never_corrects():
    g = good()
    g["objects"].append({"id": "obj_3", "name": "bad box", "bbox_2d": [900, 900, 100, 100], "mobility": "uncertain"})       # inverted
    g["objects"].append({"id": "obj_4", "name": "out of range", "bbox_2d": [0, 0, 1500, 10], "mobility": "uncertain"})
    u, dropped = P.usable(g)
    assert [o["id"] for o in u["objects"]] == ["obj_1", "obj_2"] and len(dropped) == 2
    assert u["objects"][0]["bbox_2d"] == [800, 600, 900, 700]


def test_consistency_describes_the_models_own_contradictions():
    g = good()
    g["objects"].append(copy.deepcopy(g["objects"][1]))                                         # duplicate id + identical box
    g["objects"].append({"id": "obj_9", "name": "inside robot box", "bbox_2d": [810, 610, 850, 650], "mobility": "static_equipment_or_goods"})
    g["relations"].append({"id": "rel_2", "subject": "obj_77", "relation": "on", "object": "obj_1", "basis": "inferred", "evidence": ""})
    g["semantic_areas"].append({"id": "area_2", "function_name": "copy", "bbox_2d": [100, 100, 200, 300], "related_objects": ["obj_55"]})
    g["destination"]["proposed_area_id"] = "area_8"
    c, _ = P.consistency(g)
    assert c["duplicate_objects_ids"] == ["obj_2"] and c["relations_referencing_missing_ids"] == ["rel_2"]
    assert c["near_identical_object_boxes"] and c["area_box_copies_an_object_box"][0][:2] == ["area_2", "obj_2"]
    assert c["area_related_objects_missing"] == {"area_2": ["obj_55"]} and c["destination"]["proposed_area_exists"] is False
    assert c["boxes_of_mobile_robot_entries_containing_other_objects"] == {"obj_1": ["obj_9"]}


def test_one_run_guard(tmp_path):
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "manifest.json").write_text(json.dumps({"prompt_sha256": "p", "image_sha256": "i"}))
    (tmp_path / "r2").mkdir()
    (tmp_path / "r2" / "manifest.json").write_text(json.dumps({"prompt_sha256": "other", "image_sha256": "i"}))
    assert P.previous_runs(tmp_path, "p", "i") == ["r1"] and P.previous_runs(tmp_path, "p", "z") == []


# ------------------------------------------------------------------ visualization
def test_object_and_area_boxes_are_drawn_in_different_ways_and_nothing_is_added(tmp_path):
    img = np.full((1080, 1920, 3), 60, np.uint8)
    meta = {"run_id": "t", "parse_status": "STRICT_JSON", "image_name": "x.png"}
    paths = VZ.render_all(img, good(), tmp_path, meta, P.usable)
    om, am, cp = (cv2.imread(str(paths[k])) for k in ("object_map", "semantic_area_map", "comparison"))
    assert om.shape[1] == am.shape[1] == 1920 and om.shape[0] > 1080 and cp.shape[1] > 2 * 900
    assert not np.array_equal(om[:1080], am[:1080])
    # the robot box is drawn magenta in the object map, and the area map has no magenta object box
    robot_px = om[int(0.6 * 1080), int(0.8 * 1920)]
    assert tuple(robot_px) == (255, 0, 255)
    assert len(paths["review"]) == 1
    assert np.array_equal(img, np.full((1080, 1920, 3), 60, np.uint8))                      # the input array is not modified


def test_rendering_survives_a_reply_with_no_areas_and_a_withheld_destination(tmp_path):
    g = good()
    g["semantic_areas"] = []
    g["destination"] = {"interpreted_purpose": "ship", "decision": "withheld", "proposed_area_id": None, "selection_evidence": "",
                        "related_objects": [], "related_areas": [], "unknowns": ["x"]}
    paths = VZ.render_all(np.zeros((1080, 1920, 3), np.uint8), g, tmp_path, {"run_id": "t", "parse_status": "S", "image_name": "x"}, P.usable)
    assert paths["semantic_area_map"].exists()
