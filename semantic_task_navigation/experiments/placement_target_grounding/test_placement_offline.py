"""Offline checks for placement_target.py (synthetic inputs; no model, no GPU).

  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/placement_target_grounding/test_placement_offline.py
"""
import inspect
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import placement_target as P  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
E3 = CFG["exp3"]


def test_prompts_name_no_scene_object_and_no_reference():
    """The model must not be told what to find: no object/place word of the scene beyond the user's command."""
    forbidden = ["pallet", "팔레트", "grate", "격자", "cone", "콘", "truck", "트럭", "dock", "하역", "bay", "floor", "바닥", "1593"]
    for text in (CFG["exp1"]["prompt"], E3["comparison"]["prompt"]):
        body = text.lower()
        assert not [w for w in forbidden if w in body], [w for w in forbidden if w in body]


def test_selection_code_cannot_see_the_reference():
    for fn in (P.select, P.apply_comparison, P.dino_phrases, P.merge, P.detections, P.zone_crop_rect):
        assert "evaluation_reference" not in inspect.getsource(fn) and "reference" not in str(inspect.signature(fn))
    src = inspect.getsource(P.main)
    assert src.index("evaluate(") > src.index("exp3_selection.json") > src.index("exp2_candidates.json")


def test_zone_crop_rect_rule_and_clipping():
    zc = {"margin_ratio": 0.25, "margin_min_px": 120}
    assert P.zone_crop_rect([1248, 0, 1824, 280.8], (1920, 1080), zc) == [1104, 0, 1920, 425]
    assert P.zone_crop_rect([500, 500, 520, 520], (1920, 1080), zc) == [380, 380, 640, 640]       # min margin wins for a small box


def test_qwen_box_mapping_full_and_crop():
    assert P.qwen_box_to_image([0, 0, 500, 1000], [0, 0, 1920, 1080]) == [0.0, 0.0, 960.0, 1080.0]
    assert P.qwen_box_to_image([500, 500, 1000, 1000], [1104, 0, 1920, 425]) == [1512.0, 212.5, 1920.0, 425.0]
    assert P.qwen_box_to_image([10, 10, 5, 20], [0, 0, 100, 100]) is None and P.qwen_box_to_image(None, [0, 0, 1, 1]) is None


def test_phrase_rule_uses_only_model_names():
    resp = {"A": {"candidates": [{"name_en": "Shipping Truck.", "suitability": "suitable"}, {"name_en": "work table", "suitability": "unsuitable"}],
                  "selected": "shipping truck", "decision": "selected"},
            "B": {"candidates": [{"name_en": "shipping truck", "suitability": "possible"}], "selected": None, "decision": "cannot_determine"}, "C": None}
    phrases, org = P.dino_phrases(resp, 10)
    assert phrases == ["shipping truck", "work table"]
    assert [o["condition"] for o in org["shipping truck"]] == ["A", "B"] and org["shipping truck"][0]["selected"] is True
    assert P.norm_phrase("a b c d e f g h i j k l", 10) == "a b c d e f g h i j"


def _raw(boxes, size=(200, 100)):
    return {"image_size": list(size), "results": [{"prompt": "thing.", "boxes_xyxy": [b for b, _ in boxes], "scores": [s for _, s in boxes], "phrases": ["thing"] * len(boxes)}]}


def test_detections_map_back_and_keep_everything():
    d = P.detections(_raw([([0, 0, 20, 10], 0.3), ([0, 0, 200, 100], 0.9)]), "ROI_X2", [100, 50, 200, 100], 2.0, 0.25)
    assert [x["status"] for x in d] == ["FILTERED_OVERSIZE", "KEPT"] and [x["rank_in_condition"] for x in d] == [1, 2]
    assert d[1]["bbox_xyxy"] == [100.0, 50.0, 110.0, 55.0]


def test_merge_is_iou_only_and_keeps_small_inside_large():
    dets = [{"id": "F_1", "condition": "FULL", "prompt": "area", "score": 0.5, "status": "KEPT", "bbox_xyxy": [0, 0, 100, 100]},
            {"id": "F_2", "condition": "FULL", "prompt": "thing", "score": 0.3, "status": "KEPT", "bbox_xyxy": [40, 40, 50, 50]},
            {"id": "R_1", "condition": "ROI", "prompt": "thing", "score": 0.4, "status": "KEPT", "bbox_xyxy": [40, 40, 51, 50]}]
    m = P.merge(dets, 0.5)
    assert len(m) == 2 and m[1]["member_ids"] == ["R_1", "F_2"] and m[1]["conditions"] == ["FULL", "ROI"]


def _cand(i, box, phrases, score=0.3):
    return {"id": i, "bbox_xyxy": box, "phrases": phrases, "score": score, "conditions": ["FULL"], "rank": 1}


ORG = {"tray": [{"condition": "A", "candidate_index": 0, "suitability": "suitable", "selected": True}],
       "shelf": [{"condition": "A", "candidate_index": 1, "suitability": "unsuitable", "selected": False}]}
RESP = {"A": {"selected": "tray", "decision": "selected", "candidates": []}, "B": None}
ROI = [0, 0, 100, 100]


def test_unique_linked_box_is_selected_without_comparison():
    d = P.select([_cand("m1", [10, 10, 20, 20], ["tray"]), _cand("m2", [50, 50, 60, 60], ["shelf"])], ORG, RESP, {"tray": [[10, 10, 21, 20]]}, ROI, E3, 10)
    assert d["status"] == "SELECTED" and d["selected"] == "m1" and not d["needs_comparison"]


def test_two_boxes_or_no_qwen_link_need_one_comparison():
    d = P.select([_cand("m1", [10, 10, 20, 20], ["tray"]), _cand("m2", [30, 30, 40, 40], ["tray"])], ORG, RESP, {"tray": [[10, 10, 21, 20]]}, ROI, E3, 10)
    assert d["status"] is None and d["needs_comparison"] == ["m1", "m2"]
    d = P.select([_cand("m1", [10, 10, 20, 20], ["tray"])], ORG, RESP, {"tray": [None]}, ROI, E3, 10)
    assert d["needs_comparison"] == ["m1"]


def test_outside_context_or_unsuitable_is_not_relevant():
    d = P.select([_cand("m1", [150, 150, 160, 160], ["tray"]), _cand("m2", [10, 10, 20, 20], ["shelf"])], ORG, RESP, {}, ROI, E3, 10)
    assert d["n_relevant"] == 0 and d["status"] == "NOT_FOUND"
    d = P.select([], ORG, {"A": {"selected": None, "decision": "cannot_determine", "candidates": []}}, {}, ROI, E3, 10)
    assert d["status"] == "AMBIGUOUS"


def test_different_selected_names_force_comparison():
    org = {**ORG, "shelf": [{"condition": "B", "candidate_index": 0, "suitability": "suitable", "selected": True}]}
    resp = {"A": RESP["A"], "B": {"selected": "shelf", "decision": "selected", "candidates": []}}
    d = P.select([_cand("m1", [10, 10, 20, 20], ["tray"]), _cand("m2", [50, 50, 60, 60], ["shelf"])], org, resp, {"tray": [[10, 10, 20, 20]]}, ROI, E3, 10)
    assert d["rule"].startswith("1:") and set(d["needs_comparison"]) == {"m1", "m2"}


def test_comparison_needs_a_suitable_shown_number():
    base = {"needs_comparison": ["m1", "m2"], "rule": "r", "status": None, "selected": None}
    ok = P.apply_comparison(dict(base), {"candidates": [{"id": 2, "suitable": "yes"}], "best": 2}, None)
    assert ok["status"] == "SELECTED" and ok["selected"] == "m2"
    for obj in ({"candidates": [{"id": 2, "suitable": "uncertain"}], "best": 2}, {"candidates": [], "best": None}, {"candidates": [{"id": 3, "suitable": "yes"}], "best": 3}):
        assert P.apply_comparison(dict(base), obj, None)["status"] == "AMBIGUOUS"
    assert P.apply_comparison(dict(base), None, "INVALID")["status"] == "AMBIGUOUS"
