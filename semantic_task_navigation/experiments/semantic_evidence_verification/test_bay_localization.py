"""Offline checks of bay_localization (no GPU, no model). From the project root:
  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q experiments/semantic_evidence_verification/test_bay_localization.py

(1) synthetic photos with a known painted L: the method must recover it, and must NOT report a bay when there is no paint,
when the paint is a solid line (no dashes) or when only one side exists. (2) design rules: object boxes are not floor;
nothing from the answer side is read. (3) a pin of what the method gives on the stored E1 photo (observed behaviour, not accuracy).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bay_localization as B  # noqa: E402
import observations as O  # noqa: E402
import verifier2 as V  # noqa: E402

V02 = B.load_cfg()
W, H = 800, 500
QBOX = [100, 20, 780, 420]
TRUCK = {"object_id": "truck_1", "class_name": "truck", "bbox_xyxy": [330, 120, 560, 300]}
CONE = {"object_id": "cone_1", "class_name": "traffic_cone", "bbox_xyxy": [150, 330, 190, 390]}


def synth(left=True, bottom=True, dashed=True, dark_floor=True):
    img = np.full((H, W, 3), 55 if dark_floor else 200, np.uint8)   # the real mat is ~50-60 gray
    corner = (250, 380)
    def side(p, q, n):
        for i in range(n):
            a = np.array(p) + (np.array(q) - np.array(p)) * (i / n)
            b = np.array(p) + (np.array(q) - np.array(p)) * ((i + 0.55) / n) if dashed else np.array(p) + (np.array(q) - np.array(p)) * ((i + 1) / n)
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (245, 245, 245), 5)
    if bottom:
        side(corner, (640, 380), 12)
    if left:
        side(corner, (250, 90), 9)
    return img


def test_synthetic_L_is_recovered():
    img = synth()
    res, _ = B.localize(img, QBOX, [TRUCK, CONE], V02)
    assert res["status"] == "FLOOR_MARKING_ESTABLISHED", [c for c in res["checks"] if c["status"] != "PASS"]
    xy = np.array(res["bay_polygon_xy"])
    assert np.allclose(xy[0], [250, 380], atol=8)          # the corner
    # the painted extent ends with the last dash (start + 0.55 of the spacing + line cap), not at the nominal end of the side
    assert xy[:, 0].max() == pytest.approx(628, abs=6)      # far end of the bottom side
    assert xy[:, 1].min() == pytest.approx(102, abs=6)      # far end of the left side


def test_no_paint_is_not_a_bay():
    res, _ = B.localize(synth(left=False, bottom=False), QBOX, [TRUCK, CONE], V02)
    assert res["status"] == "FLOOR_MARKING_NOT_ESTABLISHED" and res["bay_polygon_xy"] is None and res["free_floor"] is None


def test_one_side_only_is_not_a_bay():
    res, _ = B.localize(synth(left=False), QBOX, [TRUCK, CONE], V02)
    assert res["status"] == "FLOOR_MARKING_NOT_ESTABLISHED"


def test_solid_line_is_not_dashes():
    res, _ = B.localize(synth(dashed=False), QBOX, [TRUCK, CONE], V02)
    assert res["status"] == "FLOOR_MARKING_NOT_ESTABLISHED"


def test_paint_under_a_detected_object_box_is_ignored():
    img = synth()
    img[130:290, 340:550] = (250, 250, 250)                 # a white truck-like blob inside the truck box
    res, _ = B.localize(img, QBOX, [TRUCK, CONE], V02)
    assert all(not (340 <= c["center"][0] <= 550 and 130 <= c["center"][1] <= 290) for c in res["dash_components"])


def test_region_outside_qwen_box_fails_the_check():
    res, _ = B.localize(synth(), [400, 20, 780, 420], [TRUCK, CONE], V02)   # Qwen box that does not contain the painted L
    assert res["status"] == "FLOOR_MARKING_NOT_ESTABLISHED"


def _filled(ff):
    mask = np.zeros((H, W), np.uint8)
    for p in ff["polygons_xy"]:
        cv2.fillPoly(mask, [np.round(np.array(p)).astype(np.int32)], 1)
    return mask


def test_free_floor_is_smaller_than_bay_and_excludes_object_boxes():
    truck = {"object_id": "truck_1", "class_name": "truck", "bbox_xyxy": [300, 60, 500, 260]}   # crosses the bay's top edge
    res, _ = B.localize(synth(), QBOX, [truck], V02)
    assert res["status"] == "FLOOR_MARKING_ESTABLISHED"
    ff = res["free_floor"]["margin_0px"]
    assert ff["area_px"] < res["bay_area_px"] and not ff["hole_check"]["interior_hole_lost"]
    assert _filled(ff)[110:250, 310:490].sum() == 0           # the truck box is not placeable floor


def test_object_fully_inside_the_bay_is_flagged_not_silently_kept():
    """v0.2 placement_region keeps only external contours: an object standing inside the bay leaves no hole. The wrapper must say so."""
    truck = {"object_id": "truck_1", "class_name": "truck", "bbox_xyxy": [300, 200, 500, 360]}
    res, _ = B.localize(synth(), QBOX, [truck], V02)
    hc = res["free_floor"]["margin_0px"]["hole_check"]
    assert hc["interior_hole_lost"] is True
    assert hc["exact_free_px_after_object_boxes"] < res["bay_area_px"] - 20000   # the exact figure does remove the box


def test_checks_are_all_named_and_pass_when_established():
    res, _ = B.localize(synth(), QBOX, [TRUCK, CONE], V02)
    assert len(res["checks"]) >= 6 and all(c["status"] == "PASS" for c in res["checks"])


def test_module_does_not_read_answer_side_data():
    src = (HERE / "bay_localization.py").read_text(encoding="utf-8") + (HERE / "localize_bay.py").read_text(encoding="utf-8")
    for forbidden in ("semantic_reference_zones", "Z06", "reference_zone", "human_confirmed", "answer_region"):
        code_lines = [l for l in src.splitlines() if forbidden in l and not l.strip().startswith(("#", '"', "'")) and "not_used_as_input" not in l]
        assert not [l for l in code_lines if "import" in l or "open(" in l or "read_text" in l], (forbidden, code_lines)


def test_stored_E1_pin():
    """Observed behaviour on the stored E1 photo (thresholds were set by eye on this photo: not an accuracy figure)."""
    cfg = V.load_config()
    case = O.build_case(next(c for c in cfg["stored_cases"] if c["id"] == "E1"), HERE)
    img = cv2.imread(case["image_path"])
    res, _ = B.localize(img, case["candidates"][0]["box_px"], case["observations"]["objects"], V02)
    assert res["status"] == "FLOOR_MARKING_ESTABLISHED"
    assert sorted(l["n_dashes"] for l in res["lines"]) == [6, 11]
    assert np.allclose(res["bay_geometry"]["corner_xy"], [1426.7, 172.5], atol=1.0)
    assert 0.1 < res["bay_area_px"] / 161741 < 0.3            # the painted bay is a small part of Qwen's box
    hc = res["free_floor"]["margin_0px"]["hole_check"]
    assert not hc["interior_hole_lost"] and hc["exact_free_px_after_object_boxes"] == 8377
