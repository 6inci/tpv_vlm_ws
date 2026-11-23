"""Test E (ROI zoom + exact back-transform) and Test K (a red blob is never a target by itself)."""
import copy

import cv2
import numpy as np
import pytest

from semantic_task_nav.config import load_config
from semantic_task_nav.contracts import OBSERVED, RAW_PX
from semantic_task_nav.geometry.coordinate_frames import RoiTransform, cell_to_px, px_to_cell
from semantic_task_nav.perception import target_inspection as ti

CUBE = [1000, 500, 1012, 512]          # red square (cube-like)
CONE = [1060, 470, 1076, 492]          # red triangle (cone-like confuser)
FAR_CUBE = [200, 900, 212, 912]        # red square far from the conveyor
CONVEYOR = {"object_id": "conveyor_1", "class_name": "conveyor", "bbox_xyxy": [900, 300, 980, 700],
            "knowledge": OBSERVED, "coordinate_frame": RAW_PX}


def scene(with_far=False):
    img = np.full((1080, 1920, 3), 90, np.uint8)
    cv2.rectangle(img, tuple(CUBE[:2]), (CUBE[2] - 1, CUBE[3] - 1), (0, 0, 220), -1)
    cv2.fillPoly(img, [np.array([[1068, 470], [1060, 491], [1076, 491]])], (0, 0, 220))
    if with_far:
        cv2.rectangle(img, tuple(FAR_CUBE[:2]), (FAR_CUBE[2] - 1, FAR_CUBE[3] - 1), (0, 0, 220), -1)
    return img


def mcfg():
    m = copy.deepcopy(load_config()["models"])
    m["grounding_dino"]["roi_detector"] = False
    return m


def test_roi_transform_roundtrip():
    roi = RoiTransform([123, 45, 623, 445], 2.37)
    for x, y in [(123, 45), (400.5, 300.25), (622.9, 444.9)]:
        rx, ry = roi.raw_to_roi_point(x, y)
        assert roi.roi_to_raw_point(rx, ry) == pytest.approx((x, y), abs=1e-9)
    assert roi.zoom_size == (round(500 * 2.37), round(400 * 2.37))


def test_grid_cell_mapping():
    assert px_to_cell(*cell_to_px(7, 11, 8), 8) == (7, 11)


def test_isotropic_roi_margin():
    roi = ti.make_roi(CONVEYOR["bbox_xyxy"], (1920, 1080), mcfg()["target_inspection"])
    x0, y0, x1, y1 = roi.crop_xyxy
    margin = 0.6 * 400                         # ratio * long side (400 px) > min margin
    assert 900 - x0 == margin and 300 - y0 == margin and x1 - 980 == margin


def test_hsv_candidate_maps_back_to_raw_pixels():
    img = scene()
    t = mcfg()["target_inspection"]
    roi = ti.make_roi(CONVEYOR["bbox_xyxy"], (1920, 1080), t)
    assert roi.scale > 1.0
    cands = ti.hsv_candidates(ti.zoom_roi(img, roi), roi, t)
    best = min(cands, key=lambda c: abs(c["bbox_xyxy"][0] - CUBE[0]) + abs(c["bbox_xyxy"][1] - CUBE[1]))
    assert best["bbox_xyxy"] == pytest.approx(CUBE, abs=1.0)
    assert best["bbox_roi"]["frame"] == "roi_local_px"


def _fake_verify(yes_boxes):
    def fake(client, mcfg, image, box, description, crop_path):
        hit = any(abs(box[0] - b[0]) < 3 and abs(box[1] - b[1]) < 3 for b in yes_boxes)
        return {"status": "yes" if hit else "no", "answer": {"answer": "yes" if hit else "no"},
                "call": {"model": "fake", "commit_hash": None}}
    return fake


@pytest.mark.parametrize("yes,expect", [([CUBE], "TARGET_SELECTED"), ([CUBE, CONE], "TARGET_AMBIGUOUS"),
                                        ([], "TARGET_NOT_FOUND")])
def test_verification_decides(monkeypatch, tmp_path, yes, expect):
    monkeypatch.setattr(ti, "verify_box", _fake_verify(yes))
    res = ti.inspect_target(object(), mcfg(), load_config()["planning"], scene(), 1,
                            {"color": "red", "class": "cube", "source_reference": "conveyor"}, [CONVEYOR], [], tmp_path)
    assert res["status"] == expect
    if expect == "TARGET_SELECTED":
        assert res["target"]["bbox_xyxy"] == pytest.approx(CUBE, abs=1.0)
        assert res["target"]["verification"]["status"] == "yes"
    else:
        assert res["target"] is None


def test_relation_disambiguates_near_conveyor(monkeypatch, tmp_path):
    monkeypatch.setattr(ti, "verify_box", _fake_verify([CUBE, FAR_CUBE]))
    res = ti.inspect_target(object(), mcfg(), load_config()["planning"], scene(with_far=True), 1,
                            {"color": "red", "class": "cube", "source_reference": "conveyor"}, [], [], tmp_path)
    assert res["status"] == "TARGET_AMBIGUOUS"          # no conveyor -> cannot use the relation
    res = ti.inspect_target(object(), mcfg(), load_config()["planning"], scene(with_far=True), 1,
                            {"color": "red", "class": "cube", "source_reference": "conveyor"}, [CONVEYOR], [], tmp_path)
    assert res["status"] == "TARGET_SELECTED"           # far cube is outside the conveyor ROI


def test_red_blob_without_model_is_not_a_target(tmp_path):
    res = ti.inspect_target(None, mcfg(), load_config()["planning"], scene(), 1,
                            {"color": "red", "class": "cube", "source_reference": "conveyor"}, [CONVEYOR], [], tmp_path,
                            model_mode=False)
    assert res["n_candidates"] >= 1 and res["target"] is None and res["status"] == "NOT_RUN_IMAGE_DEMO"
