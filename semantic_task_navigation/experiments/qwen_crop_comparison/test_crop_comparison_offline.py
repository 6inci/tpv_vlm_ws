"""Offline checks (no GPU, no model). From the project root:
  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/qwen_crop_comparison
(1) crop helpers and the declared decision rule on synthetic input; (2) pins of the latest live run (what it recorded; not evidence of accuracy)."""
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
import compare as C  # noqa: E402
from semantic_task_nav.perception.grounding import make_verify_crop  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
TCFG = C.load_config()["models"]["target_inspection"]


def test_own_crop_function_reproduces_v02_and_marker_is_the_only_difference(tmp_path):
    img = np.random.RandomState(0).randint(0, 255, (300, 400, 3), np.uint8)
    box = [150.4, 120.2, 170.9, 140.7]
    ref = make_verify_crop(img, box, TCFG, tmp_path / "ref.png")
    own = C.make_crop(img, box, TCFG, TCFG["verify_crop_pad_ratio"], None, True, tmp_path / "own.png")
    assert np.array_equal(cv2.imread(ref["path"]), cv2.imread(own["path"])) and own["crop_xyxy_raw"] == ref["crop_xyxy_raw"]
    nomark = C.make_crop(img, box, TCFG, TCFG["verify_crop_pad_ratio"], own["scale"], False, tmp_path / "b.png")
    diff = np.any(cv2.imread(nomark["path"]) != cv2.imread(own["path"]), axis=2)
    assert diff.any() and nomark["crop_xyxy_raw"] == own["crop_xyxy_raw"]
    ctx = C.make_crop(img, box, TCFG, CFG["conditions"]["C"]["pad_ratio"], own["scale"], True, tmp_path / "c.png")
    assert ctx["scale"] == own["scale"] and ctx["size_wh"][0] > own["size_wh"][0] and ctx["crop_xyxy_raw"][0] < own["crop_xyxy_raw"][0]


def a(cube, c1, c2, c3):
    return {"cube": cube, "c1": c1, "c2": c2, "c3": c3}


@pytest.mark.parametrize("A,B,C_,label_c,overall", [
    (a("yes", "no", "no", "no"), a("yes", "no", "no", "no"), a("yes", "no", "no", "no"), "CUBE_ONLY", "SITUATION_1"),
    (a("yes", "yes", "no", "no"), a("yes", "no", "no", "no"), a("yes", "no", "no", "no"), "CUBE_ONLY", "SITUATION_2"),
    (a("no", "no", "no", "no"), a("no", "no", "no", "no"), a("uncertain", "no", "no", "no"), "NONE_ACCEPTED", "SITUATION_3"),
    (a("no", "no", "no", "no"), a("yes", "no", "no", "no"), a("no", "no", "no", "yes"), "CONFUSABLE_ONLY", "SITUATION_4"),
])
def test_declared_decision_rule(A, B, C_, label_c, overall):
    per, ov = C.classify("cube", ["c1", "c2", "c3"], {"A": A, "B": B, "C": C_})
    assert ov.startswith(overall) and per["C"]["label"] == label_c


def test_frozen_prompt_is_unchanged():
    m = C.load_config()["models"]["prompts"]["verify_object"]
    assert m["id"] == CFG["prompt"]["id"] and C.sha256_text(m["text"].replace("{description}", "a small red cube")) == CFG["prompt"]["sha256"]


def latest():
    runs = sorted((HERE / CFG["results_dir"]).glob("*_crop_comparison"))
    assert runs, "run compare.py first"
    return runs[-1], json.loads((runs[-1] / "result.json").read_text(encoding="utf-8"))


def test_latest_run_pins():
    run, r = latest()
    assert r["claims"]["detection_accuracy"] == "NOT_COMPUTED" and r["label_status"] == "AI_PRELIMINARY_REVIEW"
    assert r["v02_code_before"] == r["v02_code_after"] and r["v02_code_before"].startswith("254cfb31da7f")
    assert r["pipeline_control"]["all_raw_responses_equal_stored"] and r["offline_checks_ok"]
    assert r["prompt"]["sha256"] == CFG["prompt"]["sha256"] and r["model"]["load"]["model_id"] == "Qwen/Qwen3-VL-4B-Instruct"
    cs = r["candidates"]
    assert len(cs) == 4 and [c["id"] for c in cs] == ["target_cand1", "target_cand2", "target_cand5", "target_cand3"]
    for c in cs:
        assert c["stored_A"]["raw_response"] and c["A_control"]["call"]["prompt_sha256"] == c["B"]["call"]["prompt_sha256"] == c["C"]["call"]["prompt_sha256"]
        assert c["crops"]["B"]["differs_from_A_only_at_marker_pixels"] and c["crops"]["C"]["candidate_scale_equals_A"]
        for k in ("A_control", "B", "C"):
            assert c[k]["call"]["raw_response"] and c[k]["call"]["error"] is None
    assert r["decision_rule_applied"]["across_conditions"].startswith("SITUATION_4")
    for f in ("01_comparison.png", "comparison_table.md", "result.json", "manifest.json", "config_snapshot.yaml"):
        assert (run / f).exists()
