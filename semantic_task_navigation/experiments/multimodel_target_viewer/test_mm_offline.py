#!/usr/bin/env python3
"""Offline checks of the multi-model viewer (no model call, no GPU):
  /home/binci/miniconda3/envs/stnav_gui/bin/python experiments/multimodel_target_viewer/test_mm_offline.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mm_pipeline as M  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def test_pure():
    rc = CFG["reference_classes"]
    assert M.ref_class("robot", rc) == "robot" and M.ref_class("mobile robot", rc) == "robot" and M.ref_class("robot arm", rc) is None
    assert M.ref_class("conveyor belt", rc) == "conveyor" and M.ref_class("truck bed", rc) == "truck" and M.ref_class(None, rc) is None
    assert M.as_label("red cube") == "a red cube" and M.as_label("orange object") == "an orange object" and M.as_label("a pallet") == "a pallet"
    r = M.square_crop([10, 10, 20, 30], (1920, 1080), 0.1, 96)
    assert r[2] - r[0] == r[3] - r[1] == 96 and r[0] >= 0 and r[1] >= 0
    r = M.square_crop([1900, 1060, 1915, 1075], (1920, 1080), 1.0, 96)
    assert r[2] <= 1920 and r[3] <= 1080 and r[2] - r[0] == 96
    e = np.eye(3)
    rk = M.siglip_rank(e[:2], e, 10.0, -5.0, ["x", "y", "z"])
    assert abs(sum(s for _, s in rk["combined"]) - 1.0) < 1e-3 and {rk["combined"][0][0], rk["combined"][1][0]} == {"x", "y"}
    assert M.all_duplicates([[0, 0, 10, 10], [0, 0, 10, 11]], 0.7) and not M.all_duplicates([[0, 0, 10, 10], [50, 50, 60, 60]], 0.7)
    fw = CFG["checks"]["facility_words"]
    assert M.mentioned_facilities("로봇 주변에 있는 물건", fw) == ["robot"] and M.mentioned_facilities("로봇 팔 옆의 큐브", fw) == []
    assert M.mentioned_facilities("트럭 주변의 큐브를 컨베이어 옆으로", fw) == ["conveyor", "truck"]
    assert M.first_failure([{"name": "a", "status": "OK"}, {"name": "b", "status": "REUSED"}, {"name": "c", "status": "UNCERTAIN"}]) == "c"


def test_reply_checks():
    good = {"candidates": [{"id": k, "appearance": [f"thing {k}"], "role": "other"} for k in range(1, 7)], "object_to_move_id": None}
    assert M.check_reply(good, 6, 5, True) == (True, [])
    assert not M.check_reply(None, 6, 5, True)[0]
    out = M.check_reply({"candidates": good["candidates"] + [{"id": 9, "role": "other"}], "object_to_move_id": None}, 6, 5, True)
    assert not out[0] and any(p.startswith("MAPPING_ERROR") for p in out[1])
    part = {"candidates": good["candidates"][:5], "object_to_move_id": None}
    assert M.check_reply(part, 6, 5, True)[0] is False and M.check_reply(part, 6, 5, False)[0] is True      # incomplete: fatal for the source only
    # regression: the preserved F001 reply (20 candidates all 'green box / suitable / empty') must be rejected
    f001 = M.VP.HERE / "failure_records/F001_destination_roles_all_green_box/response_raw.txt"
    reply, err = M.P.parse_reply(f001.read_text(encoding="utf-8"), M.S.DEST_SCHEMA)
    ok, probs = M.check_reply(reply, 20, CFG["checks"]["degenerate_min_candidates"], False)
    assert err is None and not ok and any(p.startswith("DEGENERATE_RESPONSE") for p in probs), probs


def test_no_answers_and_old_program_untouched():
    ref = yaml.safe_load((M.TC.HERE / "evaluation_reference.yaml").read_text(encoding="utf-8"))
    boxes = [o["bbox_xyxy_px"] for o in ref["E1_objects"].values()]
    for f in ("mm_pipeline.py", "mm_app.py", "mm_worker.py", "config.yaml"):
        t = (HERE / f).read_text(encoding="utf-8")
        assert "evaluation_reference" not in t, f
        for b in boxes:
            assert str(b) not in t and ", ".join(map(str, b)) not in t, (f, b)
    before = (HERE / "baseline_check/instruction_target_viewer_BEFORE.sha256").read_text().split("\n")
    changed = []
    for line in filter(None, before):
        h, p = line.split("  ", 1)
        q = HERE.parent / p
        if not q.exists() or hashlib.sha256(q.read_bytes()).hexdigest() != h:
            changed.append(p)
    assert not changed, f"instruction_target_viewer files changed: {changed}"


def test_narrative_and_draw():
    img = np.zeros((1080, 1920, 3), np.uint8)
    cand = lambda i, b, top: {"id": i, "bbox_xyxy": b, "siglip_top3": [(top, 0.5), ("x", 0.2), ("y", 0.1)]}   # noqa: E731
    refs = {"source": {"name": "conveyor", "relation": "next_to", "box": [200, 200, 400, 700], "region": {"type": "vicinity", "rect": [80, 80, 600, 900]}},
            "destination": {"name": "truck", "relation": "near", "box": [1400, 0, 1700, 200], "region": {"type": "vicinity", "rect": [1300, 0, 1900, 300]}}}
    base = {"run_id": "run_001_X", "rules": "R1", "first_failure": None, "instruction": "t", "photo": {"photo_id": "X", "path": "/x"}, "references": refs,
            "vlm_backend": {"key": "k", "label": "VLM"},
            "interpretation": {"parsed": {"source_object": {"name_en": "red cube"}, "destination_meaning": "shipping", "source_relation": "next_to", "destination_relation": "near"}}}
    ok = {**base, "source": {"status": "SELECTED", "ids": ["c1"], "boxes": [[420, 410, 460, 450]], "candidates": [cand("c1", [420, 410, 460, 450], "a red cube")],
                             "ratings": {"c1": {"appearance": ["red cube"]}}},
          "destination": {"status": "SELECTED", "id": "m5", "bbox_xyxy": [1590, 250, 1640, 290], "kind": "DIRECT", "goal_kind": "OBJECT", "placement": "VLM_SUITABLE_EMPTY",
                          "load_state": "empty", "candidates": [cand("m5", [1590, 250, 1640, 290], "a small pallet")], "ratings": {"m5": {"appearance": ["support"]}},
                          "accessibility": {"m5": {"accessible": True, "gap_to_drivable_cm": 1.0, "approach_goal_image_px": [1600, 330]}}}}
    n = M.narrative(ok)
    assert n[-1].startswith("계획: (1)") and "주행 경로가 아닙니다" in n[-1]
    bad = {**base, "source": {"status": "TYPE_UNCERTAIN", "ids": ["c1"], "boxes": [[420, 410, 460, 450]], "candidates": [cand("c1", [420, 410, 460, 450], "an orange traffic cone")],
                              "ratings": {"c1": {"appearance": ["red cube"]}}},
           "destination": {"status": "MODEL_RESPONSE_INVALID", "why": "image 2: DEGENERATE_RESPONSE", "candidates": []}}
    n = M.narrative(bad)
    assert "확정하지 않았습니다" in n[1] and n[-1].startswith("계획 미확정") and "모델 응답 오류" in n[2]
    for r in (ok, bad):
        for c in (False, True):
            assert M.draw(img, r, c).shape == img.shape


if __name__ == "__main__":
    for f in (test_pure, test_reply_checks, test_no_answers_and_old_program_untouched, test_narrative_and_draw):
        f()
        print("ok", f.__name__)
