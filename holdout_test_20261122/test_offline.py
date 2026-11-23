"""독립 시험 도구의 오프라인 검사 (모델·GPU 없음).
  /home/binci/miniconda3/envs/stnav_gui/bin/python -B -m pytest -q -p no:cacheprovider test_offline.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

import hold as H
import score as S

OLD_RUN = Path("/home/binci/vlm_ws/nl_to_route_20261122/results/run_20261122_all15_final")
NEW_DEC = Path("/home/binci/vlm_ws/context_policy_20261122/results/test_20261122_183736/decisions.json")
GT15 = Path("/home/binci/vlm_ws/context_policy_20261122/data/gt_test.json")


def test_score_reproduces_the_15_photo_table():
    """context_policy README 표: 11-22 통합 A 13/1/1, B 11/1/3, 경로 9 · 맥락 정책 최종 A 14/0/1, B 12/1/2, 경로 11"""
    gt = S.normalize_gt(json.loads(GT15.read_text(encoding="utf-8")))
    _, tot = S.score(gt, {"old": S.old_decisions(OLD_RUN), "new": S.new_decisions(NEW_DEC)})
    t = tot["all"]
    assert t["n_photos"] == 15
    assert [t["old"]["A"][g] for g in ("맞음", "틀리게 확정", "놓침(보류)", "맞는 보류")] == [13, 1, 1, 0]
    assert [t["old"]["B"][g] for g in ("맞음", "맞는 보류", "놓침(보류)", "틀리게 확정")] == [11, 1, 3, 0]
    assert [t["new"]["A"][g] for g in ("맞음", "틀리게 확정", "놓침(보류)", "맞는 보류")] == [14, 0, 1, 0]
    assert [t["new"]["B"][g] for g in ("맞음", "맞는 보류", "놓침(보류)", "틀리게 확정")] == [12, 1, 2, 0]
    assert (t["old"]["paths"], t["new"]["paths"]) == (9, 11)


def test_decision_rule():
    base = {g: 0 for g in S.GRADES}
    def tot(w_old, c_old, w_new, c_new):
        mk = lambda w, c: {"A": base, "B": base, "paths": 0, "wrong_commits": w, "correct": c}
        return {"primary": {"n_photos": 16, "old": mk(w_old, c_old), "new": mk(w_new, c_new)}}
    assert S.decide(tot(1, 28, 0, 28))["verdict"].startswith("새 버전")
    assert S.decide(tot(1, 28, 1, 29))["verdict"].startswith("새 버전")
    assert S.decide(tot(0, 28, 1, 31))["verdict"].startswith("예전")      # 잘못 확정이 늘면 맞음이 많아도 유지
    assert S.decide(tot(1, 28, 0, 27))["verdict"].startswith("예전")


def test_new_system_wrapper_reproduces_stored_decisions():
    """연결 모듈이 policy.run_set을 그대로 부르는지: 저장된 15장 특징으로 다시 결정하면 저장된 결정과 같아야 함"""
    import new_system as NS
    _, recs = NS.decide_set("test", NS.NEW)
    stored = {Path(r["photo"]).stem: r for r in json.loads(NEW_DEC.read_text(encoding="utf-8"))["photos"]}
    assert len(recs) == len(stored) == 15
    for r in recs:
        s = stored[Path(r["photo"]).stem]
        for k in ("A", "B"):
            assert (r[k]["status"], r[k]["id"], r[k]["box"]) == (s[k]["status"], s[k]["id"], s[k]["box"])


def test_gt_checks_block_unfinished_answers(tmp_path):
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), np.zeros((4, 4, 3), np.uint8))
    ok = {"confirmed_by": "사람", "confirmed_at": "t", "photos": {"x": {"sha256": H.sha256_file(p), "scenario": "P1", "reviewed": True,
                                                                         "cube": [[1.0, 2.0]], "empty_pallet": []}}}
    assert H.gt_problems(ok, [p]) == []
    for change in ({"confirmed_by": None}, ):
        assert H.gt_problems({**ok, **change}, [p])
    for change in ({"reviewed": False}, {"scenario": "P9"}, {"sha256": "0"}, {"cube": [1, 2]}):
        bad = json.loads(json.dumps(ok))
        bad["photos"]["x"].update(change)
        assert H.gt_problems(bad, [p]), change
    assert H.gt_problems(ok, [p, tmp_path / "y.png"])            # 정답 없는 사진


def test_gt_tool_save_confirm_lock(tmp_path, monkeypatch):
    import gt_tool as T
    monkeypatch.setattr(H, "PHOTOS", tmp_path / "photos")
    monkeypatch.setattr(H, "GT_FILE", tmp_path / "gt.json")
    monkeypatch.setattr(H, "RESULTS", tmp_path / "results")
    (tmp_path / "photos").mkdir()
    cv2.imwrite(str(tmp_path / "photos" / "a.png"), np.zeros((4, 4, 3), np.uint8))
    draft = {"photos": {"a": {"cube": [[10, 20]], "empty_pallet": [], "scenario": "P3", "reviewed": False}}}
    assert T.save(draft)[0] == 200                                   # 저장은 되지만
    code, body = T.save(draft, confirm_name="사람")
    assert code == 400 and body["problems"]                         # 검토 완료가 없으면 확정 안 됨
    draft["photos"]["a"]["reviewed"] = True
    assert T.save(draft, confirm_name="사람")[0] == 200
    assert H.load_gt()["confirmed_by"] == "사람" and H.gt_problems(H.load_gt(), H.photos()) == []
    assert T.save(draft)[0] == 409                                  # 확정 뒤에는 고칠 수 없음


def test_camera_shift():
    import run_holdout as R
    ref = cv2.imread(H.CFG["reference_photo"], cv2.IMREAD_GRAYSCALE)
    same, n = R.camera_shift_px(ref, ref)
    assert same is not None and same < 0.5 and n > 100
    moved = cv2.warpAffine(ref, np.float32([[1, 0, 5], [0, 1, 0]]), (ref.shape[1], ref.shape[0]))
    s, _ = R.camera_shift_px(ref, moved)
    assert 4.0 < s < 6.0
