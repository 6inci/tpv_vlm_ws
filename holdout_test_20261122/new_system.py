"""새 버전(context_policy_20261122)을 새 사진 묶음에 그대로 돌린다. 그 폴더의 파일은 바꾸지 않고, 쓰는 파일은 모두 out_root 안이다.

바꾸는 것은 실행 중인 프로세스의 값 세 가지뿐이다 (README '새 버전을 부르는 방법'):
- common.C['photo_sets']['holdout'] = 새 사진 폴더와 2차 사진 자세 (extract.main, common.Cam이 읽음)
- extract.HERE, policy.HERE = out_root (특징·결정 파일을 out_root/data/holdout에 씀)
- policy.gt_for = 이 시험의 정답 (policy가 남기는 참고용 grade에만 쓰임. 결정에는 쓰이지 않음. 채점은 score.py)
결정 함수(score_photo, evaluate_candidates, decide)와 기준값(data/dev/calibration.json)은 그대로 쓴다.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import hold as H

NEW = Path(H.CFG["systems"]["new"]["dir"])
OLD = Path(H.CFG["systems"]["old"]["dir"])
sys.path.insert(0, str(NEW))
import common as CM  # noqa: E402  (읽기만)


def _gt_in_policy_format(gt):
    if gt is None:
        return {}
    out = {}
    for stem, e in gt["photos"].items():
        out[stem] = {"cube": e["cube"][0] if e["cube"] else None, "empty_pallet": e["empty_pallet"][0] if e["empty_pallet"] else None,
                     "empty_pallet_all": e["empty_pallet"]}
    return out


def register_set(name, photos_dir):
    CM.C["photo_sets"][name] = {"dir": str(photos_dir), "pose": H.CFG["pose_file"]}


def extract_set(name, photos_dir, out_root):
    """GPU. extract.main()을 그대로 부르고 출력만 out_root/data/<name>"""
    register_set(name, photos_dir)
    import extract as EX
    EX.HERE = Path(out_root)
    argv = sys.argv
    sys.argv = ["extract.py", "--set", name]
    try:
        EX.main()
    finally:
        sys.argv = argv


def decide_set(name, data_root, gt=None, photos_dir=None):
    """모델 없음. policy.run_set을 그대로 부른다. data_root/data/<name>의 특징을 읽음"""
    if photos_dir is not None:
        register_set(name, photos_dir)
    import policy as PO
    PO.HERE = Path(data_root)
    if gt is not None:
        PO.gt_for = lambda _name, _g=_gt_in_policy_format(gt): _g
    thr = json.loads(Path(H.CFG["systems"]["new"]["thresholds_file"]).read_text(encoding="utf-8"))["thresholds"]
    meta, recs = PO.run_set(name, thr)
    return thr, recs


def finish(name, thr, recs, od):
    """policy.main()의 그림·경로·저장 부분과 같은 순서 (--paths 포함). od = 결과 폴더"""
    import policy as PO
    od = Path(od)
    od.mkdir(parents=True, exist_ok=True)
    (od / "config_snapshot.yaml").write_text((NEW / "config" / "policy.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    cam = CM.Cam(name)
    summ = {"A": {}, "B": {}}
    for r in recs:
        PO.draw_photo(cam, r, od / f"{Path(r['photo']).stem}_policy.jpg")
        for k in ("A", "B"):
            summ[k][r[k]["grade"]] = summ[k].get(r[k]["grade"], 0) + 1
    sys.path.insert(0, str(OLD))
    import nl_to_path as N                               # 읽기만 (좌표·정차·경로·그림 함수 재사용, policy.main과 같음)
    geo, planner = N.Geo(), N.Planner()
    ins = N.CFG["default_instruction"]
    for r in recs:
        if r["A"]["status"] == "SELECTED" and r["B"]["status"] == "SELECTED":
            pr = N.run_one(Path(r["photo"]), ins, None, geo, planner, od / "paths", ab_px=[r["A"]["box"], r["B"]["box"]])
            pj = Path(pr["files"]["record"])
            j = json.loads(pj.read_text(encoding="utf-8"))
            j["perception"]["note"] = "A·B = context_policy_20261122 policy.py 결정 (모델 출력은 extract.py, 정책은 Python)"
            j["policy_decision"] = {k: {kk: r[k][kk] for kk in ("status", "why", "id", "box", "s1", "s2", "grade")} for k in ("A", "B")}
            pj.write_text(json.dumps(j, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            r["path_result"] = pr["result"]
    (od / "decisions.json").write_text(json.dumps({"set": name, "thresholds": thr, "summary": summ, "photos": recs,
                                                   "created": datetime.now().isoformat(timespec="seconds")},
                                                  indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return od / "decisions.json"
