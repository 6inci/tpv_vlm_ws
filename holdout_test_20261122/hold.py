"""독립 시험 공통: 설정, 경로, sha256, 정답 파일 읽기와 검사, 실행 상태 (config/holdout.yaml)."""
import hashlib
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config" / "holdout.yaml").read_text(encoding="utf-8"))
PHOTOS = HERE / CFG["photos_dir"]
GT_FILE = HERE / CFG["ground_truth"]["file"]
FREEZE_FILE = HERE / "FREEZE.json"
RESULTS = HERE / "results"
SCENARIOS = {**CFG["scenarios"]["primary"], **CFG["scenarios"]["edge"]}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def photos():
    return sorted(PHOTOS.glob("*.png"))


def load_gt(path=None):
    path = Path(path or GT_FILE)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def gt_problems(gt, photo_files):
    """정답 파일이 실행에 쓸 수 있는 상태인지. 문제 목록 (빈 목록 = 통과)"""
    if gt is None:
        return ["정답 파일 없음 (gt_tool.py로 만든다)"]
    out = []
    if not gt.get("confirmed_by") or not gt.get("confirmed_at"):
        out.append("정답이 아직 '확정'되지 않음")
    have = gt.get("photos", {})
    for p in photo_files:
        e = have.get(p.stem)
        if e is None:
            out.append(f"{p.name}: 정답 없음")
            continue
        if e.get("sha256") != sha256_file(p):
            out.append(f"{p.name}: 정답을 찍은 뒤 사진 파일이 바뀜 (sha256 다름)")
        if e.get("reviewed") is not True:
            out.append(f"{p.name}: '검토 완료' 표시 없음 (없음 [] 과 아직 안 찍음을 구분하려고 필요)")
        if e.get("scenario") not in SCENARIOS:
            out.append(f"{p.name}: 시나리오 태그 없음 또는 모름 ({e.get('scenario')})")
        for k in ("cube", "empty_pallet"):
            v = e.get(k)
            if not isinstance(v, list) or any(not (isinstance(q, list) and len(q) == 2) for q in v):
                out.append(f"{p.name}: {k}는 [[u, v], ...] 목록이어야 함 (없으면 [])")
    extra = sorted(set(have) - {p.stem for p in photo_files})
    if extra:
        out.append(f"사진 없이 정답만 있음: {extra}")
    return out


def run_dirs():
    return sorted(RESULTS.glob("run_*")) if RESULTS.exists() else []


def finished_run():
    return next((d for d in run_dirs() if (d / "RUN_DONE").exists()), None)
