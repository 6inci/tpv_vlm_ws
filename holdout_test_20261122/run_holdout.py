#!/usr/bin/env python3
"""독립 시험 실행 (한 번만). 실행 전 확인 -> 예전 버전 -> 새 버전 -> 채점.

  $PY -B run_holdout.py --check-only      확인만 (아무 결과도 만들지 않음)
  $PY -B run_holdout.py                   확인을 모두 통과하면 results/run_<시각>/ 에 실행 (GPU, 사진 1장에 약 100 s)
  $PY -B run_holdout.py --retry-reason "..."   끝나지 못한 실행이 있을 때만. 이유는 manifest에 남음

실행 전 확인 (하나라도 어기면 멈춤):
  1. 코드·설정·모델이 FREEZE.json과 같다 (freeze.py)
  2. 사진: *.png, 1920 x 1080, 이미 본 15장이 아님, 카메라가 기준 사진에서 3 px 안, 로봇 마커가 출발 자리 3 px 안
  3. 정답: 모든 사진에 있고, '검토 완료'와 시나리오가 있고, 사람이 '확정'했다
  4. 끝난 실행이 없다 (한 번만 실행)
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import freeze as FZ
import hold as H

PY = H.CFG["python"]
OLD = Path(H.CFG["systems"]["old"]["dir"])


def camera_shift_px(ref_gray, img_gray):
    """기준 사진 -> 이 사진 배경 특징점 정렬 (ORB + RANSAC 닮음 변환). 네 모서리·가운데의 최대 이동 px"""
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(ref_gray, None)
    k2, d2 = orb.detectAndCompute(img_gray, None)
    if d1 is None or d2 is None:
        return None, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(m) < 30:
        return None, len(m)
    a = np.float32([k1[x.queryIdx].pt for x in m])
    b = np.float32([k2[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or int(inl.sum()) < 30:
        return None, 0 if inl is None else int(inl.sum())
    h, w = ref_gray.shape
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]])
    moved = pts @ M[:, :2].T + M[:, 2]
    return float(np.linalg.norm(moved - pts, axis=1).max()), int(inl.sum())


def check_photos(files):
    pr = H.CFG["photo_rules"]
    seen = {H.sha256_file(p) for p in Path(pr["not_in"]).glob("*.png")}
    ref = cv2.imread(H.CFG["reference_photo"], cv2.IMREAD_GRAYSCALE)
    sys.path.insert(0, str(OLD))
    import nl_to_path as N                              # 읽기만: 출발 자리 마커 확인 (start_pose)
    geo, planner = N.Geo(), N.Planner()
    rows, bad = [], []
    for p in files:
        img = cv2.imread(str(p))
        sha = H.sha256_file(p)
        r = {"photo": p.name, "sha256": sha, "size": None if img is None else [img.shape[1], img.shape[0]]}
        if img is None:
            bad.append(f"{p.name}: 읽을 수 없음")
            rows.append(r)
            continue
        if r["size"] != pr["size"]:
            bad.append(f"{p.name}: 크기 {r['size']} (필요 {pr['size']})")
        if sha in seen:
            bad.append(f"{p.name}: 이미 본 2차 사진 15장 중 하나와 같음")
        shift, n = camera_shift_px(ref, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        r["camera_shift_px"], r["camera_inliers"] = None if shift is None else round(shift, 2), n
        if shift is None or shift > pr["camera_max_shift_px"]:
            bad.append(f"{p.name}: 카메라 어긋남 {r['camera_shift_px']} px (한도 {pr['camera_max_shift_px']}, 특징점 {n})")
        sp = N.start_pose(geo, planner, img)
        r["robot_marker_shift_px"], r["robot_status"] = sp["marker_shift_px"], sp["status"]
        if sp["status"] != "OK":
            bad.append(f"{p.name}: 로봇 출발 자리 확인 실패 ({sp['status']}, 마커 {sp['marker_shift_px']} px)")
        rows.append(r)
    if N.CFG["default_instruction"] != H.CFG["instruction"]:
        bad.append("예전 버전 기본 지시가 시험 지시와 다름")
    return rows, bad


def preflight():
    problems = []
    if not H.FREEZE_FILE.exists():
        problems.append("FREEZE.json 없음 (freeze.py --write)")
    else:
        d = FZ.diff(json.loads(H.FREEZE_FILE.read_text(encoding="utf-8")), FZ.collect())
        problems += [f"고정 뒤 바뀜: {x}" for x in d]
    files = H.photos()
    if not files:
        problems.append(f"사진 없음: {H.PHOTOS}")
        return [], problems
    rows, bad = check_photos(files)
    problems += bad
    problems += H.gt_problems(H.load_gt(), files)
    if H.finished_run():
        problems.append(f"이미 끝난 실행이 있음: {H.finished_run()} (한 번만 실행)")
    return rows, problems


def stage(name, run):
    """자식 프로세스에서 한 버전만 실행 (GPU 작업 프로세스를 버전마다 깨끗하게 띄우고 닫음)"""
    run = Path(run)
    if name == "old":
        sys.path.insert(0, str(OLD))
        import nl_to_path as N
        N.CFG["photo_set"]["dir"] = str(H.PHOTOS)        # --all 사진 목록에만 쓰이는 값 (README)
        (run / "old").mkdir(parents=True, exist_ok=True)
        (run / "old" / "OVERRIDE.json").write_text(json.dumps({"CFG.photo_set.dir": str(H.PHOTOS)}, ensure_ascii=False), encoding="utf-8")
        sys.argv = ["nl_to_path.py", "--all", "--out", str(run / "old")]
        N.main()
    else:
        import new_system as NS
        NS.extract_set("holdout", H.PHOTOS, run / "new")
        thr, recs = NS.decide_set("holdout", run / "new", gt=H.load_gt())
        NS.finish("holdout", thr, recs, run / "new")
    (run / f"STAGE_{name}_DONE").write_text(datetime.now().isoformat(timespec="seconds"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--retry-reason")
    ap.add_argument("--stage", choices=["old", "new"], help=argparse.SUPPRESS)
    ap.add_argument("--run", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.stage:
        stage(a.stage, a.run)
        return
    rows, problems = preflight()
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    if problems:
        print("\n실행할 수 없습니다:\n  " + "\n  ".join(problems))
        sys.exit(1)
    unfinished = [str(d) for d in H.run_dirs() if not (d / "RUN_DONE").exists()]
    if a.check_only:
        print(f"\n확인 통과: 사진 {len(rows)}장. 실행하려면 --check-only 없이 다시 실행")
        return
    if unfinished and not a.retry_reason:
        raise SystemExit(f"끝나지 못한 실행이 있습니다 {unfinished}. 다시 하려면 --retry-reason '이유'")
    run = H.RESULTS / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    man = {"experiment_id": H.CFG["experiment_id"], "run_id": run.name, "started": datetime.now().isoformat(timespec="seconds"),
           "freeze_sha256": H.sha256_file(H.FREEZE_FILE), "gt_sha256": H.sha256_file(H.GT_FILE), "photos": rows,
           "config_snapshot": H.CFG, "python": sys.version, "platform": platform.platform(), "gpu": gpu,
           "retry_reason": a.retry_reason, "unfinished_before": unfinished}
    (run / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    for name in ("old", "new"):
        t0 = time.time()
        with open(run / f"{name}.log", "w", encoding="utf-8") as log:
            rc = subprocess.run([PY, "-B", str(H.HERE / "run_holdout.py"), "--stage", name, "--run", str(run)], cwd=H.HERE, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{name}: 종료 코드 {rc}, {time.time() - t0:.0f} s (로그 {run / f'{name}.log'})", flush=True)
        if rc != 0 or not (run / f"STAGE_{name}_DONE").exists():
            raise SystemExit(f"{name} 단계 실패. 결과는 남겨 두었습니다: {run}")
    (run / "RUN_DONE").write_text(datetime.now().isoformat(timespec="seconds"))
    subprocess.run([PY, "-B", str(H.HERE / "score.py"), "--run", str(run)], cwd=H.HERE, check=True)


if __name__ == "__main__":
    main()
