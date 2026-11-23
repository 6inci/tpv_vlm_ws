#!/usr/bin/env python3
"""사진 픽셀 <-> 지도 env2_20261119 좌표 변환 (2026-11-20, 메카넘 로봇 주행 쪽에서 만듦)

필요: Python 3.8+, numpy, opencv-python (cv2)
이 파일은 1_required/ 폴더 안에서 그 폴더의 파일만 읽는다.

좌표 규약
- 지도 좌표 (x, y, z), 단위 m. 원점 = 카메라 쪽 왼쪽 매트 바깥 모서리 근처,
  +x = 카메라 쪽 노란 선 A -> B 방향(사진 오른쪽), +y = 카메라에서 멀어지는 쪽, z = 위.
- 카메라 자세(rvec, tvec)는 'Camera World' 기준이다. Camera World = (-x, y, -z)  (지도 x 부호 반대, 아래쪽이 +Z).
  OpenCV 식: 카메라 좌표 = R(rvec) @ CameraWorld + tvec
- 원본 사진(렌즈 왜곡 있음)은 왜곡 계수(dist)를 함께 쓰고, 왜곡을 편 사진(newK = K)은 dist 없이 쓴다.

명령 예
  python3 pixel_to_map.py --selftest                                   자체 시험
  python3 pixel_to_map.py --pose register-raw 793.17 621.81            등록 사진 '원본' 픽셀 -> 지도
  python3 pixel_to_map.py --pose register-undist 800 620               등록 사진 '편 사진' 픽셀 -> 지도
  python3 pixel_to_map.py --pose photo5 1712.0 705.6                   팀원 사진 5calibration.png(편 사진) 픽셀 -> 지도
  python3 pixel_to_map.py --pose photo5 --to-pixel 2.678 0.4822        지도 -> 픽셀
  python3 pixel_to_map.py --pose photo5 --z 0.03 1563 875              바닥에서 0.03 m 높이 평면으로 투영 (물체 윗면 등)
  python3 pixel_to_map.py --fit my_points.json --out my_pose.json      새 사진: 기준점 픽셀로 카메라 자세 구하기

--fit 입력 형식 (기준점 이름은 map/map_reference_points.json 과 같음, 대소문자 그대로, 4개 이상, 6개 이상 권장):
  {"undistorted": true, "points": {"C": [u, v], "D": [u, v], "E1": [u, v], ...}}
  원본(왜곡 있는) 사진의 픽셀이면 "undistorted": false.
  B 는 지도 값이 실제 테이프 모서리와 약 3.7 cm 달라서 코드가 자동으로 자세 계산에서 뺀다 (바닥 오차는 보여 줌).
  모르는 이름이 있으면 멈춘다. 점이 6개보다 적거나 거의 한 줄이면 경고한다. 카메라 높이가 1.8~2.6 m 밖이면 멈춘다
  (우리 카메라는 바닥 위 약 2.2 m 고정. 점 자리나 이름이 틀렸을 가능성이 크다).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CAL = HERE / "camera" / "calibration_result.json"
REF = HERE / "map" / "map_reference_points.json"
REG = HERE / "camera" / "register_20261119_194739"
P5 = HERE / "camera" / "photo5_pose.json"
SKIP_FOR_FIT = ("B",)
HEIGHT_OK_M = (1.8, 2.6)           # 카메라 높이 허용 범위 (등록 때 2.216 m)
MIN_POINTS_ADVISED = 6


def intrinsics():
    c = json.loads(CAL.read_text(encoding="utf-8"))
    return np.array(c["camera_matrix"], float), np.array(c["distortion_coefficients_k1_k2_p1_p2_k3"], float)


def ref_points():
    r = json.loads(REF.read_text(encoding="utf-8"))["points"]
    return {n: np.array(v["map_m"], float) for n, v in r.items()}


def load_pose(name):
    """'register-raw' | 'register-undist' | 'photo5' | 자세 json 경로 -> (rvec, tvec, dist 또는 None)"""
    K, dist = intrinsics()
    if name in ("register-raw", "register-undist"):
        e = np.load(REG / "camera_extrinsic.npz")
        return e["rvec"].reshape(3, 1), e["tvec"].reshape(3, 1), (dist if name == "register-raw" else None)
    j = json.loads((P5 if name == "photo5" else Path(name)).read_text(encoding="utf-8"))
    return (np.array(j["rvec"], float).reshape(3, 1), np.array(j["tvec"], float).reshape(3, 1),
            None if j.get("undistorted", True) else dist)


def to_world(xy, z=0.0):
    xy = np.asarray(xy, float).reshape(-1, 2)
    return np.c_[-xy[:, 0], xy[:, 1], np.full(len(xy), -float(z))]


def camera_height(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    c = -R.T @ np.asarray(tvec, float).reshape(3)
    return float(-c[2])


def pixel_to_map(px, rvec, tvec, K, dist=None, z=0.0):
    """픽셀 (N x 2) -> 지도 (N x 2, m). 바닥에서 높이 z(m)인 평면과 만나는 점"""
    px = np.asarray(px, float).reshape(-1, 1, 2)
    n = cv2.undistortPoints(px, K, dist).reshape(-1, 2)
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    c = -R.T @ np.asarray(tvec, float).reshape(3)
    d = np.c_[n, np.ones(len(n))] @ R                     # 각 행 = R.T @ [n, 1]
    s = (-float(z) - c[2]) / d[:, 2]
    w = c + s[:, None] * d
    return np.c_[-w[:, 0], w[:, 1]]


def map_to_pixel(xy, rvec, tvec, K, dist=None, z=0.0):
    p, _ = cv2.projectPoints(to_world(xy, z).reshape(-1, 1, 3), np.asarray(rvec, float).reshape(3, 1),
                             np.asarray(tvec, float).reshape(3, 1), K, dist)
    return p.reshape(-1, 2)


def fit_pose(points_px, K, dist=None, skip=SKIP_FOR_FIT, quiet=False):
    """기준점 픽셀 {이름: [u, v]} -> (rvec, tvec, 쓴 이름, 기준점별 바닥 오차 cm)"""
    ref = ref_points()
    unknown = [n for n in points_px if n not in ref]
    if unknown:
        raise SystemExit(f"모르는 기준점 이름 {unknown}. 쓸 수 있는 이름(대소문자 그대로): {list(ref)}")
    names = [n for n in points_px if n not in skip]
    if len(names) < 4:
        raise SystemExit(f"기준점이 4개 이상 필요합니다 (B 제외). 받은 것: {list(points_px)}")
    xy = np.array([ref[n] for n in names])
    ev = np.linalg.eigvalsh(np.cov((xy - xy.mean(0)).T))
    if not quiet and len(names) < MIN_POINTS_ADVISED:
        print(f"경고: 기준점 {len(names)}개 (B 제외). {MIN_POINTS_ADVISED}개 이상을 권합니다", file=sys.stderr)
    if not quiet and math.sqrt(max(ev[0], 0) / ev[1]) < 0.15:
        print("경고: 기준점이 거의 한 줄에 있습니다. 자세가 크게 틀릴 수 있으니 다른 방향의 점을 더하세요", file=sys.stderr)
    obj = to_world(xy)
    img = np.array([points_px[n] for n in names], float)
    ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise SystemExit("solvePnP(IPPE) 실패")
    ok, rv, tv = cv2.solvePnP(obj, img, K, dist, rv, tv, True, cv2.SOLVEPNP_ITERATIVE)
    h = camera_height(rv, tv) if ok else float("nan")
    if not ok or not HEIGHT_OK_M[0] <= h <= HEIGHT_OK_M[1]:
        raise SystemExit(f"자세가 이상합니다: 카메라 높이 {h:.3f} m (허용 {HEIGHT_OK_M[0]}~{HEIGHT_OK_M[1]} m). "
                         "기준점 이름·픽셀, 편 사진/원본 사진 구분(undistorted)을 확인하세요")
    err = {n: float(np.linalg.norm(pixel_to_map([points_px[n]], rv, tv, K, dist)[0] - ref[n]) * 100)
           for n in points_px if n in ref}
    return rv, tv, names, err


def selftest():
    K, dist = intrinsics()
    ref = ref_points()
    ok_all = True

    def judge(cond, text):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(("  통과 " if cond else "  실패 ") + text)

    reg = json.loads((REG / "reference_points_px.json").read_text(encoding="utf-8"))
    rv, tv, _ = load_pose("register-raw")
    print(f"[1] 등록 사진(2026-11-19 19:47) 기준점 -> 지도. 카메라 높이 {camera_height(rv, tv):.3f} m")
    worst = 0.0
    for n, p in reg["points"].items():
        e1 = float(np.linalg.norm(pixel_to_map([p["raw_px"]], rv, tv, K, dist)[0] - ref[n]) * 100)
        e2 = float(np.linalg.norm(pixel_to_map([p["undistorted_px"]], rv, tv, K, None)[0] - ref[n]) * 100)
        same = abs(e1 - p["floor_err_cm"]) < 0.01 and abs(e2 - e1) < 0.05
        if p["used"]:
            worst = max(worst, e1)
        judge(same, f"{n:3s} 원본 픽셀 오차 {e1:5.2f} cm · 편 사진 픽셀 오차 {e2:5.2f} cm · 기록 {p['floor_err_cm']:5.2f} cm"
              + ("" if p["used"] else "  (등록 때 안 씀)"))
    judge(worst <= 1.05, f"등록에 쓴 기준점 오차 최대 {worst:.3f} cm (기준 1.05 cm 이하, 등록 기록 1.004 cm)")

    print("[2] 지도 -> 픽셀 -> 지도 왕복")
    xy = np.array([[0.7, 0.35], [1.87, 0.53], [3.1, 1.3], [2.68, 0.48]])
    for label, d in (("원본", dist), ("편 사진", None)):
        back = pixel_to_map(map_to_pixel(xy, rv, tv, K, d), rv, tv, K, d)
        diff = float(np.abs(back - xy).max() * 100)
        judge(diff < 0.01, f"{label}: 왕복 차이 최대 {diff:.5f} cm")

    print("[3] 팀원 사진 5calibration.png (편 사진) 자세")
    p5 = json.loads(P5.read_text(encoding="utf-8"))
    rv5, tv5, d5 = load_pose("photo5")
    errs = {n: float(np.linalg.norm(pixel_to_map([px], rv5, tv5, K, d5)[0] - ref[n]) * 100)
            for n, px in p5["reference_px"].items()}
    used = [errs[n] for n in p5["fit_names"]]
    judge(max(used) <= 1.2, f"자세에 쓴 기준점 {len(used)}개 오차 최대 {max(used):.2f} cm")
    rvf, tvf, _, _ = fit_pose({n: p5["reference_px"][n] for n in p5["fit_names"]}, K, None, quiet=True)
    R1, _ = cv2.Rodrigues(rvf)
    R0, _ = cv2.Rodrigues(rv5)
    move = float(np.linalg.norm(R1.T @ tvf.reshape(3) - R0.T @ tv5.reshape(3)) * 100)
    judge(move < 0.1, f"fit_pose 로 다시 구한 자세와 저장된 자세의 카메라 위치 차이 {move:.4f} cm")

    print("[4] 잘못된 입력을 막는지")
    for label, pts in (("소문자 이름 'c'", {**{n: p5["reference_px"][n] for n in p5["fit_names"]}, "c": [0.0, 0.0]}),):
        try:
            fit_pose(pts, K, None, quiet=True)
            judge(False, f"{label}: 멈추지 않음")
        except SystemExit as e:
            judge("모르는 기준점" in str(e), f"{label}: 멈춤 ({str(e)[:40]}…)")
    bad = {n: p5["reference_px"][n] for n in p5["fit_names"]}
    bad["E1"], bad["R1"] = bad["R1"], bad["E1"]                 # 이름을 서로 바꿔 넣은 실수
    try:
        _, _, _, er = fit_pose(bad, K, None, quiet=True)
        judge(max(er.values()) > 5, f"E1·R1 픽셀 바꿔 넣음: 멈추지 않았지만 바닥 오차 최대 {max(er.values()):.1f} cm 로 드러남")
    except SystemExit as e:
        judge(True, f"E1·R1 픽셀 바꿔 넣음: 멈춤 ({str(e)[:40]}…)")

    print("결과: " + ("모두 통과" if ok_all else "실패 있음"))
    return 0 if ok_all else 1


def main():
    ap = argparse.ArgumentParser(description="사진 픽셀 <-> 지도 env2_20261119 좌표 변환")
    ap.add_argument("values", nargs="*", type=float, help="u v [u v ...] 또는 --to-pixel 이면 x y [x y ...]")
    ap.add_argument("--pose", default="register-raw", help="register-raw | register-undist | photo5 | 자세 json 경로")
    ap.add_argument("--to-pixel", action="store_true", help="지도 -> 픽셀")
    ap.add_argument("--z", type=float, default=0.0, help="바닥에서 높이 (m)")
    ap.add_argument("--fit", help="기준점 픽셀 json 으로 자세 구하기")
    ap.add_argument("--out", help="--fit 결과 자세를 저장할 json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    K, dist = intrinsics()
    if a.fit:
        j = json.loads(Path(a.fit).read_text(encoding="utf-8"))
        und = bool(j.get("undistorted", True))
        rv, tv, names, err = fit_pose(j["points"], K, None if und else dist)
        print(f"쓴 기준점 {names} · 카메라 높이 {camera_height(rv, tv):.3f} m (등록 때 2.216 m)")
        for n, e in err.items():
            print(f"  {n:3s} 바닥 오차 {e:5.2f} cm" + ("" if n in names else "  (자세 계산에 안 씀)"))
        if a.out:
            Path(a.out).write_text(json.dumps({"undistorted": und, "rvec": rv.ravel().tolist(), "tvec": tv.ravel().tolist(),
                                               "fit_names": names, "floor_err_cm": err}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
            print(f"저장: {a.out}  (다음부터 --pose {a.out})")
        return
    if len(a.values) % 2 or not a.values:
        ap.error("값을 두 개씩 주세요 (u v 또는 x y)")
    rv, tv, d = load_pose(a.pose)
    v = np.array(a.values).reshape(-1, 2)
    out = map_to_pixel(v, rv, tv, K, d, a.z) if a.to_pixel else pixel_to_map(v, rv, tv, K, d, a.z)
    for i, o in zip(v, out):
        print(f"({i[0]:.4f}, {i[1]:.4f}) -> ({o[0]:.4f}, {o[1]:.4f})" + (" px" if a.to_pixel else " m"))


if __name__ == "__main__":
    main()
