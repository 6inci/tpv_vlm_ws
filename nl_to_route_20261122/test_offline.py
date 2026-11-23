#!/usr/bin/env python3
"""모델 없는 검사: 좌표 변환, 정차 자세, 경로계획 (nl_to_path.py).
  /home/binci/miniconda3/envs/stnav_gui/bin/python -B test_offline.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nl_to_path as N  # noqa: E402


def main():
    geo, pl = N.Geo(), N.Planner()
    # 1 공유 사진 자세: 기준점 픽셀 -> 지도, 자세에 쓴 점 오차 <= 1.1 cm, 왕복
    ref = N.p2m.ref_points()
    err = max(float(np.linalg.norm(geo.to_map([geo.pose["reference_px"][n]], 0.0)[0] - ref[n]) * 100) for n in geo.pose["fit_names"])
    xy = np.array([[0.7, 0.35], [2.68, 0.48]])
    rt = float(np.abs(geo.to_map(geo.to_px(xy, 0.035), 0.035) - xy).max() * 100)
    assert err <= 1.1 and rt < 0.01, (err, rt)
    print(f"1 자세: 기준점 오차 최대 {err:.2f} cm, 왕복(z 3.5 cm) {rt:.5f} cm  OK")
    # 2 팀원이 계산한 큐브 위치 (2.645, 0.206) 앞 정차: 카메라 쪽 노란 선 밖이므로 270°, 팀원 계산(차체 1 cm 여유에서 약 2 cm 모자람)보다 조금 더 모자람
    sp = pl.stop_pose([2.645, 0.206], 0.0125)
    assert sp and sp["yaw_deg"] == 270 and 2.0 <= sp["reach_gap_cm"] <= 8.0 and sp["body_clearance_cm"] >= 3.0, sp
    print(f"2 정차: {sp['xy_m']} {sp['yaw_deg']}°, 몸체 여유 {sp['body_clearance_cm']} cm, 모자람 {sp['reach_gap_cm']} cm  OK")
    # 3 팀원 Path A와 같은 출발·도착 (팀원시작 180° -> 큐브파지 270°): 경로가 있고, 팀원 판정 불가 0, 모든 구간이 축 방향, 회전 한 번
    r = pl.plan([2.9718, 0.5035], 180, [2.678, 0.4822], 270)
    pts = [r["route"]["start_xy_m"]] + r["route"]["waypoints_xy_m"] + [r["route"]["goal_xy_m"]]
    axis = all(abs(a[0] - b[0]) < 1e-6 or abs(a[1] - b[1]) < 1e-6 for a, b in zip(pts[:-1], pts[1:]))
    assert r["status"] == "OK" and r["check_teammate"]["bad_cm"] == 0 and axis and r["route"]["rotate_at"], r
    print(f"3 경로: {r['check_teammate']['length_m'] * 100:.0f} cm, 경유점 {len(pts) - 2}, 회전 {r['route']['rotate_at']}, "
          f"몸체 여유 최소 {r['check_teammate']['min_body_clearance_cm']} cm, 회전 여유 {r['check_teammate']['rotate_min_clearance_cm']} cm  OK")
    # 4 막힌 도착점 (가운데 구역 안)은 경로를 만들지 않음
    r = pl.plan([2.9718, 0.5035], 180, [1.45, 0.83], 180)
    assert r["status"] in ("GOAL_BLOCKED", "NO_PATH"), r
    print(f"4 막힌 도착점: {r['status']}  OK")
    print("모두 통과")


if __name__ == "__main__":
    main()
