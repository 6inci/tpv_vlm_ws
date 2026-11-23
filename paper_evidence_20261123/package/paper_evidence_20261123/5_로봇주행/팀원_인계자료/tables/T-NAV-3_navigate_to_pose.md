# T-NAV-3 env2 NavigateToPose 사전 지정 목표 (X24, 2026-11-19 20:03, 그리퍼판, Nav2 P0, 재계산)

| 목표 | 시작 (KST) | 시간 (s) | 종료 (nav2.log) | 시작 거리 (cm) | 최소 거리 (cm) | 끝 거리 (cm) | 끝 방향 오차 (°) | 이동 (cm) | DWB 주기 (Hz) | 바퀴/명령 선속도 | 멈춤 (s) | No valid trajectories 경고 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20:03:59 | 17.3 | 다음 목표로 대체(preemption) — 실패 판정 전 | 54.7 | 20.3 | 20.3 | 39.7 | 43.0 | 15.5 | 0.56 | 6.05 | 0 |
| 2 | 20:04:16 | 34.7 | bt_navigator Goal failed (ABORTED) | 203.1 | 182.5 | 182.9 | 10.6 | 47.0 | 17.6 | 0.25 | 2.96 | 37 |
| 3 | 20:05:17 | 22.3 | bt_navigator Goal failed (ABORTED) | 58.2 | 57.6 | 57.7 | 12.1 | 16.7 | 10.1 |  | 0.11 | 47 |

- 목표 좌표 = bag `/goal_pose`, 거리 = `/odometry/global` 기준. 목표 1 은 실패 판정 전에 목표 2 가 들어와 대체됐다(`Received goal preemption request`).
- 이 bag 에는 action 상태 토픽이 없어, 종료는 nav2.log 의 bt_navigator 사건으로 판정했다. 목표 도달 0/3.
