# Navigation Handoff 데이터 (photo_1, 2026-11-20)

> **상태: 개발용 참조 데이터입니다. `NOT_FOR_DRIVING`, 파지·배치 `UNVERIFIED`.**
> - 이 경로로 실제 로봇을 주행한 적이 없습니다.
> - 안전 여유(2.0 cm)가 좌표 오차(3.5~5.2 cm)보다 작습니다.
> - 그리퍼 치수를 측정하지 않았습니다.

## 1. 목적

외부 고정 카메라 사진 1장(`1calibration.png`)에서 만든 다음 데이터를 넘겨드립니다. ROS2/Nav2 주행 개발을 바로 시작하실 수 있도록 하는 것이 목적입니다.

- 작업장 정적 지도
- 로봇 시작 자세
- 빨간 큐브 파지 정차 자세, 팔레트 배치 정차 자세
- 그 사이의 직교 경로(Path A, Path B)

실제 주행 검증, Nav2 파라미터 조정, 위치추정은 받는 분의 작업 범위입니다.

## 2. 파일

| 파일 | 역할 |
|---|---|
| `maps/map.pgm`, `maps/map.yaml` | Nav2 map_server용 정적 지도. 168×115 셀, 0.02 m/셀 |
| `maps/map_preview.png` | 지도 미리보기 (축과 원점 표시) |
| `paths/path_A.csv` | Path A 원본 좌표 103점: 로봇 시작 → 큐브 파지 정차 |
| `paths/path_B.csv` | Path B 원본 좌표 88점: 파지 정차 → 팔레트 배치 정차 |
| `paths/waypoints.yaml` | 자세 목록: 시작, 방향 전환점, 회전, 정차 |
| `figures/path_planning_photo.png` | 원본 사진 위 경로 (바닥 높이 투영) |
| `figures/path_planning_map.png` | 지도 위 경로 |
| `validation/validation.json` | 충돌·경계 검사 결과와 상태 |
| `validation/path_plan.json` | 상세 기록: 구간, 이벤트, 접근 후보, 이격거리 |
| `handoff_summary.yaml` | 이 README의 주요 값을 기계가 읽을 수 있게 정리한 파일 |
| `handoff_check.json` | 이 패키지에 대한 최종 확인 결과 |
| `reference/` | 좌표 변환(`coordinate_frames.json`), 확정 위치(`approved_positions.yaml`), 여유 조건 비교(`comparison.json`) |

## 3. 좌표계와 데이터 형식

### 지도 (`map.yaml`)

```yaml
image: map.pgm
mode: trinary
resolution: 0.02
origin: [-0.1, -0.1, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
```

| PGM 값 | 뜻 | 내용 |
|---|---|---|
| 254 | free | 노란 주행 영역 |
| 0 | occupied | 적치 구역, 금지 구역, 노란선 밖 작업장 |
| 205 | unknown | 매트 바깥 |

이 지도는 STEP 2에서 만든 지도와 바이트 단위로 같습니다(map.pgm sha256 `3737df031a7f62b5…`). 정적 지도이므로 로봇, 큐브, 팔레트는 들어 있지 않습니다.

### `map` 좌표계

- ROS REP-103을 따릅니다(오른손 좌표계, z 위쪽, 단위 m).
- **원점 (0, 0):** 바닥 매트 격자에서 카메라 쪽 왼쪽 모서리입니다. 실제 바닥에서 찾을 수 있는 점입니다.
- **+x:** 사진의 오른쪽입니다.
- **+y:** 카메라에서 멀어지는 쪽(사진의 위쪽, 트럭 쪽)입니다.
- **작업장 범위:** 매트 3×2장이며 x 0~3.15 m, y 0~2.10 m입니다. 지도 파일은 둘레 0.1 m 여백을 포함해 3.36 × 2.30 m입니다.
- **제공하지 않는 것:** `odom`과 `base_link`는 이 데이터에 없습니다. `map → odom` 변환은 로봇의 위치추정이 제공해야 합니다.

### Yaw

- `map`의 +x에서 **반시계 방향**으로 잽니다.
- 값은 deg(`yaw_deg`)와 rad(`yaw_rad`)로 함께 제공합니다.
- **yaw는 차체 전면(그리퍼 쪽, `base_link` +X)이 향하는 방향**입니다. 이동 방향과 다릅니다.
- 270°는 −90°와 같습니다. quaternion은 z = sin(yaw/2), w = cos(yaw/2)입니다.

| 자세 | yaw | quaternion (z, w) |
|---|---|---|
| 시작 | −178.2° | (−0.99988, 0.01571) |
| 정차 | 270° (= −90°) | (−0.70711, 0.70711) |

270°를 그대로 넣으면 (0.70711, −0.70711)이 나옵니다. q와 −q는 같은 회전입니다.

### 경로 CSV 열

| 열 | 뜻 |
|---|---|
| `index` | 진행 순서. 0이 시작점 |
| `x_map_m`, `y_map_m` | `map` 좌표 (m). **이 두 열을 쓰시면 됩니다** |
| `row`, `col` | map.pgm의 셀 위치 (0행이 이미지 위쪽) |
| `x_floor_cm`, `y_floor_cm`, `u_px`, `v_px` | 바닥 cm 좌표와 사진 픽셀. 참고용입니다 |
| `move_to_next`, `move_yaw_map_deg` | 다음 점으로의 이동 방향 (±x_map, ±y_map) |
| `body_yaw_map_deg` | 그 점을 떠날 때의 차체 방향 |
| `is_waypoint` | 방향 전환, 회전, 끝점이면 1 |

- 모든 점은 2 cm 셀 중심이고, 이웃한 점끼리는 x나 y 한 축으로만 2 cm 떨어져 있습니다(4방향, 대각선 없음).
- 지도 셀과의 대응 규칙: col = floor((x + 0.1) / 0.02), row = 114 − floor((y + 0.1) / 0.02)

## 4. Path A와 Path B

### 시작 자세와 목표

| 항목 | x (m) | y (m) | yaw | 비고 |
|---|---|---|---|---|
| 로봇 시작 (추정 base_link) | 2.9446 | 0.5390 | −178.2° | 사진에서 읽은 바퀴 중심. 방향은 바퀴 축으로 계산 |
| 경로 시작점 | 2.95 | 0.53 | −178.2° | 시작을 포함한 셀의 중심 (1.05 cm 차이) |
| A 큐브 (파지 목표) | 2.6425 | 0.2175 | — | 충전소 옆 회색 팔레트 위. 노란선 밖 |
| **파지 정차 자세** | **2.65** | **0.51** | **270°** | 그리퍼가 차체 전면에서 **11.8 cm** 앞까지 닿아야 함. 옆 어긋남 0.7 cm. `UNVERIFIED` |
| B 팔레트 (배치 목표) | 0.9355 | 0.2238 | — | 흰 판과 초록 선반 사이 작은 팔레트. 노란선 밖 |
| **배치 정차 자세** | **0.93** | **0.49** | **270°** | 그리퍼가 차체 전면에서 **9.1 cm** 앞까지 닿아야 함. 옆 어긋남 −0.5 cm. `UNVERIFIED` |

### Path A (204 cm, `path_A.csv`)

| # | 구간 | 차체 yaw | 차체 기준 이동 |
|---|---|---|---|
| 1 | (2.95, 0.53) → (1.83, 0.53): −x 112 cm | −178.2° | 전진 |
| 2 | (1.83, 0.53) → (1.83, 0.57): +y 4 cm | −178.2° | 횡이동 |
| — | (1.83, 0.57)에서 제자리 회전 −178.2° → 270° (반시계 88.2°, index 58) | | |
| 3 | (1.83, 0.57) → (1.83, 0.51): −y 6 cm | 270° | 전진 |
| 4 | (1.83, 0.51) → (2.65, 0.51): +x 82 cm | 270° | 횡이동 |

시작점 부근(오른쪽 경계)과 파지 정차 위치(Z04 아래 38 cm 통로) 모두 회전할 공간이 없습니다. 그래서 왼쪽 열린 곳(x 1.83 m)까지 갔다가 회전한 뒤 되돌아옵니다.

### Path B (174 cm, `path_B.csv`, 회전 없음)

| # | 구간 | 차체 yaw | 차체 기준 이동 |
|---|---|---|---|
| 1 | (2.65, 0.51) → (0.93, 0.51): −x 172 cm | 270° | 횡이동 |
| 2 | (0.93, 0.51) → (0.93, 0.49): −y 2 cm | 270° | 전진 |

### 경로계획 조건

- **충돌 기준:**
  - 바퀴 외곽 27.2 × 24.7 cm(URDF 값) + 여유 2.0 cm는 노란 주행 영역(free) 안에 있어야 합니다.
  - 차체 35 × 24 cm + 여유 2.0 cm는 적치 구역, 금지 구역, 매트 바깥에 닿으면 안 됩니다.
  - 차체가 노란선 밖으로 나가는 것은 허용했습니다(최대 1.0 cm). 노란선 밖 물체는 지도에 없어 충돌 여부를 확인하지 않았습니다.
  - 그리퍼는 노란선 밖으로 뻗을 수 있다고 가정했습니다.
- **경로 탐색:** 4방향 A*(방향 전환 비용 포함)로 계산했습니다. 정차 자세 후보는 0°, 90°, 180°, 270°입니다.
- **검사 결과 (모두 PASS):**
  - 4방향 연결, 대각선 구간 없음
  - 바퀴 경계 이탈 0, 차체 충돌 0
  - 바퀴와 노란 경계 사이 최소 3.25 cm, 차체와 지도 장애물 사이 최소 3.0 cm
  - 회전 중 차체 정적 검사(1° 간격) 통과
- **여유 조건 비교 (`reference/comparison.json`):**

  | 여유 (cm/변) | 결과 |
  |---|---|
  | 3.5 | 경로 없음 (파지 정차 자세 없음) |
  | 3.0 이하 | 경로 생성 |
  | 0 (공칭) | A 182 cm, B 182 cm |

## 5. Nav2 적용 방법 (둘 중 선택)

### A. Nav2가 경로를 직접 계산하는 방식

1. 지도를 올립니다: `ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=<경로>/maps/map.yaml` (lifecycle 활성화 필요).
2. **`map → odom`이 필요합니다.** 현재 `mecanum_bringup`에는 LiDAR가 없고 EKF가 `odom → base_link`만 냅니다. AMCL(`/scan` 필요)은 그대로는 쓸 수 없습니다. 위 시작 자세를 알고 놓는 경우의 정적 변환이나 외부 위치추정이 필요합니다.
3. 목표를 보냅니다.
   - `NavigateToPose`로 파지 정차 자세, 그다음 배치 정차 자세를 보냅니다.
   - 또는 `NavigateThroughPoses`에 `waypoints.yaml`의 자세 목록을 넣습니다.

   예시:

   ```bash
   ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
     "{pose: {header: {frame_id: map}, pose: {position: {x: 2.65, y: 0.51}, orientation: {z: -0.70711, w: 0.70711}}}}"
   ```

4. **이 경우 Nav2는 Path A/B를 따르지 않고 자기 경로를 새로 계산합니다.** 현재 `nav2_params.yaml`과의 차이 때문에 경로가 없거나 다르게 나올 수 있습니다.

   | Nav2 설정 | 현재 값 | 이 데이터 기준 |
   |---|---|---|
   | footprint | 29 × 24.6 cm | 차체 35 × 24 cm |
   | inflation_radius | 0.55 m | 통로 폭 38~43 cm |
   | xy_goal_tolerance | 0.25 m | 그리퍼 정렬에 cm 단위 필요 |
   | yaw_goal_tolerance | 0.25 rad | |

### B. 이 A* 경로를 그대로 쓰는 방식

1. `path_A.csv`와 `path_B.csv`를 `nav_msgs/Path`로 바꿉니다(`frame_id: map`). 아래는 데이터 변환 예시이며 노드는 아닙니다.

   ```python
   import csv, math
   from nav_msgs.msg import Path
   from geometry_msgs.msg import PoseStamped

   def csv_to_path(csv_file):
       path = Path()
       path.header.frame_id = "map"
       with open(csv_file, newline="") as f:
           for r in csv.DictReader(f):
               p = PoseStamped()
               p.header.frame_id = "map"
               p.pose.position.x = float(r["x_map_m"])
               p.pose.position.y = float(r["y_map_m"])
               yaw = math.radians(float(r["body_yaw_map_deg"]))   # body heading, not the move direction
               p.pose.orientation.z = math.sin(yaw / 2)
               p.pose.orientation.w = math.cos(yaw / 2)
               path.poses.append(p)
       return path
   ```

2. **Path를 publish하면 RViz에 표시될 뿐이고, 로봇이 따라가지는 않습니다.** 실제로 따라가려면 경로 추종 컨트롤러가 필요합니다. 예를 들어 Nav2 `controller_server`의 `FollowPath` 액션에 횡이동이 가능한 컨트롤러를 쓰거나(현재 설정은 DWB, vy ±0.12 m/s), 별도 컨트롤러를 만들어야 합니다.
3. Path A의 index 58에는 **제자리 회전**(−178.2° → 270°)이 있습니다. 경로 추종기가 제자리 회전을 처리하지 못할 수 있으니, 그 지점에서 경로를 나누고 회전(예: Spin 동작)을 따로 실행하는 방식을 검토해 주세요.
4. 각 구간에서 차체 방향은 일정하고, 이동은 전진 또는 횡이동입니다(메카넘). 위 4절 표를 참고하세요.

## 6. 확인한 것과 확인하지 않은 것

**이 패키지에서 확인한 것** (`handoff_check.json`, 모두 PASS):

- 필수 파일 8개 존재
- map.yaml 형식과 상대 경로(`image: map.pgm`), PGM 형식(P5, 168×115, 값 0/205/254), STEP 2 지도와 동일
- 경로의 모든 점: 지도 셀 대응, free 셀 위치, 4방향 연결, 바닥·사진 좌표와 일치
- 시작·끝점 연결(A 끝 = B 시작), waypoint 순서·좌표·yaw와 CSV 일치
- 사진과 지도 그림이 같은 좌표에서 그려짐
- 미검증 상태가 기록되어 있음

**확인하지 않은 것:**

- Nav2 map_server 실제 로드 (Nav2 미설치, 형식만 확인)
- 실제 주행 전반

## 7. 주행 담당자가 추가로 검증할 항목

1. **위치추정 (`map → odom`):** 시작 자세를 실제 매트 원점 기준으로 맞추는 방법을 정해야 합니다.
2. **좌표 정확도:** 바닥 변환은 잠정값이고(사람이 측정한 기준점 없음) 오차가 3.5~5.2 cm입니다. 큐브와 팔레트의 높이 시차는 보정하지 않았습니다.
3. **지도:** 구역은 사람이 확인하지 않은 DRAFT입니다. 노란선 밖 물체(팔레트, 충전소 기둥 등)는 지도에 없습니다.
4. **로봇 치수:**
   - 바퀴 폭 40 mm: URDF 값만 있음
   - 차체 35 × 24 cm가 `base_link`를 중심으로 하는지
   - 차체 전면(그리퍼 쪽)이 `base_link`의 +X인지
   - 그리퍼 오프셋과 도달 거리: 정차 위치에서 차체 전면 앞 11.8 cm(파지), 9.1 cm(배치)까지 닿아야 함
5. **안전 여유:** 이 데이터는 여유 2.0 cm 기준이고 위치 오차보다 작습니다. 여유 3.5 cm에서는 파지 정차 자세가 없었습니다.
6. **동작:** 제자리 회전(1.83, 0.57 m 지점)과 횡이동이 실제로 수행되는지, 정지 정확도가 그리퍼 정렬에 충분한지 확인이 필요합니다.
7. **Nav2 파라미터:** footprint, inflation, goal tolerance를 위 4절 조건과 맞춰야 합니다(5절 A 표 참고).

## 출처

- 생성 스크립트: `experiments/navigation_handoff/scripts/make_handoff_package.py`
- 원본 실행 폴더: `experiments/navigation_handoff/results/20261120_055014_wheelbound_photo_1/m2.0_reduced/photo_1/`
- 원본 파일은 바이트 단위로 복사했고 sha256으로 확인했습니다(`handoff_summary.yaml`의 `sources`). 원본 결과는 수정하지 않았습니다.
