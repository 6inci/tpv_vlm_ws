# Real Environment Experiment Plan

- Version: **v3 (2026-11-15)**. v2(`REAL_ENV_EXPERIMENT_PLAN.md`, 2026-11-14)는 수정하지 않고 보존했다 (사용자 지시: 기존 자료를 남기고 새 파일로 갱신).
  - v3에서 바꾼 곳: §7 (D2 대체 표시), §8 (E3 개정), §9 (E4-lite 개정), §10 (결정 항목 추가), §11 (v3 개정 기록). 나머지 §0–§6은 v2와 같다.
  - v2 SHA-256 앞 16자리: `1e67157667939e0e`
- Status: `DESIGN` (구현·실행 전). v3 개정은 사용자 승인(2026-11-15, "갱신하되 기존 자료 남기고")에 따른 것이다. E3 실행 승인은 아니다.
- 적용 대상: 2026-11-14 구축 완료된 실제 미니어처 작업환경 (고정 3인칭 카메라)
- 표기: CONFIRMED = 2026-11-14 파일·장치 조회로 확인, INFERRED = 근거 있는 추정, UNKNOWN = 미확인, PROPOSED = 제안값(E0에서 확정)

---

## 0. 전제

### 0.1 이전 결과의 위치

- `results/`의 기존 run 7개는 AI 생성/합성 이미지(`warehouse_robotics_1920x1080.png`, `approximate_environment.png`) 또는 휴대폰 사진(`finding_modeling_object_experiment`)에서 나왔다. 실제 환경 실험의 **증거로 쓰지 않는다.**
- legacy(`wanna_find_detailed_experimental_x,Y`)의 geometry 수치도 다른 카메라 설치에서 나왔고, legacy D002에 따라 논문 수치로 인용하지 않는다.
- 재사용 대상은 코드와 설계뿐이다: 모델 추론 wrapper, JSON schema 검증, 지시문 유형 A–E, run 폴더 규약, 측정 기록 형식.

### 0.2 연구 범위 — E0 6단계 전에 결정해야 함

- legacy `RESEARCH_SCOPE.md`와 `AGENTS.md`의 연구 목표는 자연어 지시 → grasp-and-transport다. 최근 작업은 자연어 지시 → 이동 목표 grounding이다.
- 범위와 **무관한** 부분: 준비(R0), 재calibration(R1), E0 1–5단계(카메라, world frame, 구역, 고정 물체), E1.
- 범위에 **따라 달라지는** 부분: 이동 물체와 layout(E0 6단계), registry 항목(grasp interface 여부), 지시문, GT 기준.
- 권장(INFERRED): 이번 논문은 "자연어 → world 목표 grounding"으로 한정하고 grasp-transport는 후속으로 둔다. 결정 후 `AGENTS.md`와 legacy `RESEARCH_SCOPE.md`를 갱신한다.

### 0.3 확인된 자산·상태 (2026-11-14)

| 항목 | 상태 | 근거 |
|---|---|---|
| 기존 camera calibration | **재사용 불가 수준 (CONFIRMED)** | `/home/binci/camera_calibration_1920x1080/calibration_report.txt`: "reconstructed from the screenshots supplied in chat. No original raw frame", 1536×864 창 → 1920×1080 rescale. 보드 내부 코너 5×3·40 mm, 검증 8장 중 4장(2.2–8.4 px) 제외. 장치명은 "HD Web Camera"뿐이라 동일 장치 여부 판정 불가 → **R1에서 무조건 재calibration** |
| Undistortion 규약 | 충돌 (CONFIRMED) | viewer `compare_raw_undistorted.py`는 newCameraMatrix = K, 출력 1920×1080. legacy `geometry/legacy.py`는 저장된 `new_camera_matrix`와 2016×1133 기준 스케일 사용 → 하나로 고정 필요 (§2.1) |
| 외부 고정 카메라 연결 | 현재 미연결 (CONFIRMED) | `/dev/video0–3`이 모두 `HP True Vision FHD Camera`(노트북 내장) |
| Robot footprint | CONFIRMED (config 값) | 0.29 × 0.246 m. legacy `config/physical_geometry.json` ← `mecanum_bringup/config/nav2_params.yaml` |
| ArUco 도구 | CONFIRMED 존재 | calibration repo `aruco_pose.py`, `aruco_trajectory.py` |
| Legacy 모듈 | 존재 CONFIRMED, 재검증 안 함 | `perception/camera_source.py`, `world/{zones,object_footprint,measurement_registry,physical_geometry}.py`, `planning/collision_geometry.py`, `task/parser.py` |
| Legacy task parser | CONFIRMED | phrase 추출은 Qwen, alias 해석은 결정론적. VLM 없는 baseline으로 쓰려면 phrase 추출을 대체해야 함 |
| 실행 환경 | CONFIRMED | conda `qwen_vl`, `yolo_world`, `env_isaaclab`. RTX 5070 Ti Laptop 12 GB. `v4l2-ctl` 있음. `qwen_vl`에 cv2(5.0.0)·scipy 있음, shapely·statsmodels·labelme 없음 |
| Git | 없음 (CONFIRMED) | `/home/binci/vlm_ws`는 git worktree가 아님 |

---

## 1. 전체 구조

```text
R0 준비 ─► R1 재calibration ─► E0 환경 동결 ─► E1 geometry 검증 ─► 캡처·봉인 ─► D2 detector 결정(dev) ─► E3 메인(test) ─► E4-lite
                                                                                     (후속) E5 로봇 실행·외부 시각 검증
```

| 단계 | EXP ID | legacy Phase | 핵심 질문 | 로봇 구동 |
|---|---|---|---|---|
| R0 준비 | — | — | 기록·취득 도구가 갖춰졌나 | 없음 |
| R1 재calibration | EXP099 | Phase 0 일부 | 현재 카메라의 intrinsics/왜곡은? | 없음 |
| E0 | EXP100 | Phase 0 | 무엇을 어떤 좌표계로 고정했나 | 없음 |
| E1 | EXP101 | Phase 1 | pixel → world 변환이 얼마나 정확한가 | 없음 |
| D2 | EXP102 | Phase 4 일부 | 어떤 detector를 쓸 것인가 (결정 단계) | 없음 |
| E3 | EXP103 | Phase 5·7 | 자연어 지시를 올바른 world 목표로 바꾸나 | 없음 |
| E4-lite | EXP104 | — | 실패는 어디서 오나 | 없음 |
| E5 (후속) | EXP105 | Phase 2·3 | 로봇이 실제로 도달하나 | 있음 (사용자 실행) |

- EXP ID는 기존 합성 run(EXP001/EXP002)과 섞이지 않도록 099번부터 새로 쓴다.
- 선행 단계가 `PASS`가 아니면 의존하는 단계를 실행하지 않는다. 상태값은 AGENTS.md의 8개만 쓴다.
- 예상 규모(INFERRED, 검토자 추정): 1.5인 기준 5–7주. 대부분은 E3의 지시문·GT 작성이다.

---

## 2. 공통 행동지침 (모든 단계)

### 2.1 데이터 취득

1. 평가용 이미지는 **파이프라인이 직접 저장한 1920×1080 무손실 PNG**만 쓴다. 화면 캡처, JPEG, 휴대폰 사진은 쓰지 않는다.
2. raw 프레임을 원본으로 저장한다. undistorted 프레임은 raw + calibration 파일로 재생성 가능한 파생물이다.
3. 프레임마다 metadata JSON: timestamp(KST, ISO 8601), 카메라 경로(`/dev/v4l/by-id` 또는 `by-path`), USB VID:PID·serial, 해상도, FOURCC, `v4l2-ctl --all` 출력, calibration 파일 SHA-256, layout ID, 조명 조건.
4. 카메라는 장치 번호가 아니라 고정 경로로 연다.
5. autofocus, 자동노출, 자동 white balance를 끈다(`v4l2-ctl -l`로 지원 여부 확인). R1 이후 설정을 바꾸면 R1부터 다시 한다.
6. **undistortion 규약 (PROPOSED)**: `initUndistortRectifyMap(K, D, None, K, (1920,1080))`. viewer와 같고 출력 크기가 입력과 같다. config에 고정하고, legacy의 `new_camera_matrix`·2016×1133 경로는 쓰지 않는다.
7. **VLM 입력**: undistorted 1920×1080 PNG. Qwen의 0–1000 정규화 좌표 → pixel 변환은 Python에서 하고 변환식과 결과를 함께 기록한다.

### 2.2 카메라·환경 고정

1. R1 시작 후 카메라를 건드리지 않는다. 움직였으면 R1·E1을 새 run으로 다시 한다.
2. **drift 측정**: 작업영역 바깥 테두리 바닥에 평평한 ArUco 4–6개를 고정하고, 매 세션 시작·종료 시 코너를 자동 검출해 pixel 이동량을 계산한다. 수동 클릭은 쓰지 않는다. gate PROPOSED: max 2 px. 넘으면 그 세션 데이터는 E1 재검증 전까지 쓰지 않는다.
3. 테이프 구역과 고정 물체는 실험 기간 동안 옮기지 않는다. 옮기면 E0 registry를 새 version으로 갱신한다.

### 2.3 좌표계·단위·평가 영역

- 변수·필드 이름에 좌표계를 붙인다: `*_raw_px`, `*_undist_px`, `*_world_m`.
- world frame 이름은 `real_ws_v1`이다. 각도는 rad, world 기준, 반시계 방향 양수.
- homography는 바닥 평면(z=0)에만 유효하다. 높이가 있는 점(물체 윗면, 로봇 위 ArUco)은 z=0 homography로 변환하지 않는다.
- **평가 workspace**: 매트 하단과 충전 구역이 프레임 경계에 붙어 있다(INFERRED, 스크린샷). 그래서 평가 영역은 "매트 영역 ∩ 프레임 안쪽 margin(PROPOSED 20 px) 영역"으로 E0에서 정의한다.

### 2.4 사전 등록·dev/test 분리·봉인

1. GT, tolerance, 판정 규칙, 관계 규칙 상수, 프롬프트, detector vocabulary는 **test 데이터에 모델을 돌리기 전에** 파일로 고정하고 SHA-256을 기록한다.
2. 관계 규칙 상수는 로봇 치수에서 유도해 **dev 조정 전에** 고정한다. dev는 프롬프트와 schema 디버깅에만 쓴다.
3. test 프레임은 E1 직후 한 세션에서 모든 layout을 촬영하고, hash 목록으로 봉인한다. E3 최종 실행 전에는 어떤 모델에도 넣지 않는다.
4. 실행 후 실패 사례를 제외하거나 GT를 고치지 않는다. GT 오류가 발견되면 사유를 기록하고 새 GT version과 새 run으로 다시 계산한다.

### 2.5 실행·기록

1. 모든 run은 `results/EXPxxx_runNN/`에 새로 만든다. 기존 run을 덮어쓰지 않는다.
2. run마다 남길 것: **git commit**(R0의 git init이 선행 조건), config snapshot, 입력 파일 SHA-256, conda 환경, 모델 revision, raw model response, 처리 결과, metric, figure, log, `RUN_REPORT.md`.
3. VLM은 greedy decoding을 쓰고, 원문을 보존하고, JSON schema를 검증한다. 재시도 횟수는 사전에 정한 값(PROPOSED 1회)만 허용한다.
4. 보고서의 주장은 CONFIRMED / INFERRED / UNKNOWN으로 표시한다.
5. 경로, threshold, 물리 치수는 `config/`에 둔다(현재 없음 → R0에서 생성).
6. 단계 상태가 바뀌면 `PROJECT_STATUS.md`, `NEXT_STEP.md`, `DECISIONS.md`, `EXPERIMENTS.md`를 갱신한다. 작업 workspace에 아직 없으므로 R0에서 만든다.
7. polygon·충돌 연산은 legacy raster 기반 `planning/collision_geometry.py`를 재사용한다. shapely 설치는 필요할 때 승인을 받는다.

### 2.6 안전

- R0–E4는 로봇을 움직이지 않는다.
- E5는 사용자가 직접 실행한다. 순서는 offline → simulation → visualization → dry-run(명령 출력만) → 저속 실제 구동(AGENTS.md). marker 미검출, stale, 예외, 통신 실패 시 `cmd_vel = 0`.

---

## 3. R0 — 준비

1. 연구 범위 결정(§0.2).
2. `git init` (`/home/binci/vlm_ws`). 사용자 승인이 필요하다.
3. `config/` 골격, `PROJECT_STATUS.md`·`NEXT_STEP.md`·`DECISIONS.md`·`EXPERIMENTS.md`.
4. 캡처 스크립트: 고정 경로로 카메라를 열고, raw PNG와 §2.1의 metadata를 저장하고, undistorted 파생물을 만든다. `compare_raw_undistorted.py`의 로직을 port하고 출처를 기록한다(원본 repo는 수정하지 않음).
5. 카메라 설정 잠금(§2.1-5), 바닥 drift용 ArUco 고정(§2.2-2).

**완료 조건**: 캡처 스크립트로 저장한 프레임 1장과 metadata가 재현 가능하게 남는다.

---

## 4. R1 — 재calibration (EXP099)

**질문**: 현재 카메라(현재 초점·해상도)의 intrinsics와 왜곡 계수는 무엇이며, 독립 검증 프레임에서 오차는 얼마인가?

**절차**

1. 체커보드(내부 코너 9×6 이상, A3 이상) 또는 ChArUco 보드를 준비한다. 보드는 평평한 판에 붙인다.
2. 화면 네 모서리와 가장자리를 덮는 자세를 포함해 20–30장 촬영한다. 원본 PNG를 저장한다.
3. 별도 검증 프레임 5장 이상을 calibration에 쓰지 않고 남긴다. 검증 프레임은 결과를 본 뒤 제외하지 않는다.
4. 결과는 `vlm_ws` 쪽 새 파일(예: `config/real_env/camera_calibration_v1.json`)로 저장한다. 기존 calibration repo는 읽기 전용이다.

**지표**: calibration RMS(px), 검증 프레임 reprojection 오차 mean·max(px). 이미지 영역별(중앙/가장자리) 분해도 함께 낸다.

**PASS gate (PROPOSED)**: 검증 프레임 mean ≤ 0.5 px, max ≤ 1.0 px.

**진단(보조)**: 금속 직선자를 바닥에 놓고 undistorted 프레임에서 직선 적합 편차를 잰다. 손으로 붙인 테이프는 실제로 곧지 않을 수 있으므로 직선성 판정에 쓰지 않는다. 2026-11-14 스크린샷의 하단 매트 처짐(INFERRED)도 이것으로 원인을 구분한다.

---

## 5. E0 — 환경 동결 (EXP100, Phase 0)

**질문**: 실험에 쓰는 모든 물리 조건이 측정 근거와 함께 하나의 config에 고정되어 있는가?

**선행 조건**: R0 완료. 6단계 이후는 연구 범위 결정이 선행 조건이다.

**재사용**: legacy `config/physical_geometry.json`(robot footprint), `config/representative_object_measurement_TEMPLATE.json`과 `world/measurement_registry.py`(측정 기록 형식).

**절차**

1. 카메라 기록: by-id/by-path, VID:PID, serial, 설치 높이, 대략적 기울기, 마운트 사진.
2. world frame: 원점(예: 매트 모서리), +X, +Y, z=0 = 매트 윗면. 그림 1장.
3. 구역: 테이프 구역의 꼭짓점을 world 좌표로 측정한다. 경계 기준(테이프 안쪽 가장자리 vs 중심선)을 먼저 정한다. 구역마다 ID와 사람이 붙인 의미 label을 단다.
4. 고정 물체: 트럭, 선반, 로봇팔 셀, 컨베이어, 컨테이너 등의 바닥 footprint, 높이, 앞쪽 방향(정의 가능한 경우).
5. no-go 영역(예: 보행 통로, 로봇팔 셀 내부)과 평가 workspace(§2.3).
6. **anchor spot**: 이동 물체를 놓을 바닥 위치 15–20곳을 한 번만 측정하고 바닥에 작은 표식을 한다. layout은 "물체 → spot·방향" 배정표로 정의한다. layout마다 줄자 측정을 하지 않아도 된다.
7. 이동 물체 목록(class별 치수 = known geometry)과 layout 배정표. PROPOSED: L1 = dev, L2–L4 = test.
8. tolerance 확정.

**측정 방법 지침**

- 먼 점은 이미 측정한 기준점 **3개 이상**으로부터의 거리를 재고, 최소제곱으로 위치를 계산한다. 기준점 2개만 쓰면 거울 해와 기하 조건 악화가 생긴다. 키 큰 고정물이 줄자 경로를 막으면 다른 기준점을 쓴다.
- 모든 점을 2회 측정하고 차이를 기록한다.

**산출물**: `config/real_env/environment_v1.json`(camera, world frame, zones, fixed objects, anchor spots, movable object classes, layouts, no-go, eval workspace, robot, tolerances), 측정 원시 CSV, 사진, `results/EXP100_run01/RUN_REPORT.md`.

**PASS gate (PROPOSED)**: 필수 항목 100%(측정 출처 포함), 반복 측정 차이 max ≤ 5 mm, 상충값 없음.

**tolerance 결정 규칙 (PROPOSED)**

- 위치 여유(slack) = (가장 작은 목표 구역의 짧은 변 − 로봇 짧은 변 0.246 m) / 2.
- E1 holdout max ≤ 0.25 × slack, holdout RMSE ≤ 0.5 × (max gate).
- 예: 짧은 변 0.45 m인 구역이면 slack ≈ 0.10 m → max ≤ 2.5 cm, RMSE ≤ 1.3 cm.
- 충전 구역은 로봇보다 그리 크지 않아 보인다(INFERRED). slack이 너무 작으면 그 구역을 "정확 정지" 과제가 아니라 "접근" 과제로 정의한다.
- legacy 수치는 근거로 쓰지 않는다(D002).

---

## 6. E1 — geometry 검증 (EXP101, Phase 1)

**질문**: undistorted pixel → world(z=0) 변환 오차가 화면 중앙과 가장자리 모두에서 tolerance 안인가?

**선행 조건**: R1 PASS, E0 PASS.

**절차**

1. 바닥 기준점을 24–30개 잡는다: 매트 이음새 교차점, 테이프 모서리, 특징이 없는 곳에는 임시 십자 스티커. 평가 workspace 전체에 고르게, 네 모서리와 가장자리를 포함한다.
2. fit/holdout 분할을 **계산 전에** CSV로 고정한다: fit 12개(전체에 고르게), holdout 12개 이상(가장자리 4개 이상, 내부 4개 이상). holdout이 한 줄이나 fit 점 사이에만 몰리지 않게 한다.
3. pixel 좌표: raw 프레임에서 클릭(십자 스티커는 `cornerSubPix`)한 뒤 R1 calibration으로 undistort한다. 같은 점을 3회 클릭해 반복성을 기록한다.
4. world 좌표: E0 방식으로 측정한다.
5. fit 점으로 H를 추정하고, holdout 오차와 중앙/가장자리 RMSE, LOO를 보고한다.
6. (선택, 비용 낮음) 같은 점으로 `solvePnP`를 풀어 카메라 자세를 기록한다. E5에서 ArUco 높이 평면을 계산할 때 쓴다.

**재사용**: `scripts/run_metric_pipeline.py`의 `geometry_stage` 패턴(`cv2.findHomography`, residual 요약). legacy `LegacyGeometry`는 undistortion 규약이 달라 그대로 쓰지 않는다.

**PASS gate**: E0에서 확정한 tolerance.

**금지**: 결과를 본 뒤 holdout 점을 빼거나 fit으로 옮기는 것. 측정 오류가 확인되면 재측정 값을 근거로 새 run을 만든다.

**E1 직후 — 캡처·봉인**: 같은 세션에서 모든 layout(L1–L4)을 배치하고 layout마다 프레임 3장을 촬영한다. drift ArUco로 세션 안정성을 확인하고, test layout 프레임의 hash 목록을 봉인한다.

---

## 7. D2 — detector 결정 단계 (EXP102, Phase 4 일부)

> **v3 표시 (2026-11-15)**: D2는 `my_own_ws/EXP110_word_grounding`(EXP110 단어 찾기)이 대체한다(EXP110 DESIGN §0). 이동 물체(블록, 소형 팔레트) 탐지는 EXP111에서 다룬다. 아래 v2 원문은 기록으로 남긴다.

**성격**: PASS/FAIL gate가 아니라 사전 규칙에 따라 detector 하나를 고르는 **결정 단계**다. 1–2일로 제한한다.

**데이터**: dev layout(L1) 프레임만. 이동 물체 bbox GT(undistorted 기준 bbox + registry ID)를 OpenCV 두 점 클릭 도구로 만든다. labelme는 미설치이므로, 설치하려면 승인을 받는다.

**후보**: Grounding DINO tiny(`qwen_vl`), YOLO-World s(`yolo_world`). vocabulary는 E0의 이동 물체 class 목록으로 고정한다. 동의어 시도는 dev 안에서만 한다.

**지표**: class별 recall/precision(IoU ≥ 0.5), latency. 추가로 bbox 아랫변 중앙 → world 변환점과 anchor spot 측정 위치의 거리(m)를 잰다. 이 값이 P-real의 위치 오차 예산이다.

**결정 규칙(사전 고정)**: 이동 물체 class의 macro recall이 가장 높은 쪽. 동률이면 precision, 그다음 latency. 결과와 근거를 `DECISIONS.md`에 기록한다.

**재사용**: `results/finding_modeling_object_experiment/EXP_models_run01/run_experiment.py`(DINO/YOLO wrapper), `scripts/run_yolo_world_integration.py`.

---

## 8. E3 — 메인: 자연어 → 대상·목표 (EXP103, Phase 5·7) — v3 개정

v3에서 바뀐 점:
- (가) 대상 선택과 (나) 목표 영역을 분리했다.
- 관계를 누가 판단하는지 비교하는 조건을 추가했다.
- 장면 정보 출처 축(M-map, M-det)을 추가했다.
- 관계가 바뀌는 장면(S1–S8)을 추가했다.
- 정보 제공 원칙을 추가했다.
- v2 원문(§8.1–8.5)은 `REAL_ENV_EXPERIMENT_PLAN.md`에 그대로 있다. 설계 입력과 검토는 `my_own_ws/2026-11-15_E3_design_notes/NOTES.md`에 있다.

**연구 질문 (안)**: 고정된 3인칭 시점에서 VLM이 자연어로 표현된 객체 간 공간 관계를 얼마나 정확히 이해할 수 있으며, 기하학적 검증을 결합하면 잘못된 목적지 선택을 얼마나 줄일 수 있는가?

**하위 질문**

| ID | 질문 | 예 | 범위 |
|---|---|---|---|
| **E3a** | (가) 문장이 가리키는 물체 하나를 고르는가 | "IBC 옆 카트", "빈 소형 팔레트" | **핵심 연구. VLM 비교 실험** (§8.4) |
| E3b | (나) 고른 물체를 기준으로 도착 위치를 정하는가 | "IBC 앞으로 가" | **결정론적 모듈** (좌표 변환과 접근 목표점 계산, §8.5). 삭제하지 않는다. VLM끼리 비교하지 않고, 모듈의 정확도(목표점 유효성, 위치 오차)와 로봇 실행에서 검증한다 |

- **범위 결정 (사용자 2026-11-15)**: "VLM의 공간 관계 이해를 통한 목적 객체 선택을 핵심 연구로 설정하고, 기하학적 좌표변환과 접근 목표점 계산은 별도의 결정론적 모듈로 구현하여 실제 이동로봇 시스템과 연결한다."
- v2의 P-real vs B1+snap 비교(§8.3–8.4 원문)는 E3b를 VLM 비교로 할 때의 설계다. v3에서는 쓰지 않고 v2 파일에 기록으로 남긴다.
- **연구 방향 (사용자 2026-11-15)**
  - 핵심 평가 대상은 VLM이 시각적으로 객체 간 공간 관계를 판단하는 능력(J-vlm)이다.
  - 객체 탐지 정보로 VLM을 지원하는 것은 허용한다. 그러나 관계 판단을 처음부터 Python으로 대체하지 않는다.
  - Python 관계 계산(J-geo)은 비교·검증 조건이다. 필요하면 결합(J-gate)한다.

- 주의: E3a가 성공해도 ③ 의미적 구역 추론("하역장으로 가")이 검증된 것은 아니다. ③은 §9 E4-lite에서 따로 잰다.

**선행 조건**
- EXP110 G1 PASS (`my_own_ws/EXP110_word_grounding`, D2를 대체)
- EXP111(소형 물체) 결과
- world 좌표를 쓰면 E1 PASS
- 지시문, GT, 관계 규칙의 사전 등록

### 8.0 근거 (EXP110 dev 탐색, CONFIRMED)

- X3: 관계·상태 문장을 Qwen에게 한 번에 물었을 때 첫 박스가 지정 대상에 놓인 경우는 최고 13/48이었다.
  - "~옆" 질의 64건 중 29건이 기준 물체(IBC, 컨베이어, 트럭)를 가리켰다.
  - 없는 기준 물체("지게차 옆")에 16건 모두 박스를 냈다.
- 한계: C6는 DINO에 문장 전체를 넣어 후보가 나빴다. C1(후보 없이 Qwen 혼자)도 같은 혼동을 보였지만, 박스를 직접 그리는 형식이다. 따라서 번호를 고르는 조건(V1, V2)으로 다시 확인한 뒤 결론을 낸다.
- EXP110 사진(P01–P10)은 고정물 배치가 같아 관계의 정답이 늘 같다. 그래서 E3 평가에 쓰지 않는다.

### 8.1 정보 제공 원칙

판단 질문: "시스템이 원래 알아도 되는 규칙인가, 이번 실험에서 스스로 알아내야 하는 정답인가?"

| 구분 | 사전 제공 | E3에서의 예 |
|---|---|---|
| 시스템 규칙 | 가능 | 물체 어휘(closed-set으로 명시), JSON 형식, 관계 계산 규칙 |
| 작업 제약 | 가능 | 로봇 치수, 안전거리, 진입 금지 구역(Z01 보행 통로 등). 사전 지도 조건임을 명시 |
| 장면 정보 | 출처에 따라 다름 | detector 예측과 측정한 registry(M-map)는 가능. 사람이 그린 정답 박스는 oracle 조건에서만 |
| 문제의 정답 | 금지 | 정답 대상 ID, 정답 목적지, 이 사진의 구역 이름 |

- 명령 구조 {target, relation, reference}: Qwen이 직접 뽑으면 주 조건이다. 사람이 쓴 정답 구조를 쓰면 `oracle-parse` 조건이다.
- E3는 물체 어휘를 parser에 준다. EXP110(어휘 미제공)과 다르므로 논문에 명시한다.
- 탐지되지 않았다고 "없음"으로 단정하지 않는다. "확인 불가"로 출력한다.
- **정보 3분류** (사용자 전달 2026-11-15)

| 분류 | 내용 | 다루는 법 |
|---|---|---|
| 사전 지식 | 어휘, 로봇 치수, 주행 가능 경계, calibration·homography, 고정 시설 위치(M-map에서만) | 실행 전에 고정. 입력 가능 |
| 관측 정보 | 현재 사진에서 detector와 Qwen이 만든 결과 | 실행 중 생성. 입력 가능 |
| 평가 정답 | 사람이 그린 박스, 정답 대상 ID, 없음·모호, 구역 의미, 접근 가능한 목표 위치 | 채점 프로그램만 접근. 모델 입력과 파일·경로를 분리 |

- 판단 기준은 "고정물이 움직이는가"가 아니라 **"이 실험에서 시스템이 스스로 알아내야 하는 정보인가"**다.

### 8.2 장면 정보 출처 (사용자 결정 2026-11-15)

| 조건 | 고정물 위치 | 이동 물체 위치 | 목적 |
|---|---|---|---|
| **M-map** | 사전 지도 (E0 registry 측정값) | Detector | **우선 구현 구조이자 주 실험 조건** (최종 구조로 확정하지 않음) |
| **M-det** | Detector | Detector | 비교 조건 (사전 지도 의존성 검증, 완전 영상 기반) |

- 사용자 답변: 실제 현장에서 고정물은 사실상 움직이지 않는다. VLM 성과 분석 전까지는 노란 영역 안의 로봇만 움직인다. 다만 검증을 위해 위치를 약간 바꿀 수 있다.
- 고정물을 옮긴 layout에서는 그 layout의 registry를 다시 측정해야 M-map이 맞는 지도를 쓴다.
  - 재측정하지 않고 쓰면 "지도 갱신 누락" 조건으로 따로 표기한다. 이 경우 M-map과 M-det의 차이가 "언제 탐지가 필요한가"를 보여 준다.
- 이름은 v2의 지시문 유형 A–E, 기준선 B0/B1과 겹치지 않게 M-map, M-det으로 정했다.
- **최종 시스템 구조: 확정 보류 (사용자 2026-11-15)**
  - M-map은 가장 유력한 후보이며 우선 구현·평가 구조다.
  - 최종 구조는 E3 비교 결과(대상 선택 정확도, 잘못된 대상 비율, 보류 비율, 추론 시간, 안정성)를 본 뒤 정한다.
- M-map 결과를 "고정물까지 영상으로 인식한 결과"로 주장하지 않는다.
- **사전 지도 작성 규칙**
  - dev·test와 별도인 지도 작성용 촬영과 측량으로 만든다. world 좌표는 E1에서, 화면 박스는 그 프레임에서 얻는다.
  - test 사진이나 정답 박스에서 가져오지 않는다. test를 보고 지도를 고치면 더 이상 사전 정보가 아니다.
- 고정물을 옮긴 경우는 두 조건으로 구분한다.
  - "M-map + 갱신 지도": 재측량 비용을 기록한다.
  - "M-map + 갱신 안 된 지도": 지도 노후화에 대한 강건성 실험이다.
  - 본 test set은 고정물을 옮기지 않으므로(§8.3) 해당되지 않는다.

### 8.3 지시문과 장면

- 지시문 유형 A–E는 v2를 유지한다 (A 단일 관계, B 복합 관계, C 기능 구역, D 다단계, E 답할 수 없음).
- **관계가 바뀌는 장면** (외부 의견, 물체 2–3개만 옮겨 사진으로 평가. 로봇 주행 불필요)

| | 배치 | 정답 | 목적 |
|---|---|---|---|
| S1 / S2 | 카트 A / 카트 B가 IBC 옆 | A / B | 정답 대상이 바뀜 |
| S3 | 두 카트 모두 멀리 | 없음 | 관계 불성립 (탐지는 정상) |
| S4 | 두 카트가 비슷하게 가까움 | 모호 | 복수 후보 |
| S5 / S6 | IBC 왼쪽 / 오른쪽 | A / B | 방향 관계 |
| S7 | IBC 없음 | 확인 불가 | 기준 물체 부재 |
| S8 | IBC 옆에 카트가 아닌 물체 | 없음 | 종류와 관계 구별 |

- 고정물이 사실상 움직이지 않으므로 질의를 두 갈래로 둔다.
  - 현장과 맞는 질의: 이동 물체(소형 팔레트, 블록) 중심. 예: "트럭 옆 소형 팔레트", "빈 소형 팔레트".
  - 고정물(카트, IBC)을 옮긴 S1–S8: "관계 계산의 일반성 검증용" layout으로 표기한다.
- **결정 (사용자 승인 2026-11-15)**: S1–S8은 **고정물을 옮기지 않고** 이동 물체 배치와 질의 설계로 만든다. 사진 수를 늘리는 것보다 정답 대상과 관계가 실제로 바뀌게 구성하는 것이 목적이다.
  - 관계의 정답은 소형 팔레트와 블록의 위치로 바꾼다. 평소처럼 옮기면 되고 바닥 스티커 작업이 필요 없다.
  - 장면별 예: S1/S2 트럭 옆 팔레트가 바뀜, S3 트럭 근처에 팔레트 없음, S4 두 팔레트가 비슷한 거리, S5/S6 왼쪽·오른쪽 배치, S7 없는 기준("지게차 옆"), S8 기준 옆에 다른 종류 물체만 있음.
  - 고정물끼리의 관계("IBC 옆 카트")는 정답이 늘 같다. "고정 장면 항목"으로 따로 보고한다. M-map에서는 지도에서 미리 계산할 수 있다는 점도 함께 적는다.
  - 고정물을 옮겨 보는 시도는 몇 차례 정성적 사례로만 보고하고, 통계 주장에 쓰지 않는다.
  - 이동 물체 중심 질의는 EXP111(소형 물체 탐지)에 의존한다. oracle-det 조건으로 탐지 실패와 관계 판단 실패를 분리한다.
  - **적재 상태 교환**: 같은 질의("물건을 내려놓을 빈 팔레트")에 대해, 블록이 있는 팔레트와 빈 팔레트의 역할을 서로 바꾼 배치를 넣는다. "빈 팔레트" 같은 상태 정보는 사전 지도에 넣지 않는다.
  - dev와 test는 서로 다른 배치로 확보한다. 정답과 조건을 고정한 뒤 test는 1회만 평가한다.
- **규모**
  - 독립 단위는 사진 수가 아니라 서로 다른 (layout, referent) 조합이다. EXP110의 교훈: dev 질의 49개 중 41–44개가 4장 모두에서 같은 결과였다.
  - 초기안은 dev 8–12장, test 24–40장이다 (외부 의견). 통계적으로 충분하다는 보장은 없다.
  - v2의 검정력 계산(답변 가능 100–120개)은 (layout, referent) 단위로 다시 한다.
- **촬영**
  - 카메라 설치를 고정하고, calibration·homography 측정과 같은 설치 상태에서 찍는다.
  - EXP111 test 사진과 한 세션으로 묶는다.
  - EXP110 P01–P10은 E3 test로 쓰지 않는다.

### 8.4 비교 조건 (E3a)

**공통 입력**: 같은 사진, 같은 검출 결과와 ID(M-map 또는 M-det), 같은 문장, 같은 abstain 출력("없음", "확인 불가", "모호").

**진단 — VLM만으로 어디까지 되는가** (분석과 보고는 V0 → V3 순서로 한다. 한 번에 한 요인만 바뀐다)

| 조건 | 입력 | 최종 선택 |
|---|---|---|
| V0 | 원본 사진 + 문장 | Qwen |
| V1 | V0 + detector 박스 번호 | Qwen |
| V2 | V1 + 구조화 출력 형식 (정답 힌트 없음) | Qwen |
| V3 | V2 + 소형 물체 확대 이미지 (소형 물체 질의만, EXP111 출력 사용) | Qwen |

**관계 판단 주체 비교** (같은 parser 출력을 공통 입력으로 준다)

| 조건 | 관계 판단 | 역할 |
|---|---|---|
| J-vlm | Qwen이 관계를 판단해 ID 선택 | VLM 직접 추론 |
| J-geo | Qwen parser → Python 관계 규칙(§8.5) → ID | 기하 계산 (v2 P-real의 대상 선택 버전) |
| J-gate | J-vlm의 선택을 Python 규칙으로 검증. 불일치하면 보류 | 검증 결합 |
| J-fallback | J-gate에서 불일치하면 Python이 선택. 모호하면 보류 | 검증 + 재선택 |
| B0 | 키워드 parser + Python 규칙 (VLM 없음) | 규칙 기준선 |
| oracle-parse | 사람이 쓴 정답 구조 → J-geo | 명령 해석 오류 제거 (상한) |
| oracle-det | 정답 위치(registry/GT) → J-geo | 검출 오류 제거 (상한) |

**주 설계: 위치 정보 출처 × 관계 판단 주체 (2×2)**

| | J-vlm (Qwen 판단) | J-geo (Python 판단) |
|---|---|---|
| M-map | 안정된 관측 조건에서 VLM의 관계 판단 (**핵심**) | 같은 입력에서 기하 계산 |
| M-det | 탐지 오류가 VLM 판단에 주는 영향 | 영상 탐지 기반 기하 계산 |

- 외부 의견의 E3-A–D는 각각 M-map×J-vlm, M-map×J-geo, M-det×J-vlm, M-det×J-geo다. 이름 충돌을 피하려고 이 표기를 쓴다.
- **공정성**
  - 두 판단 주체는 같은 출처의 같은 후보 목록을 쓴다.
  - Qwen은 후보 ID를 그린 이미지와 문장을 받는다. Python은 같은 후보의 좌표를 받는다.
  - Qwen 조건에는 Python이 계산한 거리나 관계를 주지 않는다.
  - 입력 표현(시각 vs 수치)이 다르므로 "판단 방법의 시스템 성능 비교"로 보고한다. 순수 추론 능력 비교라고 주장하지 않는다.
- J-gate·J-fallback(결합), V0–V3(진단), oracle 조건은 이 2×2의 보조 조건이다.

- **"Python이 이긴다"를 전제하지 않는다.** 최종 구조는 dev에서 미리 정한 선택 기준(정답 대상 비율, 잘못된 대상 비율, 보류 처리, 지연시간)으로 고른다. test를 본 뒤 기준을 바꾸지 않는다.
- 최종 end-to-end 평가는 명령 해석부터 포함해 다시 잰다.
- C의 관계 검증기(J-gate, J-fallback)는 새 구성요소다. v2 validator는 충돌, 진입 금지, workspace만 검사한다.
- **모델 크기 (추가 비교, 사용자 계획 2026-11-15)**

| 조건 | 모델 | 실행 | 목적 |
|---|---|---|---|
| M1 | Qwen3-VL-4B | bf16, GPU (`qwen_vl`) | 기준선 |
| M2 | Qwen3-VL-8B | bf16, GPU + CPU offload (`qwen_vl`) | 8B 원본 성능. 소규모로만 돌린다 |
| M3 | Qwen3-VL-8B (4-bit) | bitsandbytes NF4, GPU (별도 환경 `qwen_vl_q4`) | 반복 실험용. 양자화 영향과 속도 |

  - M1 vs M2는 같은 bf16이라 모델 크기 효과를 본다. 단, M2의 속도에는 offload 영향이 섞이므로 속도는 크기 효과로 해석하지 않는다.
  - M2 vs M3는 양자화 효과를 본다. 양자화 설정은 dev에서 확정한다.
  - 논문에는 "Qwen3-VL-8B (4-bit)"처럼 정밀도를 명시한다.
  - 4B가 실패했다고 더 큰 모델이 해결책이라고 가정하지 않는다. 같은 후보와 같은 입력 표현에서 비교한다.
  - 8B 가중치: revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (17.55 GB).
  - `qwen_vl_q4`: `qwen_vl` 복제본에 bitsandbytes 0.50.2(CUDA 13.0 라이브러리 포함)만 추가했다. rollback은 `conda env remove -n qwen_vl_q4`이고, `qwen_vl`은 변경하지 않는다.

### 8.5 관계 규칙 (Python, 사전 고정)

- `NEXT_TO(대상, 기준)`, world 좌표(m)
  1. 기준 물체 미검출 → "확인 불가"
  2. 대상 후보 미검출 → "확인 불가"
  3. 후보마다 기준과의 간격 = 두 footprint 사이 최단 거리 (중심 간 거리가 아님)
  4. 간격 ≤ d인 후보만 남긴다. d는 로봇 치수에서 유도한다 (v2 §2.4)
  5. 0개 → "없음" / 1개 → 그 대상 / 여러 개 → 1위가 2위보다 margin 이상 가까우면 1위, 아니면 "모호 → 사용자 확인"
- 가장 가까운 것만 쓰면 멀리 있는 대상도 뽑힌다. 거리 조건만 쓰면 미니어처 환경에서 한 줄로 놓인 카트 둘이 모두 통과할 수 있다(INFERRED). 그래서 둘을 함께 쓴다.
- **민감도 분석**: d와 margin을 0.5배, 1배, 2배로 바꿔 보고한다. 값 하나를 주장하지 않는다.
- 방향(왼쪽, 오른쪽, 앞): v2 §8.2 주석 가이드를 따른다. 앞이 분명한 물체는 물체 기준, 그 외는 카메라 시점 기준. 둘 다 불분명하면 유형 E.
- **화면 좌표 임시 버전** (E1 전)
  - 간격을 픽셀로, d를 기준 물체 크기(픽셀)의 비율로 둔다.
  - 결과는 "화면 기준 임시"로 표기한다. 원근과 렌즈 왜곡 때문에 픽셀 거리는 바닥 거리가 아니다.
- bbox → world: 바닥 접촉점(아랫변 중앙)만 z=0 homography로 변환한다. 높이 있는 점은 변환하지 않는다(§2.3). footprint는 class 치수를 쓴다.

### 8.6 지표·통계 (E3a)

- **지표**
  - 정답 대상 비율
  - **잘못된 대상 비율** (로봇 오이동 위험)
  - 보류 비율
  - "없음"(S3)과 "확인 불가"(S7)를 각각 맞게 처리한 비율
  - 관계 종류별 결과, 지연시간, 최대 VRAM
- J-gate, J-fallback 추가 지표: 검증기가 VLM의 오답을 바로잡은 비율, 맞는 답을 잘못 거부하거나 바꾼 비율
- **주 비교·주 지표**: E3 test 전에 사전 등록한다. 후보는 M-map × J-vlm의 정답 대상 비율과 잘못된 대상 비율(핵심), 그리고 같은 입력에서 J-vlm vs J-geo 비교다.
- **관계 정답 작성**
  - 사진마다 사람이 사전 정의한 관계 기준에 따라 정답 대상 ID, 또는 "없음 / 모호 / 확인 불가"를 기록한다.
  - Python 관계 규칙으로 자동 생성하지 않는다. 규칙이 만든 답을 같은 규칙으로 채점하는 순환 평가를 막기 위해서다.
  - 규칙과 사람 판단의 일치율을 함께 보고한다.
  - 정답 파일은 모델 입력과 분리한다.
- **통계**: (layout, referent) 단위 cluster bootstrap 95% CI를 주 근거로 쓴다. McNemar는 참고로만 쓴다(쌍이 독립이 아님). 보조 비교는 Holm 보정을 한다.
- **안정성**: greedy decoding은 같은 입력에 같은 출력을 낸다 (EXP110 run02에서 확인). 그래서 layout마다 프레임 3장 사이의 일치율로 잰다.
- E3b를 포함하면 v2 §8.4를 그대로 쓴다.

### 8.7 행동지침

- v2 §8.5를 유지한다: test 전 SHA-256 기록, test는 조건마다 1회.
- 실제 로봇 이동은 사용자가 명시적으로 실행할 때만 한다. 확인 불가, 모호, 예외일 때 기본 명령은 `cmd_vel = 0`이다.

---

## 9. E4-lite (EXP104) 및 후속 — v3 개정

**E4-lite (이번 논문 범위, 작은 보조 실험)**

| 분석 | 내용 |
|---|---|
| 인식 오차 분해 | M-det vs M-map vs oracle-det (E3a에서 얻음) |
| **③ 의미 구역 추론** | 유형 C 지시문("하역장으로 가"). **주 조건 = 구역 번호만 준다(이름 없음)**. 구역 이름을 주는 조건은 검색 기준선·상한이다. "트럭이 있으면 하역장" 같은 고정 규칙은 규칙 기준선이다 |
| 실패 분류 | 검출 / 명령 해석 / 대상 선택 / 관계 판단 / geometry / validity |
| 모델 크기 | 4B vs 8B (§8.4) |

- ③의 정답 후보: Codex 초안 `my_own_ws/2026-11-15_zone_draft/semantic_zones_draft_v2.json`의 구역 이름(Z06 트럭 하역장 등). 사람 검토 후 **평가 정답으로만** 쓰고 입력으로는 주지 않는다.
- 구역 **경계(번호)는 사전 지식으로 주고 이름은 주지 않는 것**이 주 조건이다. 구역 경계 자체를 영상에서 찾는 것은 범위 밖으로 명시한다.
- 이 보조 실험만으로 넓은 작업환경 맥락 이해를 입증했다고 주장하지 않는다.
- test 데이터를 다시 쓰므로, 새로 조정한 설정은 "post-hoc 분석"으로 표기한다.

**후속으로 미룸**: SAM2 footprint, geometry Monte Carlo, detector 교체, E5.

**E5 (후속) 개요**

- E5a robot localization (Phase 2): 로봇 윗면 ArUco와 E1의 PnP 카메라 자세로 marker 높이 평면을 계산해 robot (x, y, yaw)를 얻는다. 줄자 측정 배치 10곳 이상과 비교한다.
- E5b 제어 (Phase 3): 외부 pose 피드백 go-to-goal. stale/미검출/예외 시 `cmd_vel = 0`.
- E5c 실행: test 지시문 일부. 외부 시각 판정과 줄자 측정을 비교해 판정 오차를 보고한다.
- 안전 순서는 §2.6을 따른다. 사용자가 직접 실행한다. ROS2·Jetson 통합 때문에 수 주 규모다(INFERRED).

---

## 10. 사용자 결정 필요 항목

1. 연구 범위(§0.2). **E0 6단계 전에** 결정해야 한다.
2. `git init` 승인.
3. world frame 원점·축, 테이프 경계 기준.
4. "앞/뒤/좌/우" 기준, 찬 구역 정책, tolerance 결정 규칙.
5. 두 번째 주석자(팀원) 참여 범위(100% 권장, 최소 20%).
6. 작업 위치: `new_experiment_environment/` 계속 사용 vs 새 workspace 폴더.
7. 재calibration 보드 종류(체커보드 vs ChArUco)와 크기.

**v3에서 추가 (2026-11-15)**

8. ~~E3b 범위~~ → 답변됨 (아래).
9. E3 주 비교와 주 지표의 사전 등록 (후보: J-vlm vs J-geo, 잘못된 대상 비율과 정답 대상 비율).
10. 관계 변화 촬영 계획: §8.3 PROPOSED(고정물은 옮기지 않고 이동 물체로 정답을 바꿈)의 확인. 고정물을 옮기지 않으면 registry 재측정이 필요 없다.
11. E1(world 좌표) 일정: 팀원의 homography와 calibration. 그 전에는 E3a의 화면 좌표 임시 버전만 가능하다.
12. ~~8B 비교 방식~~ → 답변됨 (아래).

**v3에서 답변된 항목**

- 고정물 이동: 실제 현장에서 사실상 움직이지 않는다. 검증을 위해 약간 옮길 수 있다 → §8.2 M-map(주 시스템)과 M-det(완전 영상 기반)를 둘 다 둔다.
- ③ 의미 구역 추론: 작은 보조 실험으로 포함 (§9).
- 관계 상수: 사전 고정하고 민감도 분석을 낸다 (§8.5).
- 8B 비교: M1/M2/M3 (§8.4). bf16 offload로 소규모 원본 기준을 확보한 뒤, 반복 실험은 4-bit를 별도 환경에서 돌린다.
- E3 범위: E3a는 VLM 비교(핵심), E3b는 결정론적 모듈(삭제하지 않음) (§8 하위 질문 표).
- 최종 시스템 구조: **확정 보류**. M-map은 우선 구현·주 실험 조건이고, M-det은 비교 조건이다. E3 비교 후 정한다 (§8.2).
- E3 test set: 고정물은 그대로 두고 이동 물체의 위치와 적재 상태로 정답을 바꾼다 (승인, §8.3). 이에 따라 §10-10은 답변됨.
- 연구 방향: VLM 직접 관계 판단(J-vlm)이 핵심이고, Python은 비교·검증이다 (§8 하위 질문 표 아래).
- **사용자 담당 범위 (2026-11-15)**
  - 자연어 명령 → 객체 탐지 → 목적 객체 선택 → 목표 위치 생성 → 경로계획(경로 좌표와 시각화)까지가 사용자 담당이다.
  - 실제 주행과 제어는 팀원 담당이다.
  - 지금은 이미지 좌표로 끝까지 연결한다.
  - homography(바닥 좌표 변환)는 사용자가 시작을 정할 때 E1 결과와 협업해 붙인다. ArUco로 로봇 위치를 받는 방식은 팀원과 인터페이스를 협의한다.
  - MCP는 시간이 허용될 때 하는 확장이다.

---

## 11. 검토 기록

### 2026-11-15 v3 개정 (Claude, 사용자 결정 반영)

- **입력**
  - 사용자가 붙여 넣은 외부 AI 의견 4건: 정보 제공 원칙, 역할 분담과 데이터 독립성, A0–A2/A·B·C/S1–S8 설계, 4B 역할과 8B 비교.
  - EXP110 dev 결과: X3, X1.
  - 사용자 결정: 고정물 이동, M-map/M-det 표, 계획 갱신 방식, 8B 진행.
  - 요약과 Claude 검토: `my_own_ws/2026-11-15_E3_design_notes/NOTES.md`
- **외부 의견과 다르게 둔 부분**
  - 조건 이름 A/B/C와 A0–A2는 v2의 지시문 유형 A–E, B0/B1과 겹친다. 그래서 V0–V3, J-vlm/J-geo/J-gate/J-fallback, M-map/M-det으로 바꿨다.
  - 외부 의견은 A1/A2를 v2 B1에, B를 v2 P-real에 대응시켰다. 그러나 v2 B1과 P-real은 목표 영역(E3b) 조건이다. 그래서 E3a와 E3b로 분리했다.
  - 8B 양자화와 4B bf16을 그대로 비교하면 크기와 양자화 효과가 섞인다. 그래서 bf16 offload를 우선하고, 양자화할 때는 4B도 양자화한다.
  - 확대 조건(V3)은 소형 물체 질의에만 적용하고, EXP111 출력을 쓴다.
- **같은 날 추가 반영 (사용자 결정 전달)**
  - 최종 구조 확정을 보류했다. Claude가 "최종 = M-map"이라고 했던 판단은 철회했다. 구현 안정성과 연구상의 적합성은 다르기 때문이다.
  - 2×2 주 설계, 정보 3분류, 사전 지도와 평가 정답의 구분, 관계 정답을 사람이 작성하는 규칙(순환 평가 방지), 적재 상태 교환, E4 구역 경계 규칙을 추가했다.

### 2026-11-14 독립 검토 (계획을 처음 보는 Claude 하위 에이전트, 읽기 전용)

총평: 재현성 장치는 좋다. 주 비교가 구조적으로 P에 유리했고, calibration·GT 정의·통계에 구멍이 있었다. 규모를 줄이면 실행 가능하다.

| 지적 | 심각도 | 확인 | 처리 |
|---|---|---|---|
| B1이 abstain·overlay·validity를 못 가져 strawman. B0는 oracle과 비교해야 함 | CRITICAL | 계획 본문 대조 | **반영**: 같은 overlay·schema·abstain, B1+snap 추가, GT 적중/유효성 분리, 주 비교를 P-real vs B1+snap으로 고정, B0는 P-oracle과 비교 |
| 고정 물체·구역 지시로 dev → test 누설 | MAJOR | INFERRED | **반영**: test 50% 이상 이동 물체 참조, referent별 보고, 규칙 상수 사전 고정 |
| 기존 calibration이 채팅 스크린샷을 rescale해 재구성됨 | MAJOR | **CONFIRMED** (`calibration_report.txt`) | **반영**: R1 무조건 재calibration. 직선성은 금속 직선자로 하는 진단으로 강등 |
| GT 매체·yaw·찬 구역 정책 미정 | MAJOR | 계획 본문 대조 | **반영**: top-down 지도에 GT, yaw 5° 규칙, 찬 구역 정책 |
| 검정력 부족, greedy 반복은 무의미 | MAJOR | **재계산으로 확인** (60개 → 0.58) | **반영**: 답변 가능 100–120개, 프레임 3장, cluster bootstrap, Holm |
| "E0–E4는 범위 공통"은 부분적으로 틀림 | MAJOR | AGENTS.md 대조 | **반영**: 공통/범위 의존 부분을 구분하고, 범위 결정 시점을 E0 6단계 전으로 |
| 규모 과잉, P-real 정의 모호 | MAJOR | INFERRED | **반영**: detector 2종 1–2일 결정 단계, E4-lite, E5 후속. P-real 정의 명시 |
| tolerance 근거로 legacy 수치 인용(D002 위반) | MINOR | **CONFIRMED** (D002) | **반영**: 구역 slack 기반 규칙으로 교체 |
| drift를 수동 클릭으로 측정 | MINOR | INFERRED | **반영**: 바닥 ArUco로 자동 측정 |
| 삼변측량에 기준점 2개 | MINOR | INFERRED | **반영**: 3개 이상 + 최소제곱 |
| undistortion 규약 충돌 | MINOR | **CONFIRMED** (viewer vs `legacy.py`) | **반영**: newCameraMatrix = K 규약 PROPOSED, Qwen 좌표 변환 명시 |
| 프레임 경계, test 봉인, shapely, git, simulation, 상태값, 문서 갱신 | MINOR | 일부 CONFIRMED | **반영** |

**검토 제안과 다르게 둔 부분**

- layout 수: 검토자는 3개(dev 1, test 2)를 제안했다. anchor spot 방식으로 layout 비용이 낮아졌으므로, cluster 수를 늘리려고 4개(dev 1, test 3)로 했다.
- 두 번째 주석자: 검토자는 100%를 제안했다. 팀원 가용성이 미정이라 100% 권장, 최소 20%로 했다.
- PnP: 검토자는 E5로 미루자고 했다. 같은 점으로 계산 비용이 낮아 E1에 선택 항목으로 남겼다.
