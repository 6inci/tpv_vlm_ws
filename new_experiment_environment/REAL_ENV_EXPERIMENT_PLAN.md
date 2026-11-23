# Real Environment Experiment Plan

- Version: v2 (2026-11-14). v1 초안 → 독립 검토 → 반영. 변경 내역은 §10.
- Status: `DESIGN` (구현·실행 전, 사용자 승인 전)
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

**성격**: PASS/FAIL gate가 아니라 사전 규칙에 따라 detector 하나를 고르는 **결정 단계**다. 1–2일로 제한한다.

**데이터**: dev layout(L1) 프레임만. 이동 물체 bbox GT(undistorted 기준 bbox + registry ID)를 OpenCV 두 점 클릭 도구로 만든다. labelme는 미설치이므로, 설치하려면 승인을 받는다.

**후보**: Grounding DINO tiny(`qwen_vl`), YOLO-World s(`yolo_world`). vocabulary는 E0의 이동 물체 class 목록으로 고정한다. 동의어 시도는 dev 안에서만 한다.

**지표**: class별 recall/precision(IoU ≥ 0.5), latency. 추가로 bbox 아랫변 중앙 → world 변환점과 anchor spot 측정 위치의 거리(m)를 잰다. 이 값이 P-real의 위치 오차 예산이다.

**결정 규칙(사전 고정)**: 이동 물체 class의 macro recall이 가장 높은 쪽. 동률이면 precision, 그다음 latency. 결과와 근거를 `DECISIONS.md`에 기록한다.

**재사용**: `results/finding_modeling_object_experiment/EXP_models_run01/run_experiment.py`(DINO/YOLO wrapper), `scripts/run_yolo_world_integration.py`.

---

## 8. E3 — 메인: 자연어 → world 목표 (EXP103, Phase 5·7)

**질문**: VLM이 대상(물체/구역 ID)과 관계만 고르고 목표 좌표는 결정론적으로 계산하는 방식(P)이, VLM이 목표 위치를 직접 고르는 방식(B1)보다 목표 성공률이 높은가?

**선행 조건**: E1 PASS, D2 결정 기록, 지시문·GT·규칙 사전 등록 완료.

### 8.1 지시문 세트

- 유형은 기존 A–E를 재사용한다: A 단일 관계, B 복합 관계, C 기능 구역, D 다단계, E 답할 수 없음.
- 규모 PROPOSED: dev(L1) 약 40개, test(L2–L4) 답변 가능(A–D) 100–120개 + E 유형 약 20개.
  - 검정력 계산(불일치 25%, 차이 15%p 가정): 답변 가능 60개 → 0.58, 100개 → 0.84, 120개 → 0.90.
- test 답변 가능 항목의 **50% 이상은 이동 물체·새 배치를 참조**하게 한다. 고정 물체·구역만 참조하는 지시는 dev에서 조정한 내용이 그대로 통하는 누설 위험이 있기 때문이다. 결과는 fixed/movable referent로 나눠 보고한다.
- dev와 test의 지시문 표현은 겹치지 않게 쓴다.

### 8.2 GT (방법과 독립적으로)

- **어디에 그리나**: registry(E0 측정값)로 렌더링한 metric top-down 지도 위에 그린다. 카메라 이미지 위에 그려 H로 변환하지 않는다. 그렇게 하면 GT가 방법과 같은 geometry를 공유한다.
- **누가 언제 그리나**: 주석 가이드에 따라, 방법의 출력을 보기 전에 그린다. 두 번째 주석자(팀원)는 test 100%를 권장하고 최소 20%를 독립적으로 그린다. 주석자 간 IoU를 보고한다. 주 GT는 주석자 A이고, 주석자 B의 GT로 계산한 결과를 robustness로 함께 보고한다.
- **주석 가이드에 고정할 것**
  - "앞/뒤/좌/우" 기준. PROPOSED: 앞이 분명한 물체(트럭 = 운전석 쪽, 선반 = 개방면)는 물체 기준, 그 외는 카메라 시점 기준. 둘 다 불분명하면 E 유형.
  - 물체로 찬 구역(드럼 적재, 로봇팔 셀, 트럭 dock)으로 가라는 지시: 구역 안 free 영역이 로봇 footprint를 수용하면 그 영역, 아니면 구역 경계에서 d_approach 이내의 free 띠(no-go 제외)를 정답으로 한다.

### 8.3 비교 조건

모든 VLM 조건은 같은 Qwen3-VL-4B, greedy decoding, **같은 overlay 이미지**(이동 물체 박스 ID + 구역 번호), **같은 JSON schema와 abstain 옵션**을 쓴다.

| 조건 | 이동 물체 정보 | VLM 출력 | 목표 좌표 |
|---|---|---|---|
| **P-real** (제안) | D2 detector (위치 = bbox 아랫변 중앙 → world, footprint = class 치수, 방향 = class 기본값) | 대상 ID, 관계, 또는 abstain | Python 관계 규칙 + validator |
| P-oracle (상한) | registry(anchor spot 측정값) | 동일 | 동일 |
| B1 | 동일 overlay | 목표 pixel 또는 abstain | E1 변환만 |
| **B1+snap** | 동일 overlay | 목표 pixel 또는 abstain | E1 변환 → 같은 validator로 가장 가까운 유효점에 투영 |
| B0 (선택) | registry | 없음 | 키워드·alias 매칭 + 고정 규칙. legacy alias 해석 함수 재사용, phrase 추출은 키워드 매칭으로 대체. **P-oracle과 비교** |
| P-text (선택) | registry를 텍스트로만 (이미지 없음) | 대상 ID, 관계 | P와 동일. "이미지가 필요한가"를 검증 |

- 고정 물체와 구역은 모든 조건에서 registry(사전 지도)에서 온다. 이것은 legacy `RESEARCH_SCOPE.md`의 가정("zone은 config의 고정 polygon")과 같으며, 논문에 명시한다.
- 관계 규칙 상수는 로봇 치수에서 유도해 사전 고정한다. 예: in_front_of = 물체 앞 방향으로 footprint 경계에서 (로봇 반길이 0.145 m + margin) 떨어진 띠, between = 두 footprint 사이 최단 구간, near = footprint에서 d 이내, 구역 = 구역 polygon ∩ free space. 후보 영역 ∩ free space에서 기준점에 가장 가까운 점을 고른다.
- **validator yaw 규칙 (PROPOSED)**: 0°–175°를 5° 간격으로 검사해 하나라도 충돌이 없으면 valid. 로봇 footprint가 직사각형이라 180° 대칭이다.

### 8.4 지표·통계

- **주 지표 (사전 고정)**: 답변 가능(A–D) 항목의 Success. Success = 목표점이 GT polygon 안(GT 적중) AND 유효(측정 registry 기준 충돌·no-go 없음, 평가 workspace 안). 유효성은 방법이 무엇을 인식했든 **항상 측정 registry로** 판정한다.
- GT 적중과 유효성을 **따로도** 보고한다.
- E 유형: abstain 정밀도·재현율. A–D에서 잘못 abstain한 비율도 보고한다.
- 보조 지표: GT polygon까지 거리(m), 대상 ID 선택 정확도, layout별 프레임 3장 간 출력 일치율. greedy decoding 반복은 하지 않는다.
- **주 비교 (사전 고정)**: P-real vs B1+snap. McNemar 정확 검정(`scipy.stats.binomtest`)을 쓴다. 성공률의 95% CI는 referent(및 layout) 단위 cluster bootstrap으로 낸다.
- 보조 비교(P-real vs B1, P-oracle vs B0 등)는 Holm 보정을 적용한다.

### 8.5 행동지침

- test 실행 전 체크리스트: 프롬프트, 규칙 상수, vocabulary, GT, 주석 가이드의 SHA-256 기록 완료. dev 결과 보고서 작성 완료. 봉인 hash 일치 확인.
- test는 조건마다 1회만 실행한다. 실행 후 프롬프트를 고쳐 다시 돌리면 그 결과는 test 결과가 아니다.

---

## 9. E4-lite (EXP104) 및 후속

**E4-lite (이번 논문 범위)**

| 분석 | 내용 |
|---|---|
| 인식 오차 분해 | P-real vs P-oracle (E3에서 이미 얻음) |
| 구역 이름 ablation | 구역을 번호만 줄 때 vs 이름을 줄 때 |
| 실패 분류 | 검출 / 대상 선택 / 관계 해석 / geometry / validity로 나눠 집계 |

test 데이터를 다시 쓰므로, 새로 조정한 설정은 "post-hoc 분석"으로 표기한다.

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

---

## 11. 검토 기록

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
