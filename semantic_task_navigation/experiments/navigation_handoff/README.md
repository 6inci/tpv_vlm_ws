# Navigation Handoff

ROS2 Nav2 주행 담당자에게 넘길 **정적 지도(PGM/YAML)**와 **경로 데이터(Path A/B, 목표 자세)**를 만드는 실험입니다.
- 기존 연구 코드, 결과, 사진, Conda/ROS2 환경은 읽기만 합니다.
- 모델 호출, GPU, 로봇 구동은 없습니다.

## 단계 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| STEP 1 | 기존 자산 조사 | PASS (2026-11-19 사용자 승인) |
| STEP 2 | Nav2용 PGM/YAML 생성, 좌표 검증 | PASS (오프라인 검증 기준, 2026-11-20 사용자 승인)<br>map_server 실제 로드는 미실행 |
| STEP 3 | 직교 경로계획, 그리퍼를 고려한 차체 방향 | TESTING (사용자 검토 대기)<br>기준 데이터: photo_1, 바퀴 경계 모델, 여유 2.0 cm (`NOT_FOR_DRIVING`)<br>전달 패키지: `handoff_package/navigation_handoff_photo_1_20261120/` (+ `.zip`) |
| STEP 4 | 다중 환경 경로계획 | 4-A 사진 8장 분석 완료 (`results/20261120_062203_step4a_photo_survey/`)<br>4-B photo_5 재실험 (`results/20261120_064258_wheelbound_photo_5/`): TESTING (사용자 검토 대기)<br>4-B 일괄 예비 실험, 사진 2·3·4·6·7·8 (`results/20261120_070214_step4b_batch/`, `scripts/path_planning_batch_step4b.py`): TESTING (사용자 검토 대기)<br>시각화 수정본 (기본 그림과 진단 그림 분리, 경로·검사 결과 동일): `results/20261120_080841_step4b_batch/` (`VISUALIZATION_REVISION.md`) |

## 실행

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
$PY -m pytest -q -p no:cacheprovider experiments/navigation_handoff/tests            # 오프라인 검사
$PY experiments/navigation_handoff/scripts/map_export.py                           # STEP 2: results/<시각>_step2_map/
$PY experiments/navigation_handoff/scripts/object_positions.py --photo photo_5      # STEP 3: 후보 그림 (계획 입력 아님)
$PY experiments/navigation_handoff/scripts/path_planning.py --photo photo_5         # STEP 3: 확정 위치로만 계획
```

### 위치 지정과 승인 (2026-11-20 변경)

```bash
$PY experiments/navigation_handoff/scripts/position_picker.py       # GUI
```

GUI는 tkinter(표준 라이브러리)로 만들었습니다(2026-11-20).

- 이유: `stnav_gui`의 PySide6 6.12.0이 Qt 신호를 약 550번 보내면 `free(): invalid pointer`로 종료됩니다. GUI 없이, 깨끗한 환경에서도 재현됐습니다.
- 환경 변경: 설치·삭제는 없습니다.
- 검증: 가상 X11에서 실제 입력 이벤트로 시험했습니다(마우스 이동 3만 회, 휠 480회, 화면 이동 120회, 사진 전환 31회, 승인 1회). 비정상 종료는 없었습니다.

1. 사진(1~8calibration.png)을 고릅니다.
2. A(빨간 큐브)와 B(팔레트 배치 목표점)를 클릭으로 지정합니다.
   - 확대·축소: 마우스 휠
   - 화면 이동: 오른쪽 또는 가운데 버튼을 누른 채 끌기
   - 다시 클릭하면 위치가 바뀝니다.
3. 그리퍼 작업점을 따로 지정할 수 있습니다(선택). 지정하지 않으면 A·B 클릭점을 임시 작업점으로 씁니다.
4. 로봇 위치를 확인합니다.
   - 저장된 자동 추정(사진 4·5·8)이 먼저 표시되며, 틀렸을 때만 고칩니다.
   - 나머지 사진은 직접 지정해야 합니다.
5. 로봇 방향을 버튼이나 클릭으로 지정합니다. 방향을 자동으로 추정한 결과는 프로젝트에 없습니다.
6. 승인합니다.
   - `data/approved_positions/<photo>.yaml`과 미리보기 그림이 저장됩니다.
   - 같은 내용이 `history/`에 사본으로 남으며, 이 사본은 덮어쓰지 않습니다.
7. 승인된 사진에서만 "경로계획 실행"이 켜집니다. 결과는 `results/<시각>_plan_<photo>/`에 저장됩니다.
   - `maps/`: map.pgm, map.yaml (STEP 2 파일의 바이트 동일 복사본)
   - `<photo>/`: path_A.csv, path_B.csv, waypoints.yaml, 결과 그림, validation.json, object_positions.json, path_plan.json

### STEP 3 변형: 바퀴 경계 모델 (2026-11-20)

```bash
$PY experiments/navigation_handoff/scripts/path_planning_wheel_boundary.py --photo photo_1   # results/<시각>_wheelbound_photo_1/
```

**충돌 규칙**

- 바퀴 외곽(URDF 기준 27.2×24.7 cm) + 여유: 노란 주행 영역(free) 안에 있어야 합니다.
- 차체(35×24 cm) + 여유: 고정 장애물, 금지 구역, 매트 바깥에 닿으면 안 됩니다. 노란선 밖으로 튀어나오는 것은 허용하지만, 그 영역의 물체는 지도에 없으므로 `UNVERIFIED`입니다.

**조건과 결과 저장**

- 여유 조건: 3.5 cm(기존), 2.0 cm(축소, 진단), 0 cm(공칭, 진단). 0~3.5 cm는 0.5 cm 간격 훑기로 상태만 기록합니다.
- 조건별 결과는 `<조건>/photo_1/`에, 비교 결과는 `comparison.json`, `comparison_maps.png`, `comparison_photos.png`에 저장합니다.
- `path_planning.py`와 그 결과는 바꾸지 않았습니다.

`path_planning.py --photo <name>`도 승인 파일만 받습니다(`status: USER_APPROVED`). 같은 사진을 다시 실행하면 저장된 승인 좌표를 그대로 씁니다.

참고: `config/object_positions/photo_5.yaml`(채팅으로 확인한 후보, `USER_CONFIRMED`)은 STEP 3 첫 실행의 기록이며, 이제 계획 입력이 아닙니다.
- `results/superseded_runs/20261120_034604_step3_plan_photo_5`는 폐기된 실행입니다.
  - 원인: 계획 C-space와 독립 차체 검사의 래스터 경계 규칙이 1픽셀(0.25 cm) 달랐습니다.
  - 현재 코드는 두 검사가 같은 다각형 래스터를 사용합니다.

## 좌표계

| 이름 | 정의 |
|---|---|
| `undistorted_px` | 사진 8장의 픽셀. **이미 왜곡 보정됨**(newK = K)이므로 다시 보정하지 않습니다 |
| `floor_cm` | E1 매트 이음선으로 만든 잠정 바닥 좌표(`PROVISIONAL_UNVALIDATED`)<br>원점: 매트 노드 (0,0)(카메라에서 먼 왼쪽 모서리)<br>+x: 이미지 오른쪽, +y: 카메라 쪽 |
| `map_m` | ROS2 `map` 좌표계(REP-103, 오른손 좌표계)<br>원점: 매트 노드 (0,2)(카메라 쪽 왼쪽 모서리)<br>+x: 이미지 오른쪽, +y: 카메라에서 멀어지는 쪽 |
| `grid_rc` | `map.pgm`의 행·열. 0행이 가장 큰 `y_map`입니다(Nav2 규칙) |

변환식:
- `x_map = x_floor/100`
- `y_map = (210 − y_floor)/100`
- `yaw_map = −yaw_floor`

`odom`과 `base_link`는 여기서 정의하지 않습니다. `map→odom`은 로봇의 위치추정이 제공해야 합니다.

## 정적 지도 규칙

| 영역 | PGM 값 |
|---|---|
| Z02 주행 영역 | 254 (free) |
| 적치 구역 Z03·Z04, 금지 구역 Z01·Z07, Z02 밖 작업장 | 0 (occupied) |
| 3×2 매트 바깥 | 205 (unknown) |

- 셀은 보수적으로 분류합니다. 셀 전체가 자유 공간일 때만 free입니다.
- 로봇, 큐브, 팔레트는 정적 지도에 넣지 않습니다. 사진마다 달라지는 장애물은 STEP 3 이후 사진별 계획 지도로 따로 처리합니다.
- 구역은 사람이 확인하지 않은 DRAFT이고, 바닥 변환은 잠정값입니다.

## 결정 기록 (2026-11-19)

- 차체는 35×24 cm로 계산합니다. 팀원 Nav2 footprint(29.0×24.6 cm)는 사용하지 않습니다.
- 그리퍼 치수는 측정하지 않았으므로 `null`로 둡니다. 사용자는 물리 테스트를 하지 않습니다.
- 그리퍼만 주행 영역 경계를 넘을 수 있습니다. 차체는 free 영역 안에 있어야 합니다.
- Nav2 map_server 로드 테스트는 사용자가 요청할 때만 진행합니다. 지금은 형식 검증까지만 합니다.
- 좌표 기준점은 3×2 매트(한 변 105 cm) 격자입니다. 사람이 실측한 별도 기준점은 없습니다.
