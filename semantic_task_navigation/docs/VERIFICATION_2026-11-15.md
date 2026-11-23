# 검증 기록 — v0.1 (2026-11-15)

- 실행 환경: `stnav_gui`(GUI·테스트), `qwen_vl`(모델 worker), RTX 5070 Ti Laptop 12 GB
- 표기
  - PASS: 해당 조건에서 실제로 실행해 기대 결과를 확인함
  - MOCK-PASS: 모델 대신 가짜 응답을 쓴 테스트
  - NOT_TESTED: 실행하지 않음
  - BLOCKED: 필요한 입력이 없어 실행할 수 없음

## 1. 필수 테스트 (지시문 §20)

| 항목 | 결과 | 근거 |
|---|---|---|
| A 프로그램 실행 | PASS | `run.py` GUI image·video·replay 모드 실행·종료(exit 0), headless image 모드, `test_gui_image_demo_run` |
| B 1920×1080 이미지 표시 | PASS | 무손실 QImage 변환, 가로세로 같은 배율(800×450), 클릭 좌표 왕복 오차 3 px 미만 (`test_image_shown_with_uniform_scale`) |
| C 명령 해석 | PASS (실제 Qwen) + PASS (규칙 기반) | `test_qwen_parses_example_command`(실제 모델), `test_task_parser.py` 17개 |
| D 객체 탐지 (실제 모델) | PASS | `test_dino_produces_candidate_boxes`, 참고 이미지 LOCAL_ONLY 실행에서 conveyor·truck·red_cube 채택 |
| E 작은 큐브 ROI·좌표 복원 | PASS | 합성 영상에서 1 px 이내, 왕복 1e-9 이하. 실제 실행에서 HSV·전체 DINO·ROI DINO 박스가 일치 |
| F 지도 생성 | PASS | 레이어 분리, 다각형 밖 차단, footprint 여유 (`test_layers_are_separate`) |
| G 경로 생성 | PASS | 정상 지도, 직선화, 검증 통과 |
| H 경로 검증 | PASS | 장애물을 관통하는 경로 거부, 시작·목표 충돌, 지도 밖 |
| I GUI 안정성 | PASS (실제 모델) | 실제 모델 분석 79.7초 동안 이벤트 루프 최대 지연 0.094초, p99 0.026초, 미리보기 29.9 FPS 유지. MOCK-PASS: 가짜 1.5초 추론 중 지연 0.3초 미만, 중지 후 2초 이내 추론 중단 |
| J 결과 저장 | PASS | 필수 파일, 계약 스키마, manifest 필드, 파일 SHA-256 일치, 덮어쓰기 거부 |
| K 실패 처리 | PASS | 트럭 없음 → 하역 영역·경로 B 없음. 다른 이미지 → 수동 데이터 거부와 지도 정렬 거부. 지원하지 않는 명령. 대상 없음·복수 대상 |
| L 인계 | PASS | 확인 없이 완료 불가, 허용되지 않은 출처 거부, 하드웨어 예외, 상태 스냅샷 추가 |
| M 실행 재현 | PASS | 실제 모델 실행 2건과 데모 실행에서 replay 그림 10개 모두 픽셀 동일 |

- 전체 테스트: `STNAV_RUN_MODEL_TESTS=1 python -m pytest -q` → **68 passed** (실제 모델 3개 포함)
- 기본 실행(모델 테스트 제외): 65 passed, 3 skipped

## 2. 실제 실행 결과 (`runs/`)

| run_id | 모드 / 명령 | 결과 | 비고 |
|---|---|---|---|
| `20261115_225944_49f5` | LOCAL_ONLY / 운반 (최종 코드) | PARTIAL_PLAN | 아래 상세 |
| `20261115_224618_0e4d` | LOCAL_ONLY / 운반, GUI(video 입력) | PARTIAL_PLAN | GUI 실제 모델 통합, 장면 변화 경고 확인 |
| `20261115_225333_ce4c` | IMAGE_DEMO / 운반 | PLAN_COMPLETE | 수동 객체·하역 영역(`[MANUAL]`) |
| `20261115_225614_61aa` | IMAGE_DEMO / 이동 | PLAN_COMPLETE | `navigate_to_region` |

### 최종 LOCAL_ONLY (`20261115_225944_49f5`, 77.7초, 모델 호출 30회, peak VRAM 8.87 GiB)

**성공한 단계**
- 명령 해석(Qwen3-VL-4B `ebb281e`): transport_and_place, red cube, produced_by conveyor, loading_area near truck
- 장면 이해: "industrial simulation setup"
  - 불확실 항목: 큐브 위치, 컨베이어 생산 여부, 작업 순서
- 탐지(DINO `a2bb814`, 53개 후보) → Qwen 검증
  - 채택: conveyor_1, truck_1
  - mobile robot: 후보 3개 모두 거부(로봇팔, 트럭)
  - 로봇 시작점은 지도의 수동 시작점을 사용했다.
- 큐브: [581.4, 296.4, 596.7, 307.9] px
  - HSV, 전체 이미지 DINO, ROI DINO 세 출처의 박스가 일치했다.
  - 후보 10개 중 8개를 검증해 1개를 채택했다.
- 관계
  - `conveyor_1 near red_cube_1` [COMPUTED]
  - `red_cube_1 produced_by conveyor_1` [UNVERIFIED]
- 지도: 카메라 정렬 ALIGNED. 동적 장애물 1개(컨베이어 박스가 통행 영역 가장자리와 겹침)
- 픽업 접근 위치: (620, 388) px. 경로 A 361 px(직선화 전 385 px, waypoint 45 → 2), 최소 여유 65 px, 검증 통과

**실패한 단계: 하역 영역 `SEMANTIC_REGION_INACCESSIBLE`**
- VLM이 후보 2(트럭 위쪽, 매트 밖 바닥)만 plausible로 판정했다.
- 판정 근거: "Open cargo bed facing area 2"
- 해당 후보의 접근 가능 셀은 0개였다.
- 후보 1(아래쪽)은 접근 가능 셀이 568개였지만, VLM이 "no"로 판정했다.
- 결과
  - 하역 접근 위치는 SKIPPED, 경로 B는 없다.
  - 인계 상태는 NOT_EXECUTABLE이다.
  - 그럴듯한 영역이나 경로로 대체하지 않았다.

## 3. 보존 대상 실행 (이전 코드 버전)

아래 실행은 개발 중 기록이라 삭제하지 않았다. 정리 여부는 사용자가 결정한다.

| run_id | 상태 |
|---|---|
| `20261115_222501_cf07`, `20261115_222824_d8fb`, `20261115_224116_16ec`, `20261115_225338_025d` | 이전 설정이나 코드로 만든 정상 결과 (manifest에 코드 해시 기록) |
| `20261115_223643_4ab0` | ROI 시각화 버그로 `save_results`에서 예외 발생. `result.json` 없음 (수정 완료) |
| `20261115_225334_e254` | 이동 명령 IMAGE_DEMO 버그(`target_key`). `internal_error`가 기록됨 (수정 완료) |

`runs/` 전체 크기는 약 466 MB다(크롭, ROI, 그림 PNG).

## 4. 시험하지 않았거나 차단된 항목

| 항목 | 상태 | 이유 |
|---|---|---|
| 외부 고정 카메라 라이브 분석 | BLOCKED | `HD Web Camera` 미연결. 노트북 카메라는 장치 열기와 해상도만 확인했고, 사생활 보호를 위해 프레임은 저장하지 않음 |
| `LOCAL_VLM_WITH_API_AGENT` 실제 API 호출 | NOT_TESTED | API 키 없음. 키가 없을 때 로컬로 전환되는 코드는 있음 |
| CALIBRATED_GROUND_MAP (metric) | BLOCKED | 현재 배치에 검증된 Homography가 없고, v0.1에서 구현하지 않음 |
| 정확도 (14.2–14.6) | NOT_TESTED | 사람이 만든 정답과 평가용 이미지가 없음. `evaluate_run.py`는 대용 GT로 동작만 확인 |
| 실제 로봇 주행, 파지, ROS2 publish | NOT_TESTED (의도적) | 하드웨어 명령 비활성. 인계는 JSON 요청과 시뮬레이션 상태만 |
