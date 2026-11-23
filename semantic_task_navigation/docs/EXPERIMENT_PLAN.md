# EXPERIMENT PLAN (초안)

## 1. 원칙

- **데이터 분할**
  - 개발 데이터: `assets/reference/warehouse_scene_reference_01.jpg`. 프롬프트와 임계값 조정에 이미 사용했다.
  - 평가 데이터: 새로 촬영한 이미지로 별도 구성해야 한다.
  - 평가 이미지로 설정을 바꾸지 않는다. 바꾼 경우에는 새 버전으로 기록하고 평가를 다시 한다.
- **결과 구분**: 다음 다섯 가지는 항상 따로 기록한다.
  - 자동 결과와 수동 결과
  - 이미지 좌표와 실제 좌표
  - 객체 중심과 접근 목표
  - 계획 성공과 실제 주행·파지 성공
  - 공간 관계 계산과 의미 추론
- **수치 산출**: 모든 수치는 저장된 `runs/<id>/result.json`과 사람이 만든 정답 파일에서 `scripts/evaluate_run.py`로 다시 계산한다. 손으로 만든 수치는 쓰지 않는다.

## 2. 평가 항목 (각 항목을 독립적으로 채점)

| 항목 | 입력 | 지표 | 비고 |
|---|---|---|---|
| 14.1 Task Parsing | 명령 문장 + 정답 구조 | 작업 유형·대상·출발 기준·목적지 일치 여부, 오류 코드 정확도 | 지원하지 않는 명령, 복수 대상, 목적지 없음 등 부정 사례 포함 |
| 14.2 Object Grounding | 실행 결과 + 정답 박스 | 클래스별 최고 IoU(≥0.5), 대상 IoU와 **중심점 포함 여부**(작은 물체용), TARGET_NOT_FOUND / AMBIGUOUS 비율 | IMAGE_DEMO 실행은 NOT_APPLICABLE |
| 14.3 Semantic Grounding | 사람이 그린 하역 영역 다각형 | 선택 영역 IoU ≥ 0.3, UNCERTAIN·INACCESSIBLE 비율 | VLM 후보별 판정과 접근 가능성은 `semantic_regions.json`에서 따로 분석 |
| 14.4 Goal Selection | 지도 + 정답 대상·배치 영역 | 접근 위치 존재, 목표와 정답 배치 영역의 거리(px), 단계별 실패 코드(배치/접근/도달) | reach가 PROVISIONAL이면 물리적 가능성 판정이 아님 |
| 14.5 Path Planning | 지도 | 경로 발견 여부, 검증 통과 여부, 최소 여유, 직선화 전후 길이·waypoint 수 | 정상과 고장 지도 시나리오(테스트 참조) |
| 14.6 End-to-End Planning | 명령 + 이미지 | 위 항목이 모두 자동으로 성공한 비율 | 수동 입력을 쓴 실행은 자동 E2E로 집계하지 않음 |

**정답 파일 형식**: `scripts/evaluate_run.py` docstring 참고. 사람이 작성한다.
- 이미지 SHA-256, 작업, 객체 박스, 대상 박스, 하역 영역 다각형

## 3. 제안 프로토콜

1. **카메라 고정과 확인**
   - 외부 HD Web Camera를 고정한다.
   - `run.py --mode live`로 지도 정렬 상태를 확인한다(`MAP_CAMERA_MISALIGNED`가 없어야 함).
2. **지도 검토**
   - `reference_scene_map.json`(DRAFT)을 사람이 검토한다.
   - GUI 지도 편집으로 수정하고 새 파일로 저장한다. `review_status`를 갱신한다.
3. **촬영**
   - 배치를 바꿔 가며 촬영한다: 큐브 위치, 방해물(빨간 물체), 트럭 방향.
   - 장면당 1장 이상, 개발 이미지와 겹치지 않게 한다.
4. **실행**
   - 장면마다 명령 세트를 고정해 둔다(정상, 이동, 지원하지 않는 작업, 복수 대상, 목적지 없음).
   - `LOCAL_ONLY`로 실행한다. 반복 실행에도 출력이 같은지(`do_sample=false`) 확인한다.
5. **정답 작성과 평가**
   - 정답 파일을 작성한다. 작성자, split, 날짜를 기록한다.
   - `evaluate_run.py`로 채점한다.
6. **metric 단계로 확장** (별도 Phase)
   - 바닥 기준점으로 Homography를 만든다(fit 점과 holdout 점 분리).
   - 오차를 보고한다.
   - `CALIBRATED_GROUND_MAP`을 구현하고 검증한다.

## 4. 현재까지의 측정 (개발 이미지, 정량 평가 아님)

자세한 내용은 `docs/VERIFICATION_2026-11-15.md`에 있다.

- 컨베이어, 트럭, 빨간 큐브: 검증을 통과해 채택됐다. 큐브 박스는 HSV, 전체 이미지 DINO, ROI DINO 세 출처가 일치했다.
- 하역 영역: VLM이 트럭 위쪽 바닥(로봇 작업 공간 밖)만 타당하다고 판정했다. 결과는 `SEMANTIC_REGION_INACCESSIBLE`이고 경로 B는 만들지 않았다.
- 사람이 만든 정답이 아직 없어서 정확도는 측정하지 않았다(UNKNOWN).

## 5. v0.2 추가 사항 (2026-11-17)

- **정답 작성**
  - 하역장과 배치 영역 정답은 GUI의 `정답:` 유형으로 그리고 `data/reference/`에 저장한다.
  - 정답을 작성하기 전에는 판정을 보류한다(`NO_GT`).
- **목적지 체인 평가**
  - 의미 영역 IoU와 배치 영역 IoU를 각각 기록한다.
  - 단계별 실패 코드의 분포도 기록한다(`NO_LOADING_REGION_FOUND`, `PLACEMENT_REGION_EMPTY`, `ROBOT_APPROACH_*`).
- **ablation**
  - `--set planning.destination.candidate_sources=...`로 기하 띠만 쓴 조건과 구조 후보를 더한 조건을 같은 장면에서 비교한다.
  - v0.1 전체 동작은 `archive/v0.1_2026-11-15`로 재현한다.
- **선행 조건**
  - 도달 거리를 실측하기 전에는 접근·경로 결과를 "가정 조건부"로만 보고한다.
  - 실측 후 `handoff_geometry`를 채우고 `goal_selection.*.max_dist_px`를 그 값으로 바꾼다. 이렇게 바꾸면 새 버전으로 취급한다.
- **현재 개발 장면 결과**: `docs/VERIFICATION_v0.2_2026-11-17.md`
