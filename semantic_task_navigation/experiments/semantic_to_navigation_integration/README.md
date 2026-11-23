# Semantic → Navigation 통합 프로토타입 (2026-11-18)

VLM이 영상과 자연어 지시만으로 제안한 **의미적 목적지**를, 검증을 거쳐 기존 v0.2의 **배치 영역 → 로봇 접근 목표 → A\* 경로** 체인에 연결한다. 결과와 해석: [REPORT_2026-11-18.md](REPORT_2026-11-18.md)

> 이 프로토타입은 이미지 좌표 기반 **계획 검증**이다. 로봇 작업 성공을 주장하지 않으며(`robot_task_success = NOT_CLAIMABLE`), 하드웨어 명령을 만들거나 보내지 않는다. 도달 거리(140/180 px)는 측정값이 아닌 가정이다.

## 구조 — v0.2는 수정하지 않고 상속한다

`integration.py`의 `IntegrationRun`이 v0.2 `AnalysisRun`을 **상속**한다. 아래 5개 메서드만 덮어쓰고 나머지(객체 탐지, 큐브 확인, 장면 그래프, 지도 정합성 검사와 지도 생성, 접근 목표, A\*, 경로 검증, 인계 시뮬레이션, 결과 저장)는 v0.2 코드를 그대로 쓴다.

| 덮어쓴 메서드 | 이유 (v0.2 코드에서 확인) |
|---|---|
| `_parse_task` | `task_parser.build_task`가 목적지를 "트럭에 앵커된 하역장"으로 고정한다(L111–119, 아니면 `UNSUPPORTED_DESTINATION`). 그대로 쓰면 정답이 작업에 들어간다. 대상(빨간 큐브)만 v0.2의 `rule_based_parse`로 해석하고 목적지는 열어 둔다 |
| `_analyze_scene` | 출력을 어떤 단계도 쓰지 않고, 프롬프트에 `task_summary`("destination=loading_area near truck")가 들어간다. 실행하지 않는다 |
| `_infer_destination` | 트럭 + 후보 5개 판정으로 고정되어 있다. VLM 자유 제안(STEP 3) → 검증(STEP 4) → 채택/보류(STEP 5)로 대체 |
| `_assess_placement` | 설정의 제외 클래스에 콘이 없다. 검증된 콘을 추가 장애물로 `placement_region`에 넘긴다 (함수는 그대로) |
| `_missing` | `assess_placement`가 지도 없이도 실행될 수 있다. 정합되지 않은 지도에서는 모든 계획 단계를 막는다 |

모델 호출 로그 연결: `Tools.client`를 `AnalysisRun.client`(v0.2 `TrackedClient`)로 바꿔 끼워 의미 단계 호출도 `analysis/model_calls.jsonl`에 남는다.

## 파이프라인 ↔ 코드

| STEP | 하는 일 | 코드 |
|---|---|---|
| 1, 2 | 지시문(L2)과 이미지 | `config.yaml`(동결된 NLA L2 문장과 일치 확인), v0.2 `ImageSource` |
| 3 | VLM 자유 제안 (이미지 + 지시문만 전달, 동결 프롬프트 n0) | `Tools.propose` → `natural_language_abstraction/nla_probe.run_level` |
| 4 | 제안과 근거 검증 (V1–V5) | `verify.py` + v0.2 탐지·판정·`grounding` 함수 |
| 5 | 채택 또는 보류 | `verify.decide` |
| 6 | 배치 가능 영역 | v0.2 `semantics.placement.placement_region` |
| 7 | 로봇 접근 목표 | v0.2 `navigation.goal_selector.select_goal` |
| 8 | A\* 경로 + 독립 검증 | v0.2 `planner.plan`, `path_validator.validate_path` |
| 9 | 시각화 | `visualize.render` (+ v0.2가 만드는 그림 11장) |

### 검증 V1–V5 (STEP 4)

| 검사 | 내용 | 상태 값 |
|---|---|---|
| V1 | 제안이 이 이미지의 유효한 영역인가 (범위, 최소 크기, 장면 크기가 아님, 전체 화면 거부) | VERIFIED / FAILED |
| V2 | 근거로 든 물체가 실제로 있는가. 앵커(트럭)를 **인용**해야 하고 존재해야 한다. 인용하지 않은 겹침은 근거가 아니다 | VERIFIED / FAILED, 콘은 NOT_CONFIRMED(차단 안 함) |
| V3 | 검증된 앵커와 기하학적으로 연결되는가 (v0.2의 `structure_*` 임계값 그대로) | VERIFIED / FAILED |
| V3b | 공간 정밀도. 모델 자신의 근거 문구를 grounding해 박스를 줄이거나, 트럭의 2.5배 이하여야 한다 | VERIFIED(줄임) / ESTIMATED / FAILED |
| V4 | v0.2 `region_judge`가 "yes" (+ 음성 대조 3–4개 기록). **같은 모델 계열이라 독립 검증이 아님** | ESTIMATED / FAILED |
| V5 | 다른 설비·충돌하는 영역·통행/금지 구역을 부적절하게 포함하지 않는가 (지도 검사는 정합된 경우만) | VERIFIED / ESTIMATED(지도 없음) / FAILED |

모든 필수 검사를 통과해야 v0.2 체인으로 넘어간다. Z06이나 사람이 그린 영역은 어디에도 쓰지 않는다. 지도 레이어는 제안이 나온 **뒤에** 독립 제외 검사로만 쓰고 VLM에는 보이지 않는다.

## 파일

| 파일 | 역할 |
|---|---|
| `config.yaml` | 지시문, 이미지, 검증 임계값(이 프로토타입의 가정값은 여기에 있다) |
| `integration.py` | `IntegrationRun`, `Tools`, 통합 기록 생성, CLI |
| `verify.py` | 검증 로직 (순수 함수) |
| `visualize.py` | STEP 9 오버레이 (저장된 JSON + 이미지로만 그린다) |
| `test_integration_offline.py` | GPU 없이 도는 검사 18건 |
| `results/<timestamp>/` | 실행마다 새 폴더 |

실행 폴더: v0.2 표준 구성(`input/ analysis/ planning/ figures/ handoff/ result.json manifest.json`)에 아래가 더해진다.
`vlm_proposal/`(VLM 입력 프롬프트, 원문 응답, 구조화 응답, 오버레이) / `verification/`(검사 기록, 근거 grounding 시도, 콘 크롭, 판정 입력 이미지) / `integration_result.json` / `integration_overlay.png` / `integration_manifest.json`. **VLM 출력은 `vlm_proposal/`에, Python·탐지기 검증은 `verification/`에 분리되어 있다.**

## 실행

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
$PY -m pytest -q experiments/semantic_to_navigation_integration/test_integration_offline.py     # GPU 불필요
$PY experiments/semantic_to_navigation_integration/integration.py run --image-id DEV_REF         # 또는 E1, REF_P01
# 진단용 변형 (config.yaml은 수정하지 않고, 기록과 오버레이에 VARIANT로 표시된다):
$PY experiments/semantic_to_navigation_integration/integration.py run --image-id E1 --set verification.max_unrefined_area_ratio=6.0
```

## 실행 기록

기준 실행 (코드 최종본, 의미 단계 모델 호출까지 로그에 포함):

| 이미지 | run | 상태 |
|---|---|---|
| DEV_REF (v0.2 개발 이미지, 지도 정합) | `20261118_202916_fbe7` | `SEMANTIC_NOT_VERIFIED` |
| E1 (새 사진, 지도 정합 X) | `20261118_203058_46d8` | `SEMANTIC_NOT_VERIFIED` |
| REF_P01 (지도 기준 이미지, 정합) | `20261118_203247_fd48` | `SEMANTIC_EVIDENCE_INSUFFICIENT` |
| E1 변형 (정밀도 한계 6배, 진단용) | `20261118_203426_260d` | `MAP_NOT_ALIGNED` (의미 채택, 계획 중단) |

기록 누락 버전(대체됨): `20261118_201633_9d09`, `…_201916_ab90`, `…_202120_7563`, `…_202347_2a59`. 의미 단계의 모델 호출이 `analysis/model_calls.jsonl`에 빠져 있다(호출 자체와 결과 파일은 정상). 위 기준 실행과 VLM 응답 원문, 검사 상태가 모두 같다. 지우지 않고 남겨 둔다.
