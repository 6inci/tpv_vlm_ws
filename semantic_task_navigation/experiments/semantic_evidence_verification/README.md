# Semantic Evidence Verification — V2 (2026-11-18)

VLM이 자율적으로 제안한 의미적 목적지를 **근거, 공간 관계, 작업 목적**으로 검증하는 독립 검증기(V2)와, 기존 트럭 중심 검증기(V1)와의 비교. 결과와 해석: [REPORT_2026-11-18.md](REPORT_2026-11-18.md)

> 목표는 통과율이 아니라, 근거 없는 판단이 이동로봇의 행동으로 이어지지 않게 하면서 VLM의 자율적 제안을 대체하지 않는 것이다. 이 폴더는 이미지 좌표 기반의 **오프라인 검증**이며, 모델을 호출하지 않고 경로계획도 실행하지 않는다.

## V2의 구조 — 서로 독립된 다섯 축

| 축 | 질문 | 사용하는 것 | 사용하지 않는 것 |
|---|---|---|---|
| A 근거 | VLM이 근거로 든 물체가 실제로 관찰되는가 | 탐지기 관찰(v0.2 탐지, DINO + Qwen 크롭 검증). 지원하지 않는 근거는 `UNVERIFIABLE` | VLM의 설명문, 기능 이름, 신뢰도 |
| B 공간 관계 | 검증된 근거 물체가 제안 영역 안에 있거나 인접한가 | v0.2의 연관 규칙(포함 비율 ≥ 0.5 또는 간격 ≤ band_depth) | 영역 크기 |
| C 작업 목적 | **지시문**의 목적에 필요한 능력을, 검증되고 관계가 성립하는 근거가 갖는가 | 지시문 키워드 → 목적 표, 클래스 → 능력 표 (둘 다 `config.yaml`의 데이터) | 특정 시설 이름의 요구 |
| D 배치 | 영역 안에서 쓸 수 있는 지점이 검증되었는가 | 검증된 localization(근거 문구 grounding으로 얻은 박스), 지도 충돌(정합된 경우) | **영역 크기** (보고만 함) |
| E 계획 인터페이스 | 기존 계획기가 이 목적지의 장애물을 표현할 수 있는가 | 장애물 모델 범위(탐지된 객체, 어댑터가 넘기는 검증된 객체, 지도) | 목적지를 트럭 유형으로 변환하는 것 |

의미 판정 = A + B + C. D와 E는 그 뒤에 **따로** 판정한다. 최종 상태:

| 상태 | 뜻 |
|---|---|
| `EVIDENCE_NOT_ESTABLISHED` | 시각적 근거가 확인되지 않음 (없음 / 못 찾음 / 확인할 탐지기 없음 / 관찰이 없음) |
| `EVIDENCE_RELATION_INCONSISTENT` | 근거 물체는 있으나 제안 영역과의 공간 관계가 성립하지 않음 |
| `PURPOSE_FIT_UNDETERMINED` | 근거와 관계는 성립하나 작업 목적에 적합한지 확정할 수 없음 (목적 불명, 목적에 필요한 능력을 검증할 수 없음, 다른 기능의 근거만 있음) |
| `SEMANTICALLY_SUPPORTED` | 검증된 근거가 목적에 필요한 능력을 갖고 영역과 관계가 성립함 (의미 판정까지) |
| `…_PLACEMENT_UNVERIFIED` | 의미는 타당하나 배치 가능한 지점이 검증되지 않음 → 의미 후보만 보존, 계획 중단 |
| `…_PLANNER_UNSUPPORTED` | 의미는 타당하나 기존 계획 인터페이스가 이 목적지의 장애물을 표현하지 못함 |
| `…_READY_FOR_PLANNING` | 의미 + 배치 근거 + 인터페이스 지원이 모두 확인됨 (지도 정합 등 **계획 전제조건**은 따로 `blocked_by`에 기록) |

검증기는 VLM의 선택을 **바꾸지 않는다**. 후보가 여럿이면 각각 판정하고, 선택은 하지 않는다(`selection: NOT_MADE_BY_THE_VERIFIER`).

사람이 정한 사전 지식은 코드가 아니라 `config.yaml`에 데이터로 있다 (근거 클래스 사전, 목적 표). 새 차량 클래스나 새 목적은 코드 수정 없이 추가된다(합성 시나리오 S6). 조정한 임계값은 없다: 관계 임계값은 v0.2의 `config/planning.yaml`을 읽고, 다른 수치는 V1이 이미 쓰던 값이다.

## 파일

| 파일 | 역할 |
|---|---|
| `config.yaml` | 저장된 사례 목록, 근거 클래스 사전, 목적 표, 배치 설정, 레거시 V1 크기 기준의 출처 |
| `verifier2.py` | V2 본체 (순수 함수, 관찰을 입력으로 받음) |
| `observations.py` | 저장된 통합 실행(VLM 원문 응답, v0.2 탐지, 지도 기록)을 V2의 입력으로 읽음. 원본은 수정하지 않음 |
| `v1_replay.py` | V1의 게이트를 같은 순수 함수로 오프라인 재생. 저장된 V1 결과와 일치하는지 검사 |
| `scenarios.py` | 저장 기록에 없는 상황의 합성 시나리오 10개 (손으로 입력, 기대는 실행 전에 작성) |
| `compare.py` | V1 대 V2 비교, 표(`comparison.md/json`), 사례별 오버레이 |
| `test_verifier2_offline.py` | GPU 없이 도는 검사 30건 |
| `results/<timestamp>_both/` | 실행 결과 (새 폴더, 덮어쓰기 없음) |

## 실행

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
$PY -m pytest -q experiments/semantic_evidence_verification/test_verifier2_offline.py
$PY experiments/semantic_evidence_verification/compare.py --stage both     # stored | synthetic | both
```

## 보존

V1(`../semantic_to_navigation_integration/`)과 v0.2(`src/`, `config/`), 저장된 실행, 두 실험의 동결 프롬프트는 수정하지 않는다. V1의 `max_unrefined_area_ratio`는 비교를 위해 그 설정 파일에서 읽어 `legacy_v1_size_gate`로 보고만 한다.
