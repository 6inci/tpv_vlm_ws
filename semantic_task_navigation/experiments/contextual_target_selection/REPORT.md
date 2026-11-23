# 맥락 기반 출발·배치 대상 선택 (E1, 2026-11-19)

근거: `results/20261119_152645/` (Qwen 3회, DINO 0회, 56 s). 모델 없이 입력만 점검한 실행: `results/20261119_152635_dryrun/`. v0.2 코드 지문은 실행 전후 같고, 이 폴더 밖에서 수정된 파일은 없다.

## 이전 실패의 원인 (코드와 실제 입력에서 확인)

| 실패 | 원인 | 이번 수정 |
|---|---|---|
| F1 팔레트가 최종 선택에서 사라짐 | 역할 분석이 작업지시 없이 꽉 찬 crop만 보고 "grate, 받침 아님"을 반환했고, `numbered_set`이 이 이름표로 후보를 걸렀다 | 이름표로 거르는 단계를 없앴다. 역할은 작업지시와 구역 확대 이미지로 판단한다 |
| F2 T1 판단 보류 | 출발·목적을 한 응답으로 물어, "큐브가 안 보임" 판단이 둘 다 null로 만들었다 | 출발과 목적을 별도 호출로 묻고 상태도 따로 관리한다 |
| F3 후보 누락 | 탐지 점수 기준 예산으로 후보를 잘랐다 | 구역 안 후보를 모두 번호로 보여 준다 (목적지 17개, 출발지 13개) |

## 결과

| 항목 | 결과 |
|---|---|
| 출발 대상 | `cand_r3` 선택 (규칙상 `SELECTED`). 단, Qwen은 #3·#5·#11(cand_r3, cand_r6, cand_r1)을 모두 "red cube, object_to_move"로 같게 평가하고 첫 번호를 골랐다. **사실상 동률이며 판별이 아니다.** AI 예비 라벨(cand_r1)과 다름 |
| 목적 구역 | 저장된 T1 해석 "출하 구역", 기준 설비 흰 트럭 → 저장된 검증 객체 `truck_1`에 맞춤(IoU 0.95) → v0.2 ROI [1303, 0, 1832, 294] px |
| 배치 대상 후보 | 구역 후보 17개를 모두 제시. 팔레트(#15, m28)도 포함 → **전달 성공**. 구역 전체를 덮는 박스 1개(m86)는 기록만 하고 번호를 붙이지 않음 |
| 최종 배치 대상 | **없음 (`UNVERIFIED`)**. 구역 호출은 #5(트럭 짐칸 위 상자, Qwen 표현 "truck bed")를 골랐지만, 주변을 포함한 확인 호출이 "unsuitable"로 답함 |
| 대상 선택 근거 | 트럭, 짐칸 = suitable("출하용 지정 위치"), 도로 = possible, 콘 = unsuitable, **팔레트 #15 = "manhole cover", unsuitable("출하용이 아님")**. 확인 호출의 이유는 "red cube가 아니다"로, 질문을 잘못 읽은 것으로 보인다 |
| 경로 A | 생성됨. cand_r3 접근 257.0 cm, 간격 72.9 cm, 전신 검사 통과 (출발 대상이 동률 중 임의 선택이라 조건부) |
| 경로 B | 선택된 목표가 없어 확정 경로 없음. 진단으로 `UNVERIFIED` 후보(#5)까지 계산하면 182.3 cm, 간격 68.6 cm. 이전 빈 바닥 목표는 162.1 cm, 38.0 cm |
| 실패·미검증 | 아래 |

## 최초 실패 단계와 직접 원인

**최초 실패 단계는 역할 해석이다.** 팔레트는 탐지(IoU 0.85)와 최종 선택 단계 전달까지 통과했다. 그러나 Qwen은 작업지시와 주변 맥락이 함께 주어졌을 때도 이 물체를 맨홀 뚜껑으로 해석했다. 이전 두 실험에서는 같은 물체를 grate로 해석했다. 물체를 내려놓는 받침으로 보지 않았다.

출하 목적지에 대한 Qwen의 판단은 일관되게 "트럭 짐칸"이다. 이것은 작업지시만으로 보면 타당한 해석이다. 사용자 의도(트럭 앞의 팔레트)와 다른 이유는 물체를 팔레트로 인식하지 못했기 때문이다.

남은 원인 두 가지:

1. **물체 정체가 미확정이다.** 세 가지 다른 입력에서 Qwen이 계속 격자 또는 맨홀 뚜껑으로 본 물체가 실제로 팔레트인지 사람의 확인이 필요하다. 격자 덮개라면 Qwen의 판단은 오류가 아니다.
2. **출발 물체를 판별하지 못한다.** 빨간 물체 세 개를 모두 "red cube"로 평가했다. 기존 후보 보존 실험의 판정(`AMBIGUOUS`)과 같다.

## 기록할 점

- 확인 호출의 응답 이유("red cube가 아님")를 보면, 질문의 "옮긴 물체를 내려놓을 대상"을 "옮길 물체"로 잘못 읽은 것으로 보인다.
- 경로 B는 규칙상 `SELECTED`일 때만 계산해야 하는데, 코드가 `UNVERIFIED` 후보로도 계산했다. 작업계획 상태는 `NOT_CONFIRMED`이므로 결론은 바뀌지 않는다. 이 값은 진단으로만 기록한다.
- 이 결과는 E1 한 장에 대한 것이며 일반화를 주장하지 않는다. 기준 박스는 AI가 그렸고, 큐브 라벨은 AI 예비 라벨이다.

## 재현

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
$PY experiments/contextual_target_selection/select_targets.py              # Qwen 3회 (GPU)
$PY -m pytest -q -p no:cacheprovider experiments/contextual_target_selection/test_cts_offline.py   # 5 tests
```

| 파일 | 내용 |
|---|---|
| `inputs/*_numbered.png`, `inputs/destination_context_verify.png` | 실제 모델 입력 |
| `model_calls.json`, `prompts.json` | 원문 응답 3개, 프롬프트 |
| `context_and_zones.json` | 재사용한 해석, 기준 설비, 구역, 번호 매핑 |
| `selection.json` | 후보별 역할, 결정, 이전 역할 분석과의 충돌 |
| `routes.json`, `selection_and_routes.png` | 경로 A·B, 이전 경로 비교 |
| `evaluation.json`, `validation.json` | 평가 (결정 이후에 기록), 검증 |
