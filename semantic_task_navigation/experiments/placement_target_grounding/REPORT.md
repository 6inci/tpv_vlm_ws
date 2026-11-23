# 작업 조건 기반 배치 대상 발견·위치 특정 (E1, 2026-11-19)

- 근거 실행: `results/20261119_143950_ptg/` (모델 호출 포함). 모델 없이 crop과 프롬프트만 점검한 실행은 `results/20261119_143940_dryrun/`.
- 모델: 기존 Qwen3-VL-4B-Instruct(`ebb281ec70b0`), Grounding DINO tiny(`a2bb814dd30d`). 설치·다운로드·환경 변경 없음. 순차 실행, GPU에는 한 모델만 올림(Qwen 최대 8.9 GiB, DINO 1.97 GiB).
- v0.2 코드 지문 `254cfb31da7f…`는 실행 전후 같고, 이 폴더 밖에서 수정된 파일은 없다.
- 평가 기준(회색 팔레트)은 `evaluation_reference.yaml`에만 있다. 모델 호출 전에 사용자 설명을 바탕으로 AI가 그린 박스 [1593, 252, 1640, 287] px이며, 사용자 확인이 필요하다. 이 파일은 모든 모델 출력과 결정을 디스크에 기록한 뒤 평가 단계에서만 읽었다.

## 0. 결론

**배치 대상 발견에 실패했다.** Qwen은 전체 이미지와 구역 crop 두 조건 모두에서 후보를 하나도 내지 않고 `cannot_determine`으로 답했다. 두 조건 모두 "빨간 큐브(와 컨베이어)가 보이지 않는다"를 이유로 들었다. 열린 질문에서 Qwen은 놓을 곳을 고르지 않고 판단을 보류했다. 그래서 그 뒤의 탐지·선택·경로 단계는 사전 규칙대로 실행되지 않았다(`AMBIGUOUS`, 경로 B `NOT_RUN`).

진단 조건에서 나온 정보는 다음과 같다. 모두 결정에는 쓰지 않았다.
- **탐지는 확대하면 가능하다.** 일반 물류 어휘("pallet" 포함)로 DINO를 돌리면, 구역 crop ×2에서 기준 물체 위치(IoU 0.85)가 "pallet" 프롬프트 3위(점수 0.34)로 나온다. 전체 이미지에서는 같은 프롬프트로 35위(0.145)라서 상위 15개 제한이 있으면 사라진다. 그런데 "cardboard box", "container", "cart" 같은 다른 단어도 같은 위치를 3–6위로 잡는다. DINO는 "무언가 있다"는 것은 찾지만 그것이 무엇인지는 구분하지 못한다.
- **선택만 되면 경로는 성립한다.** 기준 박스를 목표로 했다고 가정하면 경로 B는 202.1 cm로 생성되고, 차체와 대상 사이 간격은 6.2 cm다. 이전 빈 바닥 목표의 38.0 cm보다 31.8 cm 작다. 이 값은 정답을 넣고 계산한 진단값이며 시스템의 결과가 아니다.

## 1. 단계별 결과

| 단계 | 질문 | 결과 |
|---|---|---|
| Discovery | Qwen이 배치 대상을 스스로 언급했는가 | **FAIL**: 두 조건 모두 후보 0개, `cannot_determine` |
| Detection | DINO가 대상 박스를 만들었는가 | **본 실험 NOT_RUN** (Qwen이 만든 문구가 없음). 진단 어휘 조건에서는 IoU 0.85로 검출 |
| Selection | 의미에 따라 후보를 골랐는가 | `AMBIGUOUS` (규칙 4: Qwen이 판단하지 못했고 관련 후보도 없음). 비교 호출 조건을 충족하지 않아 하지 않음 |
| Grounding | 선택한 물체의 위치를 특정했는가 | NOT_RUN (선택 없음) |
| Navigation | 경로 B가 생성되는가 | NOT_RUN (선택 없음). 이전 빈 바닥 경로 B는 같은 코드로 재현됨(233, 83 cm, 162.1 cm) |

`exp5_evaluation.json`의 `Detection: "NO"`는 "본 실험에서 후보가 하나도 없었다"는 뜻이다. 문구가 없어 탐지를 실행하지 않은 경우와 실행했지만 찾지 못한 경우를 코드가 구분하지 않는다. 실행 결과와 코드를 일치시키기 위해 이 실행 후에는 코드를 고치지 않았다.

처음 실패한 단계는 **Discovery**다. 사용자가 구분하자고 한 다섯 경우 중 "모델이 배치 대상을 발견하지 못함"에 해당한다.

## 2. Experiment 1 — 열린 질문 (Qwen 2회)

**프롬프트** (`prompts/exp1.txt`, sha256 `973d999b…`, 두 조건 공통)

```
이동로봇에게 '컨베이어에서 생산된 빨간 큐브를 출하할 수 있는 적절한 장소로 옮겨줘'라는 명령이 주어졌다. 이미지에서 이 작업의 최종 배치 대상으로 적절한 물체나 표면이 무엇인지 판단하라. 관찰되는 시각적 근거를 설명하고, 여러 후보가 있다면 구분하라. 정보가 부족하면 판단할 수 없다고 답하라.

Return JSON only, no prose:
{"candidates": [{"name_en": <short English noun phrase naming the object or surface>, "name_ko": <Korean name>,
                 "bbox_2d": [x1, y1, x2, y2] or null (integers normalized to 0-1000 relative to this image's width and height),
                 "location_in_image": <short text>, "visual_evidence": [<what is visible>],
                 "suitability": "suitable" | "possible" | "unsuitable", "reason": <one short sentence>}],
 "selected": <name_en of the most appropriate candidate, or null>,
 "decision": "selected" | "cannot_determine",
 "uncertain": [<what cannot be determined from this image>]}
```

장면의 물체·장소 이름(팔레트, 콘, 트럭, 하역장, 바닥 등)은 넣지 않았다(테스트로 확인).

| 조건 | 입력 | 입력/출력 토큰 | 지연 |
|---|---|---|---|
| A 전체 이미지 | 1920×1080 | 2,329 / 75 | 4.3 s |
| B 구역 crop | 저장된 Qwen `area_5` 상자 + 여백 max(0.25 × 긴 변, 120 px) → [1104, 0, 1920, 425] px, ×2 확대(1632×850), 주석 없음 | 1,666 / 69 | 2.5 s |

**A 원문** (`qwen/exp1_A_full_image_raw.txt`)
```json
{
  "candidates": [],
  "selected": null,
  "decision": "cannot_determine",
  "uncertain": [
    "The image does not show any red cube produced on a conveyor belt.",
    "There is no clear indication of where the red cube should be moved to, as the target location is not specified in the image."
  ]
}
```

**B 원문** (`qwen/exp1_B_zone_crop_raw.txt`)
```json
{
  "candidates": [],
  "selected": null,
  "decision": "cannot_determine",
  "uncertain": [
    "The image does not contain any red cubes or conveyer belt.",
    "There is no clear indication of where the red cube should be placed or what surface is designated for shipment."
  ]
}
```

- 두 조건의 판단은 일관된다. 둘 다 후보 0개, 판단 보류다.
- **해석** (INFERRED): Qwen은 "옮길 큐브가 보이지 않음"을 "놓을 곳을 판단할 수 없음"의 근거로 썼다. 큐브는 전체 이미지에서 약 17×15 px이고, crop에는 컨베이어가 들어 있지 않다. 이전 L2 프롬프트(기능 영역을 나열하고 목적지를 제안하라는 질문)에서는 같은 상황에서도 `area_5`를 제안했다. 이번 질문의 "정보가 부족하면 판단할 수 없다고 답하라"가 보류를 허용했고, Qwen은 그것을 택했다. 이 결과를 "Qwen은 배치 대상을 볼 수 없다"는 능력 판정으로 읽어서는 안 된다. 이 프롬프트에서의 결과다.
- crop 경계: 저장된 객체(트럭, 콘, 컨베이어) 중 잘린 것은 없다. 왼쪽 가장자리의 초록 차단막은 일부 잘렸다(저장된 탐지 객체가 아님).

## 3. Experiment 2 — DINO 후보 탐지 (DINO 2회, 모두 진단)

- **본 실험**(Qwen 문구 기반, 전체 / ROI / ROI ×2): Qwen이 낸 이름이 없어 사전 규칙상 문구가 0개였고, 실행하지 않았다. 낮은 임계값 진단도 Qwen 문구를 쓰므로 실행되지 않았다.
- **진단 어휘 조건**(사전에 고정한 일반 물류 어휘 8개 `pallet, truck bed, cardboard box, shelf, table, cart, container, floor marking`, 임계값 0.10): 기준 유형("pallet")이 들어 있으므로 열린 발견의 성공으로 평가하지 않는다.

| 조건 | 상자 수 (보존) | 기준 위치 검출 |
|---|---|---|
| 전체 이미지 | 290 (282 KEPT, 8 FILTERED_OVERSIZE) | "pallet" 53개 중 35위, 점수 0.145, IoU 0.84. 조건 전체 순위 162위 |
| 구역 crop ×2 | 108 (77 KEPT) | "pallet" 20개 중 3위, 0.34, IoU 0.85. "container" 3위 0.309, "cardboard box" 3위 0.288, "cart" 4위, "shelf" 5위, "table" 6위, "truck bed" 3위 |
| 두 조건 병합 | 68개 후보, 구역 안 18개 | 기준 위치 후보 `m6`는 구역 안 4위 (1–3위: 트럭, 초록 차단막, 트럭 적재물) |

- **순위 제한**: 상위 K개로 자르지 않고 임계값을 넘은 상자를 순위와 함께 전부 저장했다. 전체 이미지만 보고 상위 15개로 잘랐다면 기준 물체는 빠졌다. EXP111 `small_pallet`과 같은 양상이다.
- **확대의 효과**: 같은 프롬프트("pallet")에서 순위가 35위에서 3위로, 점수가 0.145에서 0.34로 올랐다.
- **문구의 구분력**: 같은 박스가 어휘 8개 중 7개 프롬프트에서 검출됐다. DINO 문구는 물체의 종류를 정하는 근거가 되지 못한다.

## 4. Experiment 3·4 — 선택과 경로 B

- **선택**: 사전 규칙 4("Qwen이 판단하지 못했고 관련 후보가 없음")에 따라 `AMBIGUOUS`다. 비교 호출은 관련 후보가 1개 이상 있을 때만 하도록 정했으므로 하지 않았다. 진단 어휘 후보는 선택에 쓰지 않는다고 미리 정했으므로 비교 호출에도 넣지 않았다.
- **구역과 대상의 구분**: 구역 수준(`area_5`, 저장된 Qwen 응답)은 유지되지만, 그 안의 대상은 선택되지 않았다.
- **경로 B**: 선택이 없어 새로 생성하지 않았다. 이전 빈 바닥 목표는 같은 코드로 재현됐다(233, 83 cm, 162.1 cm, 간격 38.0 cm).

**진단: 기준 박스를 대상으로 했다면** (평가 단계에서만 계산, `reference_oracle_route_DIAGNOSTIC`)

| 항목 | 이전 빈 바닥 목표 | 기준 박스(진단) |
|---|---|---|
| 정지 위치 | (233, 83) cm | (273, 83) cm |
| 경로 B | 162.1 cm | 202.1 cm |
| 차체–대상 간격 | 38.0 cm | 6.2 cm (접촉점까지 5.9 cm) |

- 기준 물체의 바닥 접촉점(박스 아래 가장자리 중앙)은 (292.2, 64.7) cm로, 어느 DRAFT 구역에도 속하지 않는다. 주행 구역 Z02와의 거리는 1.2 cm다. 박스 중심을 투영한 점과는 4.4 cm 떨어져 있어, 높이에 따른 시차 정도를 가늠할 수 있다.
- 금지 구역이나 차체 크기는 바꾸지 않았다. 이 값은 "선택만 되면 경로·간격 측면에서는 성립한다"는 근거일 뿐, 시스템이 낸 결과가 아니다.

## 5. 모델 호출

| 모델 | 횟수 | 내용 |
|---|---|---|
| Qwen3-VL-4B | 2 | Exp 1 A, B. Exp 3 비교 호출 0회(조건 미충족) |
| Grounding DINO | 2 | 진단 어휘: 전체, 구역 ×2 (각 8개 프롬프트). 본 실험 0회 |

전체 소요 25.4 s. 호출마다 프롬프트·이미지 해시, 토큰 수, 지연, VRAM, 원문을 `model_calls.json`에 남겼다. DINO 원출력은 `dino/`에 있다.

## 6. 미검증·한계

- 기준 박스는 AI가 사용자 설명으로 그린 것이다. 사용자 확인이 필요하다.
- 기준 물체가 무엇인지(팔레트인지, 격자 덮개인지)는 영상으로 확정되지 않았다. 표면이 격자무늬이고, 이전 Qwen 응답은 화면 아래 중앙의 비슷한 물체를 "grate"라고 불렀다.
- 사진 1장, 프롬프트 1개의 결과다. 비율이나 일반화로 말할 수 없다.
- 그 물체 위에 큐브를 놓을 수 있는지(높이, 표면, 그리퍼)는 이 실험의 범위가 아니다.

## 7. 다음 실험에서 풀어야 할 문제

1. **과제 해석 문제 (Discovery)**: Qwen이 "큐브가 안 보이면 목적지도 판단 불가"로 처리했다. 정답을 알려주지 않으면서 과제를 분리하는 프롬프트를 시험해야 한다. 예를 들어 "로봇은 이미 큐브를 들고 있다. 이 장면에서 출하를 위해 큐브를 내려놓을 물체나 표면을 고르라"처럼 묻는다. 이것은 새로 설계하는 조건이므로 별도 승인 후 사전 선언해서 진행한다.
2. **선택 단독 검증 (Selection)**: 구역 안 DINO 후보 18개(진단 어휘)에 번호를 붙여 Qwen에 보여 주고, 기준 물체를 고르는지 본다. 발견과 분리된 선택 능력을 볼 수 있다. 후보 생성에 "pallet"이 들어간 어휘를 썼다는 점은 기록해 둔다.
3. **탐지 문구의 구분력**: DINO는 위치만 주고 종류는 정하지 못한다. 종류 판단은 crop 확인(Qwen)으로 해야 한다.
4. 사용자 확인: 기준 박스와 물체 종류.

## 8. 현재 연구 시스템의 검증 수준

| 단계 | 상태 |
|---|---|
| 출하 구역 추론 (`area_5`) | 저장된 결과 기준 CONDITIONAL (구역 수준만) |
| 구역 안 배치 대상 발견 | **FAIL** (이번 프롬프트) |
| 배치 대상 위치 검출 | 진단 조건에서만 가능함을 확인 (확대 필요, 종류 구분 불가) |
| 배치 대상 선택 | 미검증 (입력 없음) |
| 선택 대상까지 경로 B | 진단 계산상 가능 (202.1 cm, 간격 6.2 cm). 시스템 결과는 없음 |

지금 시스템은 "어느 구역으로 가는가"까지는 답하지만, "그 안의 무엇 위에 놓는가"는 아직 답하지 못한다. 막힌 지점은 경로나 탐지가 아니라 VLM의 과제 해석이다.

## 9. 재현

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
$PY experiments/placement_target_grounding/placement_target.py --dry-run     # crop, 프롬프트만 (모델 없음)
$PY experiments/placement_target_grounding/placement_target.py               # Qwen + DINO (GPU, 약 30초)
$PY -m pytest -q -p no:cacheprovider experiments/placement_target_grounding/test_placement_offline.py   # 12 tests
```

| 파일 | 내용 |
|---|---|
| `config.yaml` | 프롬프트, crop 규칙, 문구 규칙, 선택 규칙 (모델 호출 전 선언) |
| `evaluation_reference.yaml` | 평가 기준 (평가 단계 전용) |
| `qwen/*_raw.txt`, `model_calls.json`, `prompts/` | 모델 원문, 호출 기록, 프롬프트 |
| `exp1_open_reasoning.json` | Exp 1 파싱 결과 |
| `exp2_candidates.json`, `dino/` | 후보 전부(순위·점수·상태), 진단 결과, DINO 원출력 |
| `exp3_selection.json` | 결정과 규칙 |
| `exp4_route_B.json` | 이전 경로 B 재현, 새 경로 B(미실행 사유) |
| `exp5_evaluation.json` | 단계별 평가, 진단 경로 |
| `figures/01–05` | Qwen 응답, DINO 후보, 선택, 경로 비교, 평가(기준 박스는 05에만 표시) |
