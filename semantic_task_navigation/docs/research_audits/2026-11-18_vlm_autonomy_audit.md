# VLM 자율 의미 추론 감사 — Semantic Task Navigation v0.2 (2026-11-18)

- 성격: **읽기 전용 분석.** 소스, 설정, 실행 결과, 기존 문서(`ARCHITECTURE.md`, `VERIFICATION_*`, `CHANGELOG.md` 등)는 수정하지 않았다. 이 파일만 새로 만들었다.
- 모델 실행, 설치, 재실행은 하지 않았다. 저장된 실행 기록과 코드, 설정만 읽었다.
- 표기: `CONFIRMED` 코드·실행 기록으로 직접 확인 / `INFERRED` 확인한 사실에서 이어지는 해석 / `UNKNOWN` 확인하지 못함
- 경로 기준: `/home/binci/vlm_ws/semantic_task_navigation/` (이하 `STN/`). 줄 번호는 2026-11-18 현재 파일 기준이다.
- 분석 대상 코드 지문: 가장 최근 실제 모델 실행 `runs/20261117_111627_d4cf`의 코드 해시 `254cfb31…`. `src/`와 `config/`의 마지막 수정(11-17 11:15)이 이 실행 종료(11:17:37)보다 앞서므로, 이 보고서가 읽은 코드는 그 실행이 쓴 코드와 같은 것으로 본다 (`INFERRED`, mtime 기준. 해시를 다시 계산하지는 않았다).

---

## 0. 먼저 읽을 결론

1. **VLM에 지도, 노란 경계, Z06, 접근성, 도달 거리를 준 흔적은 없다.** 실제 모델에 전달된 프롬프트 9종을 설정에서 재구성해 실행 기록의 SHA-256과 대조했고 전부 일치했다. 판정 입력 이미지에는 흰 "T" 박스와 초록 후보 윤곽만 그려져 있다 (`CONFIRMED`, §2).
2. **그러나 "의미 추론"의 핵심 부분이 사람이 설계 단계에서 정해 둔 것이다.** 목적지 종류(하역장), 기준 물체(트럭), 후보 생성 방식, 질의 문구가 코드에 고정되어 있다. 모델이 하는 일은 사실상 "트럭 주변에서 사람이 만든 후보 5개에 예/아니오를 답하는 것"이다 (`CONFIRMED`, §3).
3. **정답 유출은 모델 입력이 아니라 개발 과정에서 일어났다.** 결정적 후보를 만든 질의 문구("parking space of the truck")는 사람 초안 Z06과의 일치(IoU 0.76)를 보고 골랐다. 이후 보고한 Z06 일치도 0.722는 독립 증거가 아니다 (`CONFIRMED`, §3.2).
4. **개발 이미지가 EXP110의 test 이미지 P06과 같은 파일이다** (SHA-256 일치). 이전 결정("P05–P10은 튜닝에 쓰지 않는다")과 충돌한다 (`CONFIRMED`, §3.3).
5. **v0.2의 VLM 자율 판단은 좁다.** 장면 이해(scene_analysis) 출력은 이후 어떤 단계도 쓰지 않는다. 객체 어휘는 고정되어 있고, 공간 관계와 작업 가능성, 이동 목표, 경로는 모두 Python이 정한다 (`CONFIRMED`, §4).
6. **물리적 실행 가능성은 구현되어 있지 않고, 코드가 정직하게 그렇게 표시한다** (`robot_task_success = NOT_CLAIMABLE`). 다만 "그리퍼 도달"은 이미지 픽셀 거리 가정(180 px)으로 대신하고 있고, 이 장면의 성공 여부가 그 가정에 2.5 px 차이로 걸려 있다 (`CONFIRMED`, §7).
7. 5S 조건 비교(A/B/C)는 연구로서 타당하지만, **현재 v0.2 구조로는 세 조건의 차이를 거의 측정할 수 없다.** 후보가 사람이 만든 5개이고 모델 출력이 예/아니오뿐이기 때문이다 (§6).

---

## 1. 현재 v0.2의 실제 VLM 추론 구조

### 1.1 모델과 호출 방식 (`CONFIRMED`)

| 항목 | 내용 | 근거 |
|---|---|---|
| VLM | Qwen3-VL-4B-Instruct, bf16, `do_sample=false` (greedy) | `STN/config/models.yaml` L11–21 |
| 탐지 모델 | Grounding DINO tiny, box/text 임계 0.10 | `models.yaml` L23–55 |
| 실행 방식 | 별도 프로세스 worker (`qwen_vl` env), GPU 배타 상주 | `STN/src/semantic_task_nav/perception/model_worker.py`, `model_client.py` |
| 메시지 구성 | `[{"role":"user","content":[image?, text]}]` 한 개뿐. **system 역할을 만들지 않는다** | `model_worker.py` `qwen_generate` L80–91 |
| 기본 system 주입 | 없음 (`INFERRED`). HF 캐시의 `chat_template.json`에 기본 system 문구("You are a helpful…")가 없다. 템플릿이 system을 주입하는지 모델로 직접 확인하지는 않았다 | `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*/chat_template.json` |
| 대화 이력 | 호출마다 독립. 이전 응답을 다음 호출에 넣지 않는다 | `qwen_generate`가 매번 새 `msgs`를 만든다 |

### 1.2 실행 단계와 모델 호출 (`transport_and_place`, 실행 `runs/20261117_111627_d4cf`)

`STN/src/semantic_task_nav/agent/orchestrator.py` `PLANS` L43–48, 단계 구현 `_parse_task`~`_create_handoff_request`.

| # | 단계 | 모델 호출 (이 실행의 호출 수) | 입력 | 출력이 이후에 쓰이는가 |
|---|---|---|---|---|
| 1 | parse_task (`_parse_task` L291) | Qwen 1회, **텍스트만** | 명령문 | 예. 단, Python 정규화와 화이트리스트를 통과한 것만 (§3.1) |
| 2 | analyze_scene (`_analyze_scene` L325) | Qwen 1회, 전체 영상 | 영상 + 작업 요약 문장 | **아니오.** 기록만 한다 (§4.3) |
| 3 | detect_objects (`_detect_objects` L340) | DINO 4회 + Qwen 검증 14회 (conveyor 8, truck 3, mobile_robot 3) | 후보별 확대 크롭 | 예 |
| 4 | inspect_target (`_inspect_target` L401) | DINO 1회 + Qwen 검증 8회 | 컨베이어 주변 확대 ROI의 후보 크롭 | 예 |
| 5 | build_scene_graph (`_build_scene_graph` L429) | 없음 | 박스 | 예 (Python 계산) |
| 6 | build_navigation_map (`_build_navigation_map` L495) | 없음 | 수동 지도 | 예 |
| 7 | **infer_destination** (`_infer_destination` L437) | **Qwen 7회**: 구조 grounding 2 + 후보 판정 5 | 전체 영상 / 후보 크롭 | 예 |
| 8 | assess_placement (`_assess_placement` L475) | 없음 | 의미 영역 | 예 (Python) |
| 9–10 | select_pickup_pose / select_dropoff_pose (L521, L534) | 없음 | 지도, 영역 | 예 (Python) |
| 11–13 | plan_path / validate_path / create_handoff_request | 없음 | | 예 (Python) |

- 총 모델 호출 36회 (Qwen 31, DINO 5). `runs/20261117_111627_d4cf/analysis/model_calls.jsonl`.
- 목적지(하역장) 추론에 쓰인 VLM 호출은 36회 중 7회다.

### 1.3 하역장 추론 체인 (`STN/src/semantic_task_nav/semantics/region_grounding.py`)

```text
트럭 박스 (DINO 후보 + Qwen 크롭 검증을 통과한 것)
  ├─ 후보 A: geometric_band 4개 (트럭 박스의 아래/위/왼/오른쪽 띠)          generate_candidates L37   [Python]
  └─ 후보 B: Qwen grounding 박스 ("loading area of the truck",
             "parking space of the truck")  + 트럭과의 기하 연관성 검사      structure_candidates L55  [VLM 박스 + Python 필터]
        ↓
  후보마다 크롭 1장 (흰 T = 트럭, 초록 = 후보) → Qwen "yes/no/uncertain"     judge_candidates L108     [VLM]
        ↓
  yes인 후보 = semantic_regions (unverified)                                 ground_loading_region L138
        ↓   (이후는 모델 없음)
  배치 영역 = 의미 영역 − 트럭 박스 − 지도 장애물·금지구역, 침식            placement.py L27
  접근 영역 = 충돌 없는 로봇 셀 ∩ 도달 거리 안 ∩ 원점에서 도달 가능         goal_selector.py L43
  목표점 = 접근 영역에서 (경로 비용 + 1.5 × 거리) 최소                       goal_selector.py L72
  경로 = A* + 직선화 + 픽셀 단위 검증                                        planner.py, path_validator.py
```

이 실행의 결과 (`analysis/semantic_regions.json`, `analysis/destination_chain.json`):

| 후보 | 출처 | VLM 답 | 이유(모델 원문) |
|---|---|---|---|
| 1 아래 띠 | geometric_band | no | "walkway with cones, not the truck's loading/unloading zone" |
| 2 위 띠 | geometric_band | no | "above the truck's cab, not its cargo bed" |
| 3 왼 띠 | geometric_band | no | "a walkway" |
| 4 오른 띠 | geometric_band | no | "lacks visual cues like designated markings or equipment" |
| 5 구조 박스 | `grounded_structure:parking space of the truck` | **yes** | "encompasses the truck's cargo bed and surrounding floor markings" |

- 구조 grounding 원문: `"loading area of the truck"` → `{"objects": []}` (빈 응답), `"parking space of the truck"` → `[728, 67, 860, 210]` (0–1000 정규화) → 원본 픽셀 `[1397.8, 72.4, 1651.2, 226.8]`.
- 후보 5는 트럭 박스를 완전히 포함한다 (`truck_containment` 1.0, 트럭과 IoU 0.67, 트럭 대비 면적 1.49배).
- 최종: `SEMANTIC_REGION_FOUND` → `PLACEMENT_REGION_FOUND` (4,729 px) → `GOAL_SELECTED` (1316, 372) → 경로 B 696 px 검증 통과 → `PLAN_COMPLETE`.
- 같은 코드로 3회 실행했을 때 의미 판정, 배치 영역, 목표, 경로가 모두 같았다 (`VERIFICATION_v0.2_2026-11-17.md` §1, 기록으로 확인. 3회를 내가 다시 대조하지는 않았다).

---

## 2. VLM에 전달되는 프롬프트와 사전 정보

### 2.1 실제 입력 확인 방법 (`CONFIRMED`)

- 실행 기록에는 프롬프트 **원문이 아니라 SHA-256만** 저장된다 (`model_calls.jsonl`의 `prompt_sha256`, `orchestrator.py` `TrackedClient.call` L96–117). 원문 응답은 보존된다.
- 그래서 `config/models.yaml`의 템플릿에 실제 코드와 같은 방식으로 치환을 적용해 문장을 재구성하고, 그 SHA-256을 실행 기록과 대조했다. 대조한 것은 `runs/20261117_111627_d4cf/analysis/model_calls.jsonl` 36건이다.

| 프롬프트 | 해시 앞 10자리 | 대조 |
|---|---|---|
| task_parse (명령문 치환) | 87910b5381 | 일치 |
| scene_analysis (작업 요약 치환) | a4601f37e1 | 일치 |
| verify_object — conveyor 설명 | aedea2d909 | 일치 |
| verify_object — truck 설명 | 95504eabfb | 일치 |
| verify_object — mobile_robot 설명 | 8127a580f9 | 일치 |
| verify_object — 빨간 큐브 설명 | c27b30e46e | 일치 |
| grounding_fallback "loading area of the truck" | 8edc2bcafb | 일치 |
| grounding_fallback "parking space of the truck" | 17fdebcae8 | 일치 |
| region_judge (치환 없음) | cee41976e0 | 일치 |

- 기록에 있는 해시 중 대조되지 않은 것은 없다. 따라서 **모델에 전달된 텍스트에 `models.yaml` 밖의 정보가 섞이지 않았다**고 판단한다.
- 이 대조는 이 실행 한 건에 대한 것이다. 다른 실행(v0.1 등)은 코드가 달라서 하지 않았다.
- 이미지 입력은 저장된 파일로 확인했다: `analysis/region_judge_cand1.png`~`cand5.png`, `analysis/crops/verify_*.png`, `input/raw_frame.png`.

### 2.2 프롬프트별 전문과 담긴 사전 정보

모두 `STN/config/models.yaml` L76–143.

**① task_parse_v1** (텍스트만)
- 명령문에서 `{status, task_type, target, destination, reason}` JSON을 만들게 한다.
- 사전 정보: 허용 task_type 두 가지(`transport_and_place`, `navigate_to_region`), 지원하지 않는 행동 목록(rotate, stack, sort, open, clean), "목적지가 명시되지 않으면 `missing_information`" 규칙.
- 실제 응답: `"region_type": "unloading_area", "reference_object": "truck"` (명령문의 "트럭이 위치한 하역장"을 옮긴 것).

**② scene_analysis_v1** (전체 영상)
- "작업 환경을 설명하라. 보이는 것에만 근거하라." + 마지막 줄 `Task: {task_summary}`.
- `{task_summary}`로 들어간 문장: `transport_and_place: target=red cube (produced_by conveyor); destination=loading_area near truck` (`orchestrator.py` `task_summary` L902–906에서 만든다). **목적지 종류와 기준 물체가 이미 정해진 문장이다.**

**③ verify_object_v1** (후보 크롭, 객체 확인)
- `The green box … marks one candidate object. Is the object inside the green box {description}?`
- 설명은 사람이 쓴 것이다 (`models.yaml` L51–55). 예: conveyor = "a conveyor or a production line containing a conveyor (robot arms or machines beside the conveyor may also be inside the box)", truck = "a truck (a vehicle with a cab and a cargo bed; a toy or model truck counts)".

**④ grounding_fallback_v1** (전체 영상, 박스 요청)
- `Find every instance of "{query}" in this image.` 질의: `loading area of the truck`, `parking space of the truck` (`config/planning.yaml` L30).

**⑤ region_judge_v1** (후보 크롭 1장, **하역장 판정의 핵심 프롬프트**)

```text
This top-down image of a work floor shows a truck marked by a white box labeled "T" and one candidate area outlined in green.
Is the outlined area part of the loading/unloading area of this truck, i.e. a place where goods are handed over to or from this truck?
Use visible cues such as floor markings, the truck's position and nearby equipment.
Answer "no" if the outlined area is a walkway, a safety zone, an area of other equipment, or unrelated floor.
Return JSON only, no prose:
{"answer": "yes" | "no" | "uncertain", "reason": <one short sentence>}
```

- 사전 정보로 들어간 것:
  - **하역장의 정의** ("물건을 트럭으로 또는 트럭에서 넘겨주는 곳")
  - **기준 물체** (트럭, 흰 "T" 박스)
  - **"아니오"로 답할 범주의 목록** (통로, 안전 구역, 다른 설비 구역, 무관한 바닥)
  - 단서 종류 (바닥 표시, 트럭 위치, 주변 설비)
- 들어가지 않은 것: 지도, 노란 경계, 로봇 통행 영역, Z06 좌표, 접근성, 도달 거리, 정답 위치.

### 2.3 이미지에 그려진 것 (`CONFIRMED`, `render_judge_input` L91–105)

- 흰 사각형 + "T" (트럭), 초록 윤곽 (후보). 그 밖의 오버레이는 없다. 트럭과 후보 주변을 `judge_crop_margin_px`(120 px)만큼 잘라 짧은 변이 아니라 긴 변 기준 896 px로 확대한다.
- 후보 크롭에는 **원본 영상에 실제로 찍힌 것**(점선 주차 구획, 라바콘, 흰 실선 등)이 보인다. 이는 모델이 볼 수 있어야 하는 환경 정보이고 유출이 아니다.
- 후보 5의 크롭에는 트럭이 초록 윤곽 안에 들어 있다 (`region_judge_cand5.png`).

### 2.4 지도와 zone 정보가 쓰이는 위치

| 정보 | 위치 | 모델 입력인가 | 근거 |
|---|---|---|---|
| 통행 가능 영역 Z02 (노란 경계 안), 고정 장애물 Z03·Z04, 금지 Z01·Z07 | `config/maps/reference_scene_map.json` → `NavMap` (`map_builder.py`) | **아니오** | `ground_loading_region`의 인자에 지도가 없다 (L138). 지도는 `_build_navigation_map`, `_assess_placement`, `select_goal`에서만 쓴다 |
| 의미 참조 영역 Z06 (`semantic_reference_zones`) | 같은 지도 파일 | **아니오** | `ground_loading_region(..., prior_zones)`가 의미 판정이 끝난 뒤 IoU만 계산한다 (L172–179, 주석 "evaluation/consistency only; never feeds planning or the model"). `validated_loading_region`은 **확정된** zone과 일치할 때만 채우고, Z06은 `confirmed: false`라 지금은 항상 `null`이다 |
| 로봇 시작점 (700, 740) | 지도의 `start_pose` | 아니오 | 로봇이 검출되지 않을 때만 경로 시작점으로 쓴다 (`_determine_start` L383) |
| 도달 거리 140/180 px, 로봇 반경 35 px | `config/planning.yaml` L5, L18–20 | 아니오 | Python 단계에서만 쓴다 |
| 사람 정답 영역 | GUI `data/reference/*.json` | 아니오 | `tests/test_gui.py::test_human_reference_regions_are_display_only`. 이 프로젝트에는 `data/` 폴더가 아직 없다 (정답 파일 없음) |
| 모델 입력 격리 테스트 | `tests/test_destination_chain.py::test_one_judgement_per_candidate_and_no_map_in_model_input` | – | 후보마다 이미지 1장, 프롬프트가 템플릿과 같음을 단언한다 |

---

## 3. 정답 유출 가능성

판단 기준: 사용자 지침대로 "코드에 지도가 있다"는 것만으로 유출로 보지 않았다. **(가) 실제 VLM 추론에 쓰인 정보인가, (나) 추론 이후 독립 검증에 쓰인 정보인가, (다) 개발 중에 정답을 보고 설계를 맞춘 것인가**로 나눴다.

### 3.1 요약표

| 항목 | 질문 | 판정 | 등급 |
|---|---|---|---|
| 노란 경계가 로봇 주행 영역이라는 정보 | VLM 입력에 있는가 | **없다.** 지도 Z02는 Python의 배치·접근 단계에서만 쓴다 | 유출 아님 (독립 검증용) |
| 특정 위치가 하역장이라는 정보(Z06) | VLM 입력에 있는가 | **없다.** 판정 뒤 일관성 수치 계산에만 쓴다 | 모델 입력 유출 없음 |
| 트럭 주변이 목적지라는 규칙 | 코드에 있는가 | **있다.** 목적지는 "트럭에 앵커된 하역장"으로 고정 (아래 ①) | **설계에 포함된 사전 정보** |
| 공간의 기능적 의미 | 사람이 지정했는가 | 부분적으로 있다. 하역장 정의, "아니오" 범주 목록, 객체 설명문 (아래 ②, ③) | **사전 정보** |
| 개발 이미지에 맞춘 규칙 | 있는가 | **있다.** 구조 질의 문구, 프롬프트 개정, 설명문 개정이 모두 이 한 장면에서 정해졌다 (아래 ④) | **개발 이미지 과적합 위험** |
| 위치별 규칙, 좌표 하드코딩 | 있는가 | 의미 단계에는 **없다.** 지도 좌표는 `config/maps`에 있고 소스에 박혀 있지 않다 | 이상 없음 |
| 특정 설비 근처가 작업 목표라는 규칙 | 있는가 | "트럭 근처"가 그렇다 (①). 빨간 큐브는 "컨베이어 주변 ROI"에서만 찾는다 (`target_inspection.py` `make_roi`) | **사전 정보** |

### 3.2 항목별 상세

**① 목적지의 종류와 기준 물체가 코드에 고정되어 있다** (`CONFIRMED`)
- `STN/src/semantic_task_nav/agent/task_parser.py`
  - `REGION_SYNONYMS` L31–33: 목적지로 허용되는 말은 `loading_area` 한 종류뿐이다 (dock, unloading_zone 등이 모두 여기로 정규화된다).
  - `REFERENCE_SYNONYMS` L34: 기준 물체는 truck 한 종류뿐이다.
  - `build_task` L111–119: 위 둘이 아니면 `UNSUPPORTED_DESTINATION`으로 **작업을 거부한다.** "loading_area grounding needs a truck reference".
- 사용자 명령문 자체도 개념과 기준을 준다: "트럭이 **위치한** **하역장**에 놓아줘" (`config/app.yaml` L4).
- 결과: 시스템은 "어느 시설이 하역장인가"를 장면 전체에서 찾지 않는다. "트럭 주변에서 하역장으로 볼 수 있는 곳이 있는가"만 묻는다. 연구 목표 3번("사용자가 지시한 목적지는 환경의 어느 영역인가")에 대한 모델의 기여가 이 안으로 제한된다.

**② 하역장 판정 프롬프트가 정의와 부정 범주를 준다** (`CONFIRMED`, §2.2 ⑤)
- "물건을 트럭으로/트럭에서 넘겨주는 곳"이라는 정의와 `walkway / safety zone / area of other equipment / unrelated floor`는 "no"라는 지침은 사람의 지식이다.
- 이는 5S식 일반 지식과 비슷하지만, **조건 A(이미지+지시만)에 해당하지 않는다.** 이미 "조건 B와 C 사이"의 지식이 프롬프트에 들어 있다 (§6).
- 이 부정 범주 문구는 개발 이미지에서 v0.1이 매트 밖 바닥을 고른 실패(`ANALYSIS_v0.1_destination_failure.md` §B) 뒤에 만든 v0.2 프롬프트의 일부다 (`INFERRED`: `models.yaml`의 `region_judge` 주석과 `ANALYSIS` 문서 순서).

**③ 객체 어휘와 설명문이 닫혀 있다** (`CONFIRMED`)
- 탐지 대상은 `conveyor`, `truck`, `mobile_robot`, `red_cube` 네 가지다 (`models.yaml` `class_prompts` L34–38). 장면에 무엇이 있는지는 묻지 않는다.
- 설명문은 사람이 쓴다. 컨베이어 설명은 개발 중 Qwen이 컨베이어+로봇팔 박스를 "robot arm"이라고 답하자 넓혔다 (`models.yaml` L51–53 DEV NOTE). "a toy or model truck counts"도 모델이 미니어처를 트럭으로 받아들이도록 하는 문구다.

**④ 결정적 후보를 만든 질의 문구가 정답(Z06)과의 일치로 선택되었다** (`CONFIRMED`) — **이 감사의 가장 중요한 항목**
- `docs/ANALYSIS_v0.1_destination_failure.md` §F: DINO는 `parking space`, `loading dock`, `loading area`, `loading bay` 등을 시도했지만 작업장 전체나 트럭 박스만 반환했다. Qwen grounding은 `"loading area of the truck"`에서 빈 응답, `"parking space of the truck"`에서 `[1398, 72, 1651, 227]`을 반환했다. 문서는 이를 "**Z06 bbox와 IoU 0.76**"이라고 적었다.
- 즉 Z06(사람 초안)과 겹치는 문구를 골라 `planning.yaml`에 넣었다 (`structure_queries`, L30. DEV NOTE: "'loading area of the truck' returned no box; 'parking space of the truck' returned the painted bay").
- 이 선택은 **정답을 본 사람이 문구를 고른 것**이다. "parking space"라는 말에는 "점선 구획이 하역 위치"라는 사람의 지식이 들어 있다.
- 결과: `VERIFICATION_v0.2_2026-11-17.md`가 보고한 "Z06 초안과 IoU 0.722"는 **독립 검증이 아니다.** 문구를 Z06 일치로 골랐기 때문이다. 문서에도 "개발 이미지, 성능 평가 아님"이라고 적혀 있으나, 0.722가 모델의 일반적 능력을 뜻한다고 읽히면 안 된다.
- 모델 입력으로 정답 좌표가 들어간 것은 아니다. 정답이 **설계 선택**에 들어갔다.

**⑤ 후보 5의 "yes"는 정보량이 적다** (`CONFIRMED` 사실, 해석은 `INFERRED`)
- 후보 5는 트럭을 완전히 포함하는 박스다. "이 트럭의 하역장의 일부인가"라는 질문에 트럭이 들어 있는 윤곽이 "yes"로 나오기 쉽다. 응답 이유도 "cargo bed and surrounding floor markings"다.
- 반면 후보 1~4(트럭 밖의 띠)는 모두 "no"였다. 후보 1은 점선 구획 바로 밖의 바닥이다.
- 이 실행에는 **음성 대조(트럭이 없는 곳, 다른 설비 주변, 통로 Z01, 충전 구역 Z05 등의 후보)가 없다.** 그래서 "VLM이 하역장 의미를 구분했다"와 "트럭이 들어 있는 박스에 yes라고 했다"를 구분할 수 없다.

### 3.3 개발 데이터 위생 (`CONFIRMED`)

| 사실 | 근거 |
|---|---|
| 개발 이미지 `assets/reference/warehouse_scene_reference_01.jpg`의 SHA-256 `d1984a19d56443e5…`는 **EXP110 `P06_cheese_2026-11-14-182900.jpg`와 같다** | `sha256sum` 대조. EXP110의 test는 P05–P10 (`PROJECT_STATUS.md` 2026-11-15 갱신) |
| IMAGE_DEMO의 수동 박스(컨베이어, 트럭, 빨간 큐브)의 출처가 "EXP110 annotation P06" | `STN/config/demo/reference_scene_manual_inputs.json` provenance 필드 |
| 지도 다각형은 P01(EXP110 dev)에서 그린 초안이고, `reference_scene_map.reference_P01.jpg`가 그 복사본 | `reference_scene_map.json`, SHA `448b5258…` = P01 |
| v0.1, v0.2의 모든 조정(DINO 임계 0.20→0.10, 설명문, ROI 여백, 구조 질의, 판정 프롬프트)이 이 한 장면에서 이루어졌다 | `ARCHITECTURE.md` §9, `EXPERIMENT_PLAN.md` §1, `planning.yaml` DEV NOTE |

- EXP110 계획서와 이전 결정은 "P05–P10은 이후 튜닝·평가에 재사용하지 않는다"였다. 이 프로젝트의 개발 이미지가 P06이므로 그 결정과 충돌한다. (두 프로젝트의 문서 체계가 달라 의도된 변경인지는 `UNKNOWN`이다.)
- `runs/20261117_222422_e316`(11-17 22:24)은 `~/Pictures/undistorted_26007129092710.png`라는 다른 이미지로 시작했으나 `analyze_scene` 중 사용자 중지로 `CANCELLED`되었다. **이 프로젝트에 실제로 완료된 held-out 이미지 실행은 없다** (`CONFIRMED`).

### 3.4 정답 유출 가능성 종합

| 경로 | 존재 여부 | 비고 |
|---|---|---|
| 모델 입력에 정답 좌표·지도·주행영역·접근성을 포함 | **없음** (`CONFIRMED`) | 프롬프트 해시 대조, 이미지 확인, 코드 인자 확인 |
| 후보 생성에 정답 좌표를 사용 | **없음** (`CONFIRMED`) | 후보는 트럭 박스와 Qwen grounding 박스에서만 나온다 |
| 설계 선택(질의 문구, 프롬프트, 설명문)에 정답과 개발 실패를 반영 | **있음** (`CONFIRMED`) | §3.2 ④. 평가 수치의 독립성을 해친다 |
| 목적지 종류·기준 물체·정의가 사람 입력 | **있음** (`CONFIRMED`) | §3.2 ①②. "자율 추론"의 범위를 좁힌다 |
| 개발 이미지와 평가 이미지의 분리 | **지켜지지 않음** (`CONFIRMED`) | 개발 이미지 = EXP110 P06, 평가용 새 이미지 없음 |

---

## 4. 역할 구분 (A–E)

범주: **A** VLM이 영상·자연어로 추론 / **B** Grounding DINO 등 탐지 모델 / **C** 사람이 사전에 지정 / **D** Python 계산 / **E** 가정값 또는 미검증

### 4.1 사용자가 지정한 8개 항목

| 항목 | A VLM | B 탐지 | C 사람 지정 | D Python | E 가정·미검증 | 코드 위치 |
|---|---|---|---|---|---|---|
| **객체 탐지** | 크롭 yes/no 검증, Qwen grounding fallback (컨베이어·트럭만) | DINO 후보 | 클래스 어휘 4개, 클래스별 프롬프트, 설명문, `verify_top_k`, 임계 0.10 | NMS, 면적 제한, 후보 병합 | **정확도 미측정** (정답 박스 없음). 로봇은 3후보 모두 불채택 | `grounding.py` `detect_reference_objects` L146 |
| **공간 관계** | **없음** | – | 임계값 (`near_px` 150, `contain_ratio` 0.8) | 박스 간격, 포함률 (이미지 픽셀). `produced_by`는 `UNVERIFIED`로 기록만 | 픽셀 거리 ≠ 바닥 거리, 임계값은 가정 | `scene_graph.py` `build_scene_graph` L18 |
| **하역장의 의미적 위치** | 후보 박스(구조 grounding) 생성, 후보별 yes/no | – | 목적지 종류, 기준 물체(트럭), 정의문, 부정 범주, 질의 2개, 띠 깊이 `band_ratio` 0.6 | 기하 띠 4개, 트럭 연관성 필터 (`structure_min_truck_containment` 0.5 등) | **정답 없음**, 질의는 개발 이미지에서 선택 | `region_grounding.py` L37–180 |
| **물체 배치 가능 영역** | **없음** | – | 지도의 장애물·금지구역, 제외 클래스 (truck, conveyor, mobile_robot) | 의미 영역 − 트럭 박스 − 지도 장애물, 대상 크기 ½ 침식 | **배치 표면 미측정**, 지도 `DRAFT_UNCONFIRMED` | `placement.py` L27 |
| **로봇 접근 가능 영역** | **없음** | – | 통행 영역 Z02 (노란 경계 초안), 장애물, 금지구역 | 충돌 없는 셀 ∩ 도달 거리 ∩ 원점 도달 가능 | **로봇 반경 35 px, 도달 거리 180 px는 가정** | `goal_selector.py` L43, `map_builder.py` |
| **실제 그리퍼 작업 가능성** | **없음** | – | – | 목표점과 배치 영역 사이의 **픽셀 거리 ≤ 가정 도달 거리**로 대신함 | **전부 미측정·미구현.** `robot_task_success = NOT_CLAIMABLE` | `orchestrator.py` `_destination_record` L692–760 |
| **이동 목표점** | **없음** | – | 거리 가중치 1.5, 도달 거리 | `select_goal`의 점수 최소 셀 | 파라미터가 가정 | `goal_selector.py` L72–90 |
| **경로 생성** | **없음** | – | 플래너 선택 (`SAFETY_COST_ASTAR`) | A* + 직선화 + 독립 검증 | 반경·여유가 가정, 이미지 좌표 | `planner.py`, `path_validator.py` |

### 4.2 그 밖의 판단

| 항목 | 분류 | 비고 |
|---|---|---|
| 명령 해석 | A (텍스트) + C (화이트리스트) + D (정규화) | VLM은 영상 없이 텍스트만 본다. 지원 범위 밖이면 Python이 거부 |
| 행동 순서 (`actions`) | **C** (고정 템플릿) | `task_parser.py` `ACTION_TEMPLATES` L19–23, `actions_source: "fixed_template (not model output)"` |
| 빨간 큐브 후보 | D (HSV 색·형태) + B (DINO) + A (Qwen 검증) | "색만으로는 채택하지 않는다" (`target_inspection.py`). 큐브 색은 `SUPPORTED_TARGET_COLORS = {"red"}`로 제한 |
| 로봇 시작점 | B+A (로봇 검출 시) / C (수동 `start_pose`) | 이 장면에는 로봇이 없어 수동 (700, 740) |
| 여러 하역 후보 중 최종 선택 | D | 의미 판정이 둘 이상 yes면 **로봇 접근 점수가 최소인 쪽**을 고른다 (`_select_dropoff_pose` L552). 의미가 아니라 로봇 지오메트리가 결정한다. 규칙은 `dropoff_choice`로 기록된다 |
| 장면 이해 | A | **출력이 쓰이지 않는다** (§4.3) |

### 4.3 장면 이해 출력은 어디에도 쓰이지 않는다 (`CONFIRMED`)

- `grep -rn "scene_analysis" src`: 읽는 곳이 없다. `orchestrator.py` L327–332(기록), L807(`result.json`에 포함), `grounding.py`(호출)뿐이다.
- 이 실행의 출력(`analysis/scene_analysis.json`)에는 "loading docks", "loading area near truck", "red cube is not visible in the image" 같은 서술이 있으나 후보 생성·판정·검증 어디에도 들어가지 않는다.
- 연구 목표의 1번(어떤 물체와 설비가 존재하는가), 2번(각 물체와 공간의 기능)은 **이 단계가 맡을 자리인데 실제 의사결정에는 연결되어 있지 않다.** 프롬프트에 목적지 종류와 기준 물체가 이미 들어 있다는 점(§2.2 ②)과 합치면, 이 호출은 연구 주장에 기여하지 않는 기록용이다.

### 4.4 연구 목표 6개 항목의 구현 현황

| # | 연구 목표 | v0.2에서 누가 하는가 | 판정 |
|---|---|---|---|
| 1 | 어떤 물체와 설비가 존재하는가 | 고정된 4개 클래스를 DINO+VLM 검증. 자유 서술(scene_analysis)은 미사용 | **부분** (닫힌 어휘) |
| 2 | 각 물체와 공간의 기능 | 하역장 yes/no만. 다른 공간(통로, 충전, 작업셀)의 기능 추론 없음 | **매우 제한** |
| 3 | 지시된 목적지가 어느 영역인가 | 트럭 앵커 후보 5개에 대한 yes/no | **제한적** (§3.2 ①) |
| 4 | 작업에 필요한 행동 | 고정 템플릿 | **VLM 아님** |
| 5 | 픽업·배치 조건 충족 여부 | 기하 + 가정값. 파지, 표면, 그리퍼 도달 없음 | **미구현** |
| 6 | 접근 위치 | Python | VLM 아님 (의도된 분리) |

---

## 5. VLM이 스스로 판단하는 항목 / 사람·Python이 정하는 항목 (종합)

**VLM이 실제 영상으로 판단하는 것 (`CONFIRMED`)**
1. 후보 크롭이 해당 객체(컨베이어, 트럭, 로봇, 빨간 큐브)인지 yes/no/uncertain
2. "parking space of the truck"에 해당하는 박스 위치 (구조 grounding)
3. 후보 윤곽이 "이 트럭의 하역 영역의 일부인가" yes/no/uncertain
4. 명령문 해석 (텍스트만)

**VLM이 판단하지 않는 것**
- 장면에 어떤 시설이 있는가 (자유 탐색), 시설의 기능, 목적지가 될 수 있는 영역을 스스로 제안하는 것
- 공간 관계, 배치 가능성, 접근 가능성, 파지·그리퍼 도달, 이동 목표, 경로, 행동 순서

**사람이 정한 것**
- 목적지 종류(하역장)와 기준(트럭), 클래스 어휘, 객체 설명문, 하역장 정의와 부정 범주, 구조 질의 문구
- 통행 영역·장애물·금지구역 지도 (초안), 로봇 시작점
- 로봇 반경, 도달 거리, 거리 가중치, 근접 임계값 (전부 이미지 픽셀 가정)

**Python이 계산한 것**
- 후보 띠 생성, 트럭 연관성 필터, 박스 NMS와 병합, 배치 영역, 접근 영역, 목표점, 경로, 경로 검증, 인계 요청

---

## 6. 5S 기반 자율 의미 추론 실험의 적절성

### 6.1 판단

**연구 질문으로는 적절하다. 그러나 현재 v0.2 구조로 세 조건의 차이를 측정하면 거의 아무것도 보이지 않을 것이다** (`INFERRED`). 이유와 조건은 아래와 같다.

### 6.2 조건 설계에 대한 검토

| 조건 | 제안 | 검토 |
|---|---|---|
| A: 이미지 + 지시 | 외부 지식 없음 | **현재 시스템은 이미 A가 아니다.** 하역장 정의와 부정 범주, 트럭 기준이 프롬프트·파서에 들어 있다 (§3.2 ①②). 진짜 A를 만들려면 명령에서 "하역장"이라는 단어도 빼고("화물을 내려놓을 수 있는 곳으로 이동해")를 쓰거나, 최소한 "A의 현재 형태"를 명시해야 한다 |
| B: + 일반 5S 원칙 | 정리·정돈·청소·청결·습관화 | 5S 자체는 하역장 위치를 정하지 않는다. VLM이 5S에서 어떤 추론 규칙을 끌어내는지가 불분명하다. B의 텍스트를 사람이 "5S에서는 구역을 표시선으로 나누고 통로와 작업 구역을 구분한다" 식으로 구체화하면 사실상 C가 된다. **B의 문구는 실험 전에 한 번 확정하고, 이 장면의 색·위치·설비 이름을 포함하지 않아야 한다** |
| C: + 현장별 공간 규칙 | 명시적 규칙 | 규칙의 구체성에 따라 둘로 나눠야 한다. **C1** 일반 유형 규칙("점선 주차 구획은 트럭이 서는 자리이고 그 안팎이 하역 위치다" 등 현장 유형 지식), **C2** 이 장면 좌표·구역 지정. C2는 "정답을 준 조건"이라 **상한/조회 기준선**으로만 쓰고 자율성 주장에 쓰지 않는다. 사용자 원칙(정답 좌표는 평가 외에 주지 않음)과도 맞는다 |

### 6.3 현재 구조로 측정하기 어려운 이유 (`CONFIRMED` 구조 사실 + `INFERRED` 해석)

1. **출력이 예/아니오다.** 조건마다 달라질 수 있는 것은 후보 5개에 대한 yes/no뿐이라 해상도가 낮다. 조건 간 차이가 "하역장을 다르게 추론했다"인지 "문구에 민감했다"인지 가를 수 없다.
2. **후보를 사람이 만든다.** 후보가 트럭 박스 주변과 `parking space` 질의 결과로 고정되어 있어, 조건 B·C가 **후보 자체를 바꾸지 못한다.**
3. **목적지 종류가 파서에 고정되어 있다.** 조건과 상관없이 `loading_area`+`truck`만 지원한다.
4. **판정 프롬프트가 이미 조건 B~C에 해당하는 정보를 가지고 있다** (§6.2 A행).
5. **프롬프트 문구 민감도를 모른다.** v0.1→v0.2에서 같은 모델이 후보 판정 방식만 바꿨을 뿐인데 결과가 달라졌다 (`VERIFICATION_v0.2` §1). 조건 텍스트 하나를 추가한 차이가 의미 차이인지 문구 효과인지 분리하려면 대조가 필요하다.
6. **개발 이미지가 하나이고 정답이 없다.** 조건 간 차이를 일반화할 수 없고 정확도 기준도 없다 (`NO_GT (pending)`).

### 6.4 실험으로 성립하려면 필요한 것 (설계 제안, 구현은 하지 않았다)

- **모델이 후보를 직접 제안**하도록 출력 형식을 확장한다. 예: "장면에서 화물을 내려놓을 수 있는 영역을 박스로 제시하고 각각의 기능 이름과 근거를 적어라". 그러면 조건별로 제안 영역의 위치, 개수, 기능 라벨이 달라질 수 있다.
- **명령의 개념을 단계적으로 줄인다.** 하역장 → "화물을 내려놓을 곳" → "작업을 수행할 곳" 순으로 조건 A를 만든다.
- **음성 대조를 넣는다.** 트럭이 없는 장면, 트럭이 이동·회전한 장면, 점선 구획이 가려진 장면, 통로(Z01)·충전(Z05)·적치 구역(Z03·Z04)을 후보로 준 판정. 같은 프롬프트로 "하역장이 아닌 곳"에 no가 나오는지 본다.
- **영상 없는 대조(텍스트만)**를 둔다. 같은 질문을 영상 없이 던져, 모델이 언어 연관성(트럭 = 하역장)만으로 답하는 비율을 본다.
- **조건 텍스트를 사전 등록하고 해시를 남긴다.** B의 문구는 실험 전에 고정하고, 결과를 본 뒤 고치지 않는다. 프롬프트 문구 변형 2~3개로 민감도도 본다.
- **정답은 평가 전용이다.** 사람이 그린 하역 영역 다각형(`scripts/evaluate_run.py`의 `loading_region_polygon`)을 실행 전에 작성하고, 모델·후보 생성·프롬프트에 넣지 않는다.
- **새로 촬영한 held-out 이미지**로 평가한다 (§3.3).
- 지표 예: 제안 영역과 정답 영역의 IoU, 거짓 양성(정답 밖 영역을 하역장이라고 제안한 비율), 보류 비율(근거 단서를 가렸을 때 "확인 불가"), 조건 간 일치율, 근거 인용의 사실성(실제 장면에 있는 단서를 말했는가).

---

## 7. 작업 가능성 판단의 현재 구현 수준

### 7.1 요구 조건별 현황 (`CONFIRMED`)

| 필요한 확인 | v0.2 구현 | 수준 |
|---|---|---|
| 픽업 위치에 접근 가능한가 | `select_goal(…, "pickup_approach", max_dist_px=140)`: 충돌 없는 셀 중 대상 박스에서 140 px 이내이고 시작점에서 도달 가능한 셀 | 지오메트리 구현. **도달 거리 140 px는 가정** |
| 물체를 파지할 수 있는가 | 없음. 큐브가 검출되고 접근 목표가 있다는 것뿐이다. 크기, 자세, 그리퍼 개구, 장애물 간섭 검사 없음 | **미구현** |
| 하역장에 적절한 배치 위치가 있는가 | 의미 영역 − 트럭 박스 − 지도 장애물, 큐브 크기의 ½로 침식한 면적이 남는가 (4,729 px) | 지오메트리 구현. **배치 표면(높이, 강도, 점유)은 미측정** (`placement.py` L16 `BASIS`) |
| 배치 가능한 위치에 로봇이 접근 가능한가 | `select_goal(…, "dropoff_approach", max_dist_px=180)` | 지오메트리 구현. **180 px는 가정** |
| 그리퍼가 배치 위치까지 도달 가능한가 | 목표점과 배치 영역 사이 픽셀 거리 ≤ 180 px의 **간접 확인**. 그리퍼 방향, 관절·링크 제한, 전면 단일 그리퍼의 방향성은 모델링하지 않았다 | **미구현** (`goal_reach_check`가 "the plan holds only if the real reach covers …"라고 명시) |

### 7.2 이 장면의 결과가 가정에 걸려 있는 정도 (`CONFIRMED`, `VERIFICATION_v0.2` §1–2)

- 선택 목표는 배치 영역까지 177.5 px, 가정 도달 거리 180 px와의 **여유가 2.5 px**다.
- 배치 영역에서 가장 가까운 충돌 없는 로봇 위치까지는 155.2 px다.
- 도달 거리를 160 px 미만으로 두면 `ROBOT_APPROACH_NOT_FOUND`, 160 px 이상이면 성공이다 (접근 셀 수 0→3→128 px).
- 도달 거리를 실측하기 전의 `PLAN_COMPLETE`는 "가정 아래의 이미지 좌표 계획"일 뿐이다. 코드와 문서가 그렇게 말한다.

### 7.3 의미 판단과 물리 검증의 분리 (`CONFIRMED`, 유지할 점)

- 의미 판정(VLM)과 배치·접근·경로(Python)는 별도 단계이고 서로 다른 실패 코드를 가진다: `NO_LOADING_REGION_FOUND`, `SEMANTIC_REGION_UNCERTAIN`, `PLACEMENT_REGION_EMPTY`, `ROBOT_APPROACH_NOT_FOUND`, `ROBOT_APPROACH_UNREACHABLE`, `PATH_NOT_FOUND` 등 (`orchestrator.py` `APPROACH_CODES` L61, `_destination_record`).
- VLM 응답만으로 물리 조건이 성공 처리되는 경로는 없다. `claims.robot_task_success`는 **항상** `NOT_CLAIMABLE`이다 (`orchestrator.py` L759). `send_to_hardware()`는 항상 예외를 낸다 (`robot/handoff.py`).
- 사용자의 요구("검증되지 않은 물리적 조건을 VLM 추정만으로 성공 처리해서는 안 된다")에 **위반 사항은 찾지 못했다.**

### 7.4 개선 여지 (구현하지 않음)

| 항목 | 내용 | 등급 |
|---|---|---|
| 상태 이름 | `overall_status = PLAN_COMPLETE`, `final_stage_status = DESTINATION_PLANNED`는 "작업 가능"으로 읽힐 수 있다. 같은 `result.json`의 `claims`가 이를 막지만, GUI·논문 표에서 `PLAN_COMPLETE`가 단독으로 보이면 오독 위험이 있다 (`INFERRED`) | 표기 |
| 노란 경계 모델 | 지도 `Z02`는 `DRAFT_UNCONFIRMED`이고 "차체는 경계를 넘지 않고 그리퍼만 넘는다"는 물리 제약이 "로봇 중심 셀 ∩ 도달 거리"로만 표현된다. 그리퍼 방향, 전면 한 방향 도달 영역은 없다 | 미구현 |
| 목표 선택 기준 | 경로 비용을 함께 반영해 배치 영역에서 먼 쪽(177.5 px)을 골랐다 (`VERIFICATION_v0.2` §4에도 기록됨). 실측 후 "배치 영역과의 거리 우선"으로 재판단 필요 | 보류 |
| 접근성이 의미 선택을 정하는 경우 | 의미 후보가 여러 개이면 로봇 접근 점수가 최소인 쪽이 `selected_loading_region`이 된다 (`orchestrator.py` L552–556). 의미 판정을 평가할 때는 `semantic_regions_plausible`(전체 yes 목록)을 기준으로 해야 한다 | 평가 시 주의 |

---

## 8. 연구 목표에 맞추기 위해 변경해야 할 부분

**코드를 수정하지 않았다.** 아래는 사용자가 결정할 제안이며 우선순위 순이다. "이 감사에서 확인된 문제"와 "추천 변경"을 연결했다.

| 순위 | 변경 | 근거 | 영향 범위 |
|---|---|---|---|
| 1 | **평가 데이터 분리.** 새로 촬영한 이미지(트럭 위치·방향 변경, 점선 구획 가림, 트럭 없음 등)와 사람 정답(하역 영역 다각형)을 모델 실행 전에 확보한다. 개발 이미지 P06은 개발·진단에만 쓰고 평가에 쓰지 않는다 | §3.3. 현재 모든 수치가 개발 이미지에 맞춘 것 | 데이터, 문서 |
| 2 | **구조 질의 문구의 지위를 바꾼다.** "parking space of the truck"는 사람 지식이 담긴 입력이므로 **조건 C1급의 사전 정보**로 기록하거나, 정답과 독립인 중립 질의 집합을 실험 전에 고정한다. Z06과의 IoU 0.722는 독립 증거로 인용하지 않는다 | §3.2 ④ | `planning.yaml`, 문서 |
| 3 | **VLM이 후보를 제안하는 경로를 추가**한다 (자유 출력: 영역 박스 + 기능 라벨 + 근거). 현재 "후보 5개에 yes/no"에서 "후보를 만들고 판정"으로 | §6.3 1–2 | `region_grounding.py`, 프롬프트 |
| 4 | **장면 이해 출력을 의사결정에 연결하거나 제거 표기한다.** 연결한다면 객체 어휘·시설 기능을 후보 생성의 입력으로 쓰고, 연결하지 않는다면 "기록용"이라고 명시한다 | §4.3 | `orchestrator.py`, 문서 |
| 5 | **조건별 사전 정보를 코드에서 분리해 기록한다.** 각 VLM 호출의 프롬프트에 어떤 정보(하역 정의, 부정 범주, 기준 물체, 조건 텍스트)가 포함되었는지 `prior_knowledge` 필드로 남긴다. 프롬프트 원문 자체도 실행 기록에 저장한다 (지금은 해시만) | §2.1 | `qwen_json`, manifest |
| 6 | **목적지 종류 확장을 별도 단계로** (충전, 적치, 작업셀 등). 현재는 `loading_area`+`truck`만 허용해 STEP 3("작업 맥락 기반 목적지")의 자율성을 평가할 수 없다 | §3.2 ① | `task_parser.py` |
| 7 | **음성 대조와 영상 없는 대조를 평가 체계에 넣는다** | §3.2 ⑤, §6.4 | 평가 스크립트, 데이터 |
| 8 | **물리량 실측** (그리퍼 도달 범위와 방향, 온보드 카메라 시야, 배치 표면, 로봇 footprint). 측정 전에는 모든 접근·경로 결과를 "가정 조건부"로 유지한다 | §7 | `handoff_geometry`, 하드웨어 측정 (사용자) |
| 9 | **오독 방지 표기.** `PLAN_COMPLETE` 옆에 항상 `robot_task_success = NOT_CLAIMABLE`을 같이 표시한다 | §7.4 | 시각화, 문서 |

---

## 9. 변경하지 않고 유지해야 할 부분

| 유지할 것 | 이유 (확인한 근거) |
|---|---|
| **의미 판단 / 배치 / 접근 / 경로 단계 분리와 단계별 실패 코드** | v0.1은 접근성을 의미 판단에 섞어 VLM이 고른 영역이 기록되지 않았다 (`ANALYSIS_v0.1` 구현상 결함). v0.2가 분리해 해결했다 |
| **VLM 입력에 지도·정답·접근성을 넣지 않는 원칙과 그 테스트** | 이번 감사에서 프롬프트 해시 9종과 이미지로 확인되었다. 이 원칙이 이 연구의 핵심 장점이다 |
| `prior_zone_consistency`는 평가용 표시 전용, `validated_loading_region`은 확정 zone과 일치할 때만 | 정답이 모델·계획에 되먹임되지 않는다 (`region_grounding.py` L172–179) |
| 후보를 하나씩 따로 판정하는 방식 | v0.1의 겹친 번호 윤곽이 판단을 흐린 사례가 있다 (`ANALYSIS_v0.1` §B). v0.2에서 오선택(매트 밖 "위쪽")이 사라졌다 |
| 기하 띠 후보를 **기준선으로 유지**하는 `candidate_sources` 스위치 | ablation이 가능하다 (`VERIFICATION_v0.2` §1: 띠만 쓰면 `NO_LOADING_REGION_FOUND`). 이 대조는 §3.2 ⑤의 음성 대조 설계의 출발점이 된다 |
| 원문 응답 보존, JSON 스키마 검증, 검증 실패 시 단계 실패 처리 | 모든 호출에서 확인했다 (`qwen_json` L42–57) |
| `robot_task_success = NOT_CLAIMABLE`, 하드웨어 전송 불가, 인계 상태 머신 | §7.3 |
| `do_sample=false`, 실행별 manifest(코드 해시, 프롬프트 해시, 설정 스냅샷, 파일 해시), replay | 재현성. 3회 반복 결과가 같았다 (`VERIFICATION_v0.2` §1) |
| 독립 경로 검증기 (플래너 격자와 별도로 2 px 샘플) | `path_validator.py`. 물리 검증을 VLM과 분리하는 원칙에 맞다 |
| v0.1 동결 스냅샷 `archive/v0.1_2026-11-15/`, 기존 실행 결과 15건 | 이 감사가 의존하는 비교 기준 |
| 지도 `DRAFT_UNCONFIRMED` 표기와 "경로계획에 쓰는 지도는 사람이 검토해야 한다"는 문구 | 정직한 상태 표시 |

---

## 10. 추천하는 후속 실험 설계

우선순위 순. 새 알고리즘이나 모델을 추가하지 않고 **현재 구조에서 할 수 있는 것**부터 둔다. 코드 변경이 필요한 항목은 §8의 해당 순위를 표시했다. 사용자가 확인한 뒤 결정한다.

### 실험 1. 개발 결과의 독립성 점검 (코드 변경 없음)

- 목적: 현재 v0.2 결과가 새 장면에서도 유지되는지 보는 최소 시험.
- 설정: **프롬프트, 질의, 임계값을 동결**(해시 기록)하고, 새로 촬영한 이미지에서 v0.2를 그대로 실행한다.
  - 장면 변화(이동 물체만): 트럭의 위치와 방향 변경, 트럭 제거, 점선 구획 일부 가림, 컨베이어 쪽 물체 배치 변경.
- 정답: 사람이 **실행 전에** 그린 하역 영역 다각형. `scripts/evaluate_run.py`의 `--gt` 형식을 그대로 쓴다 (`data/reference/`는 아직 없다).
- 지표: 의미 영역 IoU, 단계별 실패 코드 분포, 트럭 없는 장면에서의 `NO_TRUCK` 처리, 반복 일치.
- 해석 규칙(사전 고정): 새 장면에서 구조 질의가 구획을 못 찾으면 "질의가 개발 이미지에 맞춰졌다"는 증거로 기록한다. 결과를 본 뒤 질의를 바꾸지 않는다 (바꾸면 새 버전 + 새 이미지).

### 실험 2. 판정의 변별력 점검 (음성 대조, 소규모 코드 변경 또는 수동 후보)

- 목적: §3.2 ⑤의 "yes가 트럭을 포함해서 나온 것인가"를 가른다.
- 설정: 같은 `region_judge` 프롬프트로 **정답이 아닌 후보**를 판정한다.
  - 보행 통로, 충전 구역, 적치 구역, 컨베이어 작업셀, 트럭이 없는 같은 크기의 바닥, 트럭을 포함하지만 구획 밖으로 넓힌 박스, 트럭 박스 자체.
- 지표: 정답 후보의 yes 비율, 비정답 후보의 yes 비율(거짓 양성), 후보 크기·트럭 포함 여부별 yes 비율.
- 해석: 비정답 후보에서도 트럭 포함 후보가 yes이면 판정은 "트럭이 들어 있다"를 보는 것이다.

### 실험 3. 5S 조건 비교 (A / B / C1, 조건 C2는 상한)

- 구조 변경이 필요하다 (§8 순위 3, 6). 현재 구조에서는 **degenerate한 형태**(후보 5개 yes/no)만 가능하다.
- 입력 조건 (모두 같은 이미지, 같은 모델, greedy):

| 조건 | 입력 | 문구 |
|---|---|---|
| A | 이미지 + 지시 | 사전 정보 없음. 명령도 "하역장"이라는 단어를 빼는 변형을 함께 둔다 |
| B | A + 일반 5S 원칙 | 실험 전에 고정된 일반 문장. 장면의 색, 위치, 설비 이름 없음 |
| C1 | A + 일반 현장 유형 규칙 | 현장 유형 지식(바닥 표시선 구획의 의미 등). 장면 좌표 없음 |
| C2 | A + 이 현장의 명시적 공간 규칙 | **조회 상한.** 정답을 준 조건이라 자율성 주장에 쓰지 않는다 |
| T (대조) | 텍스트만 (영상 없음) + A의 문구 | 언어 연관성만으로 답하는 비율 |

- 출력: 모델이 제안한 영역(박스) + 기능 라벨 + 근거 문장. 후보 생성을 Python이 하지 않는다.
- 지표: 정답 영역 IoU, 거짓 양성(정답 밖 제안), 개수, 단서를 가렸을 때의 보류, 조건 간 일치율, 근거의 사실성(장면에 실제로 있는 단서를 말했는가), 문구 변형 민감도.
- 통계: 장면(독립 단위) 수가 작으므로 기술 통계와 개별 사례를 중심으로 보고한다.

### 실험 4. 물리 조건 실측 (사용자 하드웨어 작업, VLM과 독립)

- 측정: 전면 그리퍼의 도달 범위와 방향, 온보드 카메라 시야, 배치 표면(트럭 구획 바닥 높이·점유), 로봇 footprint, 경계를 넘는 그리퍼의 한계.
- 측정 뒤 `handoff_geometry`를 채우고 같은 장면에서 `goal_selection.*.max_dist_px`를 실측으로 바꿔 접근·경로 결과를 다시 계산한다. 실측값을 쓴 실행은 **새 버전**으로 기록한다 (`EXPERIMENT_PLAN.md` §5).
- 이 실험이 끝나기 전에는 접근·경로 결과를 "가정 조건부"로 보고한다.

### 이번 단계에서 하지 않을 것 (사용자 지시 반영)

- 웨이포인트 이동, 새 경로계획 알고리즘, 모델 변경·설치, 환경 재설치, GUI 재작성.
- "옆", "근처" 등 자연어 표현 다양성 실험.

---

## 11. 한계와 확인하지 못한 것

| 항목 | 내용 |
|---|---|
| 모델 직접 재실행 | 하지 않았다. 프롬프트는 설정에서 **재구성**하고 저장된 해시와 대조했다. 이는 이 실행(`20261117_111627_d4cf`)에 대한 확인이며, 같은 코드의 다른 실행에 대해서는 일관성 때문에 같다고 추정할 뿐이다 (`INFERRED`) |
| 코드 해시 | 실행 시점 해시(`254cfb31…`)와 현재 소스를 직접 재계산해 대조하지 않았다. mtime으로만 확인했다 |
| 기본 system 프롬프트 | 코드는 만들지 않는다. 템플릿이 기본 system을 주입하는지는 모델 입력 토큰을 직접 열어 보지 않았다 (`INFERRED`: 템플릿 파일에 기본 system 문구 없음) |
| 이미지 확인 범위 | `region_judge_cand1.png`, `cand5.png`만 직접 열어 확인했다. 나머지 후보와 `verify_*.png`는 열지 않았다. 코드 `render_judge_input`과 `make_verify_crop`이 같은 방식으로 그린다는 점으로 추정한다 |
| GUI 코드 | `gui/main_window.py`(877줄)는 정답 입력과 표시 부분(grep)과 테스트만 확인했고 전체를 읽지는 않았다 |
| v0.1 동결 스냅샷 | `archive/v0.1_2026-11-15/`의 코드와 프롬프트는 열지 않았다. v0.1 실행은 `ANALYSIS_v0.1_destination_failure.md`와 `VERIFICATION_2026-11-15.md` 기록에 의존했다 |
| 의도 | EXP110 P06이 개발 이미지로 쓰인 것이 의도된 결정인지, 단순한 파일 선택인지는 알 수 없다 (`UNKNOWN`) |
| 정확도 | 하역장 정답(사람 다각형)이 없어서 어떤 정확도 수치도 새로 계산하지 않았다 |

---

## 부록 A. 확인한 주요 코드·설정·문서

**코드** (`STN/src/semantic_task_nav/`)
- `agent/orchestrator.py` — `PLANS`, `_parse_task`, `_analyze_scene`, `_detect_objects`, `_infer_destination`, `_assess_placement`, `_select_dropoff_pose`, `_destination_record`, `task_summary`
- `agent/task_parser.py` — `REGION_SYNONYMS`, `REFERENCE_SYNONYMS`, `build_task`, `ACTION_TEMPLATES`, `rule_based_parse` (IMAGE_DEMO 전용)
- `semantics/region_grounding.py` — `generate_candidates`, `structure_candidates`, `render_judge_input`, `judge_candidates`, `ground_loading_region`, `manual_loading_region`
- `semantics/placement.py` — `placement_region`
- `navigation/goal_selector.py` — `select_goal`, `reach_basis`
- `navigation/map_builder.py` — `NavMap`, `check_alignment`, `dynamic_obstacles_from_detections`
- `perception/grounding.py` — `qwen_json`, `analyze_scene`, `verify_box`, `qwen_ground_boxes`, `detect_reference_objects`
- `perception/model_worker.py` — `qwen_generate`, `dino_detect`
- `perception/model_client.py`, `perception/target_inspection.py`, `perception/scene_graph.py`

**설정** (`STN/config/`): `models.yaml`, `planning.yaml`, `app.yaml`, `camera.yaml`, `maps/reference_scene_map.json`, `demo/reference_scene_manual_inputs.json`

**테스트**: `tests/test_destination_chain.py` (`test_one_judgement_per_candidate_and_no_map_in_model_input`, `test_structure_candidates_are_checked_against_the_truck`), `tests/test_gui.py` (`test_human_reference_regions_are_display_only`). 테스트를 실행하지는 않았다.

**문서**: `docs/ARCHITECTURE.md`, `docs/ANALYSIS_v0.1_destination_failure.md`, `docs/VERIFICATION_v0.2_2026-11-17.md`, `docs/EXPERIMENT_PLAN.md`, `scripts/evaluate_run.py` (docstring)

## 부록 B. 분석에 사용한 실제 실행 기록

| 실행 | 모드 | 상태 | 용도 |
|---|---|---|---|
| `runs/20261117_111627_d4cf` | LOCAL_ONLY (실제 Qwen+DINO) | `PLAN_COMPLETE` | **주 분석 대상.** 모델 호출 36건, 프롬프트 해시 대조, 후보 판정 5건, 결과 체인 |
| `runs/20261117_222422_e316` | LOCAL_ONLY | `CANCELLED` | 새 이미지(`~/Pictures/undistorted_26007129092710.png`)로 시작했으나 `analyze_scene` 중 사용자 중지. 호출 2건 |
| `runs/20261115_225944_49f5` (v0.1), `runs/20261117_111211_1210` (v0.2 기하 띠만), `…_111047_aa91`, `…_111321_c086` | LOCAL_ONLY | 문서의 비교표로만 확인 | 직접 열어 재대조하지는 않았다 |
| `runs/_worker_logs/` | – | 열지 않음 | – |

## 부록 C. 이 감사가 새로 만든 파일

- `STN/docs/research_audits/` (폴더, 신규)
- `STN/docs/research_audits/2026-11-18_vlm_autonomy_audit.md` (이 파일)
- 임시 분석 스크립트는 파일로 남기지 않았다 (명령행에서 `python3 -I`로 읽기 전용 실행).
