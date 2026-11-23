# 저장 이미지 기반 VLM-픽셀-월드 좌표 실험 복원 기록

## 목적과 범위

이 문서는 실제 저장 이미지에서 VLM 객체 이해, bounding box, 대표 pixel,
이미지 좌표계 보정, ground/world coordinate, 결과 artifact 및 시각화까지가
구축·검증된 시점을 복원한다. 기존 STEP 번호는 출처를 찾기 위한 참고 표기이며,
아래 Experiment 번호는 artifact 시간순으로 새로 부여했다.

포함 범위는 정적 `photo_*.png`와 저장된 calibration/geometry 자산이다. `/dev/video*`
연속 처리, live scene monitor/tracking, dynamic obstacle, occupancy/path planning,
persistence/replanning, ROS/Nav2 및 실제 로봇 실행은 포함하지 않는다.

### 자료 판정 규칙

판정은 다음 우선순위를 적용했다: (1) 실행 artifact·raw output·JSON·로그,
(2) 현재 코드와 설정, (3) 당시 Markdown, (4) 사용자 기록, (5) 이후 회고,
(6) 실행 확인 없는 제안. 따라서 코드만 존재하는 경로는 실험 성공으로 기록하지
않았다.

상태 표기는 다음처럼 엄격히 사용한다.

| 표기 | 의미 |
| --- | --- |
| 실행됨 | 실제 runtime 결과가 artifact에 남아 있다. |
| 결과 검증됨 | 실행 결과에 명시적 PASS 또는 수치 재현 근거가 있다. |
| 부분 결과 | 실행됐지만 정해진 gate 또는 전체 목표를 통과하지 못했다. |
| 구현·실행 확인 없음 | 현재 코드가 있으나 이 문서 범위에서 실행 artifact를 확인하지 못했다. |
| 제안만 됨 | 실행 또는 구현 근거를 확인하지 못했다. |
| 폐기됨 | 후속 artifact에서 rejection 또는 NOT_RUN으로 명시된 결과다. |

## 작성 전 chronology 검증

| 순서 | 실험/변경 | 확인 근거 | 확실성 | 다음 단계가 필요해진 이유 |
| --- | --- | --- | --- | --- |
| 1 | Legacy undistortion + Homography 재현 | `geometry_regression_summary.json` (2026-09-17 16:41 KST) | 높은 확률 | 결정론적 pixel-to-world 재현 후 VLM bbox를 연결할 수 있게 됨 |
| 2 | 단일 객체 `red cup` VLM grounding | `qwen_grounding_summary.json`의 `created_at=16:53:17` | 확정 | 하나의 객체를 넘어 scene-level semantic 결과의 안정성을 확인할 필요 |
| 3 | 다중 객체 scene grounding | `scene_repeatability_summary.json`의 `created_at=17:08:33` | 확정 | broad scene label과 reference label의 차이를 별도로 분석할 필요 |
| 4 | semantic reference 비교·label specificity 점검 | `semantic_reference_comparison.json`의 artifact 생성시각 17:23 KST | 높은 확률 | broad scene에서 alias/미검출이 있어 task-conditioned 질의가 필요 |
| 5 | task-conditioned target grounding | `target_grounding_summary.json`의 `created_at=17:37:58` | 확정 | 저장 이미지 전체가 legacy geometry와 호환되는지 검사할 필요 |
| 6 | stored-image geometry compatibility audit | `geometry_compatibility_audit.json` 생성시각 2026-09-18 15:46 KST | 높은 확률 | 기존 geometry를 적용 가능한 사진과 허용 불가 사진을 분리할 필요 |
| 7 | offline candidate → semantic/metric batch 시도 | `semantic_metric_grounding_summary.json` 생성시각 18:17 KST 및 raw/frame artifacts | 높은 확률 | strict parser/association이 metric 출력을 막은 원인을 representative 조건으로 재검증할 필요 |
| 8 | 기존 raw output의 candidate-conditioning audit | `qwen_output_audit.json` 생성시각 22:39 KST | 높은 확률 | 새 inference 없이 실패 원인을 보존·분류할 필요 |
| 9 | Method C full-image overlay 재검증 | `step22c_summary.json` 생성시각 23:00 KST | 높은 확률 | NO_VISIBLE 결과 후 secondary context crop 조건을 검증할 필요 |
| 10 | Method D full-image + secondary crop 재검증 | `step22d_summary.json` 생성시각 2026-09-19 12:20 KST | 높은 확률 | associated full-object bbox를 실제 metric point까지 연결 |

파일시각만 있는 항목은 보존·복사 과정에서 변경될 가능성을 완전히 배제할 수 없어
`높은 확률`로 표기했다. 다른 항목은 JSON 내부의 runtime `created_at`으로 상대 순서를
확정했다. 위 순서와 충돌하는 artifact는 확인되지 않았다.

## Experiment 01 — Legacy geometry의 결정론적 재현

**참고 STEP:** STEP 04.  
**상태:** 실행됨, 결과 검증됨.

`results/geometry_legacy_regression/geometry_regression_summary.json`은 12개 기준점을
검사하여 `pass: true`를 기록한다. stored undistortion 재현 오차와 world 재현 오차는
모두 max/mean `0.0`이고, legacy fitting error는 max `0.008796 m`, RMSE `0.005736 m`다.
이는 저장된 기하 자산을 동일하게 재현했다는 검증이지 독립적인 절대 정확도 검증은
아니다.

현재 `src/vlm_reposition/geometry/legacy.py`는 입력 이미지를 정확히 legacy size
`2016×1133`으로 검증하고, raw pixel → stored undistortion → stored Homography를
`LEGACY_WORKSPACE_COORDINATE` (meter)로 변환한다. calibration 명목 크기 `1920×1080`과
legacy image 크기가 다르므로 camera matrix는 legacy image 크기로 scale되며, 그 뒤에는
저장된 `new_camera_matrix`를 사용한다.

**보존 artifact:** `results/geometry_legacy_regression/geometry_regression_summary.json`,
`/home/binci/camera_calibration_1920x1080/calibration_result.json`,
`/home/binci/qwen_vision_test/homography_12pt_undistorted.json`.

#### 당시 판단과 이후 수정

- 당시 가정: 저장된 calibration과 Homography를 같은 image domain에서 재현하면 VLM의
  bbox 대표점을 deterministic하게 metric coordinate로 바꿀 수 있다.
- 근거: 12개 기준점의 stored pipeline 재현 결과가 0 차이였다.
- 이후 한계: 이 결과만으로 새로운 camera pose의 물리 정확도는 증명되지 않는다.
- 수정: 후속 stored-image audit에서 reference와 development-compatible candidate를
  구분했고, metric provenance를 development inference로 제한했다.

## Experiment 02 — 단일 객체 semantic-to-metric grounding

**참고 STEP:** STEP 05.  
**상태:** 실행됨, 결과 검증됨 (`not_final_paper_accuracy: true`).

`photo_04.png`의 질의 `red cup`에 대해 Qwen3-VL-4B-Instruct를 3회 실행했다.
normalized bbox는 세 run에서 동일했고, bbox `[447, 527, 483, 595]`의 bottom-center는
raw `(937.44, 674.135)` px, undistorted `(902.754, 744.593)` px가 되었다. 저장된
geometry 결과는 `(0.267293, 0.657320) m`이며 run 간 peak-to-peak은 bbox, pixel,
world 모두 0이다.

기존 rounded legacy world 값 `(0.265, 0.662) m`과의 차이 norm은 `0.005211 m`였다.
이는 stored bbox integration의 결과이고, 논문용 정확도 주장은 아니다.

**보존 artifact:** `results/qwen_grounding/qwen_grounding_summary.json` 및 raw output/
visualization이 있는 `results/qwen_grounding/`.

## Experiment 03 — 다중 객체 scene grounding과 repeatability

**참고 STEP:** STEP 06.  
**상태:** 실행됨, 부분 결과 (`PARTIAL`).

같은 `photo_04.png`에 대해 scene-level Qwen inference를 3회 실행했다. 각 run은
11개 entity를 반환했고 matching된 11개 common entity의 bbox IoU와 world coordinate
변동은 기록상 동일했다. 그러나 workspace bound 내 geometry는 각 run에서 3개뿐이고
8개는 `OUTSIDE_LEGACY_WORKSPACE`였다. 따라서 repeatability artifact는 있으나 모든
scene entity가 navigation-usable하다는 결과는 아니다.

**보존 artifact:** `results/multi_entity_grounding/scene_repeatability_summary.json`,
`results/multi_entity_grounding/`.

## Experiment 04 — label specificity와 partial reference 비교

**참고 STEP:** STEP 07.  
**상태:** 실행됨, 부분 결과.

3회 비교에서 `red cup`은 위치 IoU `0.9444`로 대응했지만 fresh label이 `red mug`여서
`LABEL_MISMATCH`였고, `white bag`은 `NO_BBOX_MATCH`였다. `gold trophy-like object`는
`trophy` alias와 IoU `0.9153`으로 PASS했다. 이 artifact는 partial semantic reference와
IoU association을 사용한 비교이지, 전체 class accuracy 측정이 아니다.

#### 당시 판단과 이후 수정

- 당시 가정: broad scene 질의로도 reference 대상의 semantic label을 충분히 특정할 수
  있다.
- 문제를 드러낸 결과: cup/mug label 차이와 white bag bbox 미매칭.
- 수정: 다음 실험에서 target을 명시하는 task-conditioned prompt를 사용했다.

**보존 artifact:** `results/semantic_refinement/semantic_reference_comparison.json`.

## Experiment 05 — task-conditioned target grounding

**참고 STEP:** STEP 08.  
**상태:** 실행됨, 결과 검증됨 (`PASS`, `not_final_paper_accuracy: true`).

`red cup`, `white bag`, `gold trophy-like object` 각각을 3회 target-conditioned로
질의했다. 모든 대상이 3/3 `TARGET_FOUND`였고 run 내 bbox/world 변동은 0으로 기록됐다.
reference bbox IoU는 각각 `0.9452`, `0.9341`, `0.9234`다. 다만 white bag의 metric
position은 매 run `OUTSIDE`였으므로, 검출 성공과 active workspace 적합성은 별개다.

**보존 artifact:** `results/task_conditioned_grounding/target_grounding_summary.json`.

## Experiment 06 — 저장 이미지 및 legacy geometry 호환성 audit

**참고 STEP:** STEP 21.  
**상태:** 실행됨, 결과 검증됨 (`OFFLINE_REPLAY_READY`).

`photo_04`를 legacy geometry reference로 고정하고, ORB/KNN/RANSAC image-to-image
diagnostic으로 저장 이미지의 compatibility를 감사했다. `photo_04`는
`REFERENCE_VERIFIED`; `photo_01`, `02`, `03`, `05`, `06`, `07`, `08`은
`DEVELOPMENT_COMPATIBLE_CANDIDATE`다. `qwen_frame.jpg`는 resolution mismatch로
metric transform `NOT_ALLOWED`다.

이 audit의 image registration transform은 **metric Homography가 아니다**. 같은 해상도도
compatibility의 충분조건이 아니며, candidate 사진의 metric 값에는
"image-registration-based development inference only" provenance가 붙는다.

**보존 artifact:** `results/offline_image_replay/geometry_compatibility_audit.json`,
`results/offline_image_replay/image_inventory.json`.

## Experiment 07 — offline change candidate의 strict semantic/metric 연결 시도

**참고 STEP:** STEP 22 초기 실행.  
**상태:** 실행됨, 부분 결과; metric point 생성은 폐기됨.

이 artifact는 5개 triggered frame, 58개 triggered candidate를 열거하지만 실제 Qwen
inference는 7회만 실행했다. 결과는 `VISIBLE_ENTITY_FOUND: 1`, `NO_VISIBLE_ENTITY: 46`,
parser failure 2, `ASSOCIATED: 0`, `METRIC_GROUNDING_AVAILABLE: 0`이다. 즉 58개 전체에
대해 Qwen 실행이 완료됐다는 기록은 없으며, 성공 metric point도 없다.

strict parser 및 candidate↔entity association gate가 유지되어 untrusted model output을
metric conversion으로 넘기지 않았다. `photo_02` ID/bbox mismatch, `photo_03`의
`NO_VISIBLE_ENTITY`, `photo_06` wrong coordinate domain rejection, `photo_07`의
`NO_VISIBLE_ENTITY` 관련 raw/frame/failure artifact는 보존되어 있다.

**보존 artifact:** `results/offline_semantic_metric_grounding/semantic_metric_grounding_summary.json`,
`frame_results/`, `qwen_raw_outputs/`, `metric_observations/`; 추가 `photo_06` retry는
`retry_v2_photo_06/semantic_metric_grounding_summary.json` (`parser_failures: 1`)이다.

#### 당시 판단과 이후 수정

- 당시 가정: candidate region만으로 full visible-object bbox와 association을 얻을 수
  있다.
- 문제를 드러낸 결과: strict parsing/coordinate-domain/association rejection으로
  `METRIC_GROUNDING_AVAILABLE: 0`이었다.
- 수정: 기존 결과는 변형하지 않고, representative candidate 하나에 대해 candidate를
  명시한 full-image conditioning을 분리 검증했다.

## Experiment 08 — 기존 raw output의 candidate-conditioning audit

**참고 STEP:** STEP 22B.  
**상태:** 실행됨, 결과 검증됨; 새 Qwen inference는 수행하지 않음.

`qwen_output_audit.json`은 기존 raw output 3개를 read-only로 점검했다. photo06 track011은
candidate bbox가 image boundary 밖이라는 설명과 함께 `NO_VISIBLE_ENTITY`였고,
photo08 track039 및 photo07 track036도 `NO_VISIBLE_ENTITY`였다. 이 단계는 실패 원인을
분류한 audit이며 VLM 재실행 또는 metric conversion이 아니다.

**보존 artifact:** `results/candidate_conditioning_audit/qwen_output_audit.json`.

## Experiment 09 — Method C: full-image overlay conditioning 재검증

**참고 STEP:** STEP 22C.  
**상태:** 실행됨, 부분 결과.

`image_014_photo_08 / track_039`에 대해 original source를 수정하지 않고 Qwen을 정확히
1회 실행했다. 결과는 `NO_VISIBLE_ENTITY`, association `NOT_EVALUATED`, metric `NOT_RUN`이다.
보호 대상 artifact 435개는 전후 동일 digest로 보존되었다. 이 실패는 final metric
성공으로 해석하지 않았고 fallback candidate도 실행하지 않았다.

**보존 artifact:** `results/candidate_conditioned_vlm_revalidation/step22c_summary.json`,
`qwen_raw_output.txt`.

## Experiment 10 — Method D: full image + secondary context crop의 대표 end-to-end 연결

**참고 STEP:** STEP 22D.  
**상태:** 실행됨, 결과 검증됨 (`PASS`); 개발용 metric inference.

대표 후보는 `image_014_photo_08`, `track_039`, candidate bbox
`[1017, 545, 1375, 684]` px다. metric coordinate domain은 수정·resize·crop·warp·undistort를
하지 않은 원본 full scene `photo_08.png` (`2016×1133`)으로 유지했다. secondary crop은
semantic context에만 쓰였고 metric source는 아니다.

Qwen 1회 실행은 `tennis racket`을 `VISIBLE_ENTITY_FOUND`로 반환했다. full-image
normalized bbox `[504, 481, 682, 604]`는 `IMAGE_1` reference로 validation되었고,
raw full-object bbox `[1016.064, 544.973, 1374.912, 684.332]` px다. candidate와 entity의
IoU `0.994579`, centroid/center gates 통과로 `ASSOCIATED`가 됐다. 이는 **same visible
entity association** 근거이며 identity proof는 아니다.

full-object bbox의 bottom-center `(1195.488, 684.332)` px를 existing `LegacyGeometry`에
넣어 stored undistorted point `(1098.033, 752.113)` px와 world coordinate
`(0.764672, 0.713486) m`를 생성했다. frame은 `LEGACY_WORKSPACE_COORDINATE`, workspace
status는 `INSIDE`다. raw Qwen output, parsed result, association evidence, geometry
provenance, metric result가 모두 저장되어 있다.

**보존 artifact 및 시각화:**

- `results/candidate_conditioned_vlm_revalidation_method_d/qwen_raw_output.txt`
- `results/candidate_conditioned_vlm_revalidation_method_d/parsed_result.json`
- `results/candidate_conditioned_vlm_revalidation_method_d/association_result.json`
- `results/candidate_conditioned_vlm_revalidation_method_d/metric_result.json`
- `results/candidate_conditioned_vlm_revalidation_method_d/bbox_reference_validation.json`
- `results/candidate_conditioned_vlm_revalidation_method_d/step22d_summary.json`
- `results/candidate_conditioned_vlm_revalidation_method_d/`의 original/full-scene/crop
  provenance 및 overlay visualization artifacts

#### 당시 판단과 이후 수정

- 당시 가정: original full image에 candidate를 명시하면 visible entity의 full bbox와
  candidate association을 얻을 수 있다.
- Method C 결과: overlay만으로는 `NO_VISIBLE_ENTITY`여서 metric 단계가 열리지 않았다.
- 수정: full scene을 metric reference로 보존하고 secondary context crop을 semantic
  보조 정보로만 추가했다.
- 결과: full-image bbox가 strict validation과 association gate를 통과했고, 하나의
  저장 이미지 대표 후보가 VLM → bbox → bottom-center → stored geometry → world XY로
  끝까지 연결됐다.

## 복원된 최종 정적 파이프라인

```text
저장 원본 image (2016×1133, full-scene domain)
  → Qwen visible-object semantic grounding
  → validated full-object bbox (normalized 0–1000, IMAGE_1)
  → raw bbox bottom-center pixel
  → stored undistortion
  → legacy Homography
  → LEGACY_WORKSPACE_COORDINATE world XY (m)
  → raw/parsed/association/metric/visualization artifact 저장
```

이 경로에서 VLM은 semantic label과 bbox만 제공한다. pixel-to-world 계산은 VLM이 하지
않고 existing deterministic geometry가 수행한다. Method D의 crop은 metric coordinate
domain이나 bbox remapping에 사용되지 않았다.

## 이번 문서에서 확정하지 않는 사항

| 항목 | 복원 판정 |
| --- | --- |
| 새 camera pose에서의 absolute metric accuracy | 구현·실행 확인 없음; current artifact는 이를 주장하지 않음 |
| legacy Homography의 논문용 최종 물리 검증 | 제안/후속 필요 상태; `not_final_paper_accuracy` 및 provenance가 명시됨 |
| physical obstacle footprint | 폐기/미수행 범위 (`NOT_AVAILABLE`) |
| planner injection, A*, replanning | 폐기/미수행 범위 (`NOT_RUN`) |
| live camera, tracking, dynamic obstacle, ROS/Nav2, robot execution | 이번 문서 범위 밖; 성공으로 기록하지 않음 |

## Live Camera 단계로 넘어가기 위한 준비 상태

후속 camera readiness 자료는 별도로 존재하지만, 실제 capture 또는 temporal evidence는
이 문서의 결론에 사용하지 않았다. 현재 정적 pipeline의 전제는 정확한 `2016×1133`
image domain과 stored legacy geometry이며, 새 live source에 이를 적용하려면 먼저
camera pose/geometry compatibility를 별도 실험으로 검증해야 한다.
