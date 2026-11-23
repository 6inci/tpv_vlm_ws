# Project Status

Updated: 2026-09-29 (Asia/Seoul)  
Scope: synthetic/offline pipeline plus the current semantic-reasoning and detector-diagnostic run. Existing runs remain read-only.

## Current conclusion

The synthetic/offline pipeline is **PARTIAL**. Actual Qwen3-VL, Grounding DINO, YOLO-World, and SAM2 inference artifacts exist, and a synthetic homography can produce a metric-style occupancy/traversability map. This does not establish detector accuracy, real-camera metric accuracy, or robot-safe navigation.

The current semantic result and latest detector diagnostic are in [`SEMANTIC_REASONING_20260929T094013+0900`](results/SEMANTIC_REASONING_20260929T094013+0900/SEMANTIC_REASONING_RUN_REPORT.md). The detector/SAM2 baseline remains [`YOLO_WORLD_INTEGRATION_20260929T012518+0900`](results/YOLO_WORLD_INTEGRATION_20260929T012518+0900/RUN_REPORT.md), with Qwen/Grounding DINO/SAM2 and geometry provenance from [`METRIC_VLM_PIPELINE_20260929T005848+0900`](results/METRIC_VLM_PIPELINE_20260929T005848+0900/RUN_REPORT.md).

## Pipeline status

| Component | Status | Confirmed result | Current limitation | Next work |
|---|---|---|---|---|
| Qwen3-VL → semantic understanding | PARTIAL | Actual 15-task structured-context inference produced 13 ANSWERED and 2 ABSTAIN results with complete JSON; raw response is retained. | Context is model-derived, not an oracle. Source-specific Qwen generation did not return a complete response after three bounded scopes and is BLOCKED; no human semantic ground truth exists. | Human-review all tasks, then retry source-specific comparison with a shorter frozen per-task protocol on a real-camera frame. |
| Grounding DINO → text-conditioned detection | PASS | Baseline CUDA inference produced 58 detections; three matched diagnostic vocabularies now include conveyor/manipulator synonyms. | Diagnostic detections are behavior observations, not accuracy. Compound/token-fragment labels and unreviewed conveyor location remain ambiguous. | Review conveyor/manipulator candidate boxes and evaluate frozen prompts against validated boxes. |
| YOLO-World → open-vocabulary detection comparison | PARTIAL | Baseline produced 39 detections; controlled probing recovered an `industrial robot arm` candidate and four `belt conveyor` candidates only in the extended-synonym trial. | q_030→cart and q_029→truck remain high-IoU classification failures. The new arm/conveyor candidates are vocabulary-dependent and not ground-truth validated. | Human-review the candidate locations, then freeze the smallest successful vocabulary and threshold. |
| SAM2 → segmentation | PASS | Grounding DINO→SAM2 produced 39 masks; YOLO-World→SAM2 produced 38 masks; 26 cross-detector mask comparisons were generated. | PASS means both prompt paths executed. There is no mask ground truth, so segmentation quality is unmeasured. | Annotate representative masks and report IoU/boundary metrics for both prompt paths. |
| Homography → pixel-to-metric conversion | PARTIAL | A synthetic 12-point homography (8 fit, 4 holdout) achieved 0.0913 m holdout RMSE and 0.114344 m maximum holdout error. | Correspondences and scale are synthetic/manual; this is not real-camera calibration accuracy. | Capture measured floor correspondences in the fixed camera, keep fit and holdout points separate, and recompute error. |
| Metric map → occupancy/traversability representation | PARTIAL | Layered metric-style maps and scene graphs were generated; YOLO box and SAM2 mask occupancy paths both project through the synthetic homography. | Some projected representatives fall outside the nominal workspace; footprint/inflation and collision safety are not validated. | Rebuild from real-camera geometry, validate boundaries/no-go regions, then test robot footprint and inflation offline. |
| MCP → experimental read-only integration prototype | PARTIAL | A local read-only prototype, schema, self-check, and non-empty demo response exist. | No external MCP client, conformance, authentication, or deployment validation was performed. | Connect one read-only client to frozen artifacts and verify schema/error behavior without enabling writes. |

## What the synthetic stage establishes

- CONFIRMED: all four model families were actually executed in the retained valid runs; raw/model outputs and runtime evidence exist.
- CONFIRMED: Qwen, Grounding DINO, and YOLO-World outputs were compared without calling the result accuracy.
- CONFIRMED: structured-context semantic grounding ran over 15 instructions and produced explicit ANSWERED/ABSTAIN outputs; source-specific Qwen comparison remains BLOCKED.
- CONFIRMED: both Grounding DINO bbox→SAM2 and YOLO-World bbox→SAM2 paths produced masks.
- CONFIRMED: synthetic pixel→metric projection, occupancy/traversability map generation, and a read-only MCP prototype execute end to end.
- INFERRED: the architecture is feasible for offline integration.
- UNKNOWN: semantic/detection/segmentation accuracy, real-world metric accuracy, and navigation safety.

## What must be repeated with the real camera

1. Capture a timestamped, undistorted 1920×1080 frame from the fixed camera and record intrinsics, distortion state, pose, and hardware conditions.
2. Measure floor correspondences in world units; fit homography on one set and reserve independent holdout points for error reporting.
3. Re-run Qwen3-VL, Grounding DINO, YOLO-World, and both SAM2 prompt paths with frozen configs and vocabularies.
4. Create human-validated labels, boxes, masks, relations, semantic regions, and traversability references; then compute actual metrics.
5. Validate workspace boundaries, no-go regions, robot footprint, and inflation through offline → visualization → dry-run. Robot motion remains out of scope until those gates pass.

Run-by-run provenance and classifications are in [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md). Non-destructive cleanup candidates are listed in [`CLEANUP_CANDIDATES.md`](CLEANUP_CANDIDATES.md).

## Update 2026-11-15 — real camera, EXP110 word grounding (appended; content above unchanged)

- Work folder: `my_own_ws/EXP110_word_grounding` (dev P01–P04, test P05–P10; Cheese JPEG, raw distorted 1920×1080 pixels).
- **Experiment status: `PASS` (G1 = executed as pre-registered).** Settings were frozen with hashes (`FREEZE_2026-11-15.sha256`). Each test condition was run once. All numbers were recomputed from raw outputs, and rescoring gave identical results.
- **G1 PASS does not mean the method is proven better.** The hypothesis (C6-dino > C1, English, strict query success) is **not established**:
  - Test results: 0.272 vs 0.208, +6.5 pp.
  - Query-cluster bootstrap 95% CI [−0.3, +14.0] pp. This is the pre-registered basis.
  - McNemar p = 0.0026 is reference only, because pairs are not independent: 39–43 of 49 queries have the same outcome on all 6 images.
- **Task objects are unresolved.** Blocks and small pallets score 0 under every condition (C1, C2, C6-dino).
- **Main bottleneck is candidate generation.** In C6-dino, 314 of 744 answer boxes were never among the detector candidates.
- **Scope: same-scene pilot.** Test images share all fixed objects with dev, so these results do not show generalization.
- Report: `my_own_ws/EXP110_word_grounding/results/run03/analysis_test_20261115T121130/TEST_REPORT.md`
- **Next**
  - EXP111: small-object detection, starting with a failure-cause analysis on dev.
  - Then E3a (VLM spatial-relation target selection), per `REAL_ENV_EXPERIMENT_PLAN_v3.md` (v2 kept unchanged).
  - The final system structure is not decided.
  - P05–P10 are not reused to tune new settings. Later experiments need newly captured evaluation images.
