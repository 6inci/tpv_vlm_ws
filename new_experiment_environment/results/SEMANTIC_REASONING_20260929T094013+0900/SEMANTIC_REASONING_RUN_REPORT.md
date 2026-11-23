# Semantic Reasoning Run Report

Overall status: PARTIAL

## 1. Research question

Can a VLM use objects, spatial relations, and functional context from a third-person workspace image to ground a natural-language instruction to a semantic region and a traversable target, while abstaining when evidence is insufficient?

## 2. Input image

`/home/binci/vlm_ws/new_experiment_environment/warehouse_robotics_1920x1080.png` (1920×1080, SHA-256 `5e1cdda9fec9a3840396dd60cb0743dd2d519fd0884587c1c0156d89a565e331`)

## 3. Models and environments

- Qwen semantic grounding: `Qwen/Qwen3-VL-4B-Instruct` in unchanged `qwen_vl`; actual image-conditioned inference.
- Grounding DINO diagnostic: `IDEA-Research/grounding-dino-tiny` in unchanged `qwen_vl`; CUDA.
- YOLO-World diagnostic: `yolov8s-world.pt` in unchanged `yolo_world`; CUDA.
- Existing DINO/SAM2 and YOLO/SAM2 masks are referenced read-only; segmentation was not redundantly rerun.

## 4. Test cases

15 tasks: TYPE_A 2, TYPE_B 5, TYPE_C 4, TYPE_D 2, TYPE_E 2. Object bbox, semantic region, obstacle/no-go, and robot goal are kept separate.

## 5. Actual inference

Qwen completed the 15-task STRUCTURED_CONTEXT inference. Source-specific Qwen generation was attempted with 15-task, 8-task, and 4-task scopes but returned no complete first response and is BLOCKED after three bounded attempts. Three controlled vocabulary trials completed independently in both detectors.

## 6. Detector Diagnostic

- manipulator: VOCABULARY_DEPENDENT
- conveyor: AMBIGUOUS
- mobile_robot: VOCABULARY_DEPENDENT
- pallet: VOCABULARY_DEPENDENT

The q_030 location was previously localized as `cart` by YOLO at IoU 0.9311, and q_029 as `truck` at IoU 0.9231. These are WRONG_CLASS_LOCALIZATION / CLASSIFICATION_FAILURE cases. Conveyor comparison now uses the same three vocabulary groups in both detectors; because no reviewed conveyor bbox exists, behavior is reported without accuracy or superiority claims.

Work Area detector evidence is stored in `06_error_analysis/error_analysis.json`; the requested before/after Qwen semantic comparison is BLOCKED because source-specific generation did not complete.

## 7. Structured-context and end-to-end results

- structured_context: {'ANSWERED': 13, 'ABSTAIN': 2} / regions {'Loading Area': 7, 'Material Area': 3, 'Storage Area': 2, 'Work Station': 1}
- qwen: {'FAILED': 15} / regions {}
- grounding_dino: {'FAILED': 15} / regions {}
- yolo_world: {'FAILED': 15} / regions {}

STRUCTURED_CONTEXT is model-derived and human-reviewable, not an oracle. End-to-end paths restrict supporting object identities to each detector source.

## 8. Semantic-region predictions and target grounding

TYPE_C predictions are in `04_semantic_regions/predictions.json`. Pixel targets and deterministic synthetic metric projections are in `05_target_grounding/target_predictions.json`. Outside-workspace projections are flagged and never clamped.

## 9. Detector source effects

Cross-source semantic agreement is BLOCKED because the source-specific Qwen paths did not complete. Detector evidence differences are still recorded on this image without accuracy or superiority claims.

## 10. Failure taxonomy

Confirmed detector errors use DETECTION_FAILURE or CLASSIFICATION_FAILURE. RELATION_FAILURE, SEMANTIC_INFERENCE_FAILURE, TARGET_GENERATION_FAILURE, and AMBIGUITY_HANDLING_FAILURE require human review and remain BLOCKED_BY_GROUND_TRUTH.

## 11. Ground truth

Human-reviewed semantic ground truth is unavailable. The Qwen reference boxes and semantic regions are model outputs, not ground truth.

## 12. Claims currently supported

- The 15-task structured-context path and both detectors' three vocabulary trials execute and leave raw/structured artifacts; source-specific Qwen paths are explicitly BLOCKED.
- High-IoU wrong-class localization is separable from failure to localize.
- Pixel targets can be projected through the retained synthetic-only homography.
- Source-dependent semantic outputs and abstention behavior can be counted without calling them correct.

## 13. Claims not supported

No semantic accuracy, relation accuracy, region IoU, instruction success rate, goal-in-region rate, abstention precision/recall, detector superiority, or real-camera metric accuracy is claimed.

## 14. Human validation

Complete `07_human_validation/HUMAN_SEMANTIC_VALIDATION.md` or the JSON template using only the listed review labels. Conveyor location and every predicted region polygon/target require review.

## 15. Real-camera repetition

Capture an undistorted fixed-camera frame, measure floor correspondences with holdouts, annotate objects/relations/regions/masks/valid targets, then rerun with frozen prompts and thresholds before any dry-run or robot motion.

## 16. Next priorities

1. Human-review all 15 tasks and the conveyor candidate detections.
2. Recompute blocked semantic and abstention metrics from reviewed annotations.
3. Capture Scenes B–E to test position, color, cue-removal, and ambiguity shortcuts.
4. Repeat homography and target validation with measured real-camera geometry.
