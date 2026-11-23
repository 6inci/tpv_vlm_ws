# EXP002 Step 01 — Reference Annotation Candidate

## 1. Step Result
`PARTIAL` — the candidate is structurally valid and usable for the next inference comparison step, but region boundaries remain preliminary and are not user-validated.

## 2. Scope
Built one semantic-region evaluation reference candidate for the protected EXP001 input. No Qwen inference, prediction evaluation, traversability/map regeneration, metric conversion, ROS2, Nav2, or hardware execution was performed.

## 3. Input / Source of Truth
Primary baseline: `/home/binci/vlm_ws/new_experiment_environment/results/EXP001_run01`. The baseline config, input analysis, runtime, report, object, relation, semantic-region, provenance, and manifest artifacts were audited before annotation.

## 4. Existing Contract Audit
- Object schema: `id`, `label`, `category`, normalized `bbox_2d`, derived `bbox_px`, confidence, provenance.
- Relation schema: `subject_id`, predicate, `object_id`, confidence; minimally extended with relation `id` so regions can reference evidence.
- Region schema: `id`, `label`, normalized polygon, derived pixel polygon, confidence, evidence/reasoning; evidence fields were specialized as `supporting_objects` and `supporting_relations`.
- Coordinates: original full image normalized to 0..1000; derived pixels use the existing deterministic conversion helpers.
- Project status vocabulary is PASS/FAIL-family, while this step explicitly defines PASS/PARTIAL/FAIL; the step rubric's `PARTIAL` is recorded separately as `step_result`.
- Provenance: `REFERENCE_GT_CANDIDATE`; creation method is separately identified as agent/manual review.

## 5. Exact Input Image Used
`/home/binci/vlm_ws/new_experiment_environment/approximate_environment.png` — 1448×1086px, SHA-256 `02a4e79589db3c15529b372768ef349b4ef2e4ebe2efb2454f605be43eb95a95`. Config, runtime, input analysis, and report all resolve to this same path.

## 6. Implementation
Reviewed the 16 EXP001 model detections rather than accepting them automatically, refined two labels, added the visible safety fence, conveyor platform, and workbench chair, added evidence relations, then created five multi-object semantic regions. Existing coordinate conversion helpers were reused. No persistent helper/script was added.

## 7. Reference Annotation Summary
- Objects: 19 candidates (16 reviewed EXP001 bboxes + 3 manually added visible objects).
- Relations: 13 image-verifiable evidence relations.
- Semantic regions: 5 candidates.
- Authoritative artifact: `/home/binci/vlm_ws/new_experiment_environment/results/EXP002_semantic_region_eval/reference/reference_annotation_candidate.json`.
- Annotation status: `REFERENCE_GT_CANDIDATE`; never user-validated Ground Truth.

## 8. Semantic Region Evidence Summary
- `region_storage`: two adjacent loaded racks plus access space.
- `region_work`: tool workbench, chair, and their spatial association.
- `region_loading`: truck, boxed pallet, three cones, adjacency, and handling space.
- `region_robot_work`: multiple manipulators, conveyor, safety fence, and grouping/adjacency evidence.
- `region_material`: loaded cart and adjacent container/drum pallets.
- `Drop Zone` remains `UNKNOWN`; one pallet alone was rejected as a semantic shortcut.

## 9. Validation / Gates
Input gate PASS. JSON parse, required fields, unique IDs, object/relation references, region evidence references, coordinate bounds, finite confidence, non-zero polygon area, allowed provenance, visualization decode/dimensions, and multi-object region evidence all PASS.

## 10. Representative Result
`region_loading` is supported jointly by truck `obj_003`, pallet `obj_016`, cones `obj_005..007`, relations `rel_004..007`, and nearby handling space. It is not inferred from the truck alone.

## 11. Tests
- JSON parse validation: PASS.
- Relation reference integrity: PASS.
- Semantic object/relation reference integrity: PASS.
- Polygon 0..1000 coordinate and area validation: PASS.
- Candidate-not-user-validated-GT validation: PASS.
- Source and overlay image decode/dimension validation: PASS.

## 12. Regression
EXP001 protected file count before/after: 20/20. Full SHA-256 mapping equality: PASS. Deleted files: 0. Modified baseline files: 0.

## 13. Files Created
- `reference/reference_annotation_candidate.json`: authoritative structured reference candidate used by a future evaluator.
- `visualizations/reference_annotation_overlay.png`: human-review overlay; not an authoritative data source.
- `REPORT_STEP01_REFERENCE_ANNOTATION.md`: step audit, validation, limitations, and handoff record.

## 14. Files Modified
NONE. EXP001 and existing source code were not modified.

## 15. Confirmed Facts
The source file exists and decodes at 1448×1086; its SHA-256 matches EXP001 runtime/input analysis. All candidate IDs/references and image-coordinate bounds validate. EXP001 remained byte-identical.

## 16. Assumptions
Functional semantic labels and approximate boundaries are based on visible object combinations, spatial relationships, usable surrounding space, and scene context. Bboxes inherited from EXP001 were visually reviewed but are not pixel-accurate segmentation.

## 17. Unknowns
Exact functional boundaries, user/domain-expert agreement, occluded object extents, and whether the lower pallet represents a Drop Zone remain unknown.

## 18. Limitations
This is `REFERENCE_GT_CANDIDATE`, not Ground Truth. It is image-coordinate, non-metric, single-view, approximate, and unsuitable as a physical obstacle map or navigation input. No inter-annotator agreement or segmentation metric exists yet.

## 19. System Changes
NONE. No packages, model files, calibration, ROS2, CUDA, drivers, services, or shell configuration were changed.

## 20. Final Status
`PARTIAL` — all structural and regression gates pass; annotation uncertainty is explicit and the candidate is usable for the next step.

## 21. Recommended Next Step
Reference Annotation Candidate → Qwen3-VL Semantic Region Direct Inference. This next step was not started.
