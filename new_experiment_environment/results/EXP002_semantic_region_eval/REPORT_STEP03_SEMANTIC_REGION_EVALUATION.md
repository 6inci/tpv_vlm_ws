# EXP002 Step 03 — Reference Candidate vs Qwen Semantic Prediction Evaluation

## 1. Step Result
`PASS` — all five reference and five predicted regions were evaluated exactly once with one-to-one preliminary semantic correspondence. PASS means this evaluation step completed, not that model performance is validated.

## 2. Scope
Compared existing Step 01 `REFERENCE_GT_CANDIDATE` regions with existing Step 02 blind Qwen predictions. Qwen was not rerun; no artifact, polygon, map, calibration, ROS2/Nav2, or hardware state was changed.

## 3. Input / Source of Truth
Authoritative structured inputs: `/home/binci/vlm_ws/new_experiment_environment/results/EXP002_semantic_region_eval/reference/reference_annotation_candidate.json` and `/home/binci/vlm_ws/new_experiment_environment/results/EXP002_semantic_region_eval/model_output/semantic_region_prediction.json`. Raw model evidence: `/home/binci/vlm_ws/new_experiment_environment/results/EXP002_semantic_region_eval/model_output/semantic_region_raw.txt`. Reports were audit-only and prediction was not reconstructed from prose.

## 4. Existing Contract Audit
Reference uses `id`/`label`, object/relation IDs, confidence, and normalized polygons. Prediction uses `region_id`/`semantic_label`, textual evidence, confidence, and null polygons. No reusable semantic correspondence evaluator existed, so a step-local deterministic coverage validator plus explicit manual judgments was used; no persistent framework was added.

## 5. Reference Summary
Five preliminary regions: Storage Area, Work Station, Loading Area, No-Go Zone, and Material Area. Provenance is `REFERENCE_GT_CANDIDATE`, not user-validated Ground Truth.

## 6. Prediction Summary
Five blind MODEL_OUTPUT concepts: Robotics operation, inventory storage, maintenance, vehicle/pallet loading, and mobile-robot operation. All five prediction polygons are `null`.

## 7. Matching Method
One-to-one assignment used semantic function, supporting objects, supporting relations, functional context, and reasoning. Label string equality and polygon geometry were not used. Judgments are `MANUAL_OR_AGENT_INFERENCE`.

## 8. Region Correspondence Results
Two `MATCH` and three `PARTIAL_MATCH`; no `NO_MATCH` or `UNKNOWN`. Every reference and prediction ID appears exactly once.

## 9. Evidence Consistency Results
- Storage: racks/shelves, boxes, and bins agree substantially; relation types differ but are compatible.
- Work: workbench, tools/laptop, chair, and co-location agree; maintenance is a narrower role interpretation.
- Loading: truck/pallet/loading evidence agrees, but cones are omitted and handcart/barrel evidence broadens scope.
- Robot work: arms, conveyor, fence, and spatial grouping substantially agree; operation versus restriction is a framing difference.
- Material: pallets/barrels overlap, but cart evidence is missing and the prediction centers mobile-robot operation.

## 10. Missed Reference Regions
NONE under the one-to-one candidate-based rubric. Three references are only partial matches; this is not a claim of full detection.

## 11. Extra Prediction Regions
NONE as unmatched predictions. `mobile_robot_zone` is retained as a partial material-area correspondence with role reinterpretation, not declared a false positive. `transport_and_loading_zone` shows merged-region behavior.

## 12. Preliminary Shortcut Diagnostic
Four predictions: `LOW_SHORTCUT_CONCERN` because their raw evidence uses multiple objects, a relation, and functional context. `mobile_robot_zone`: `POSSIBLE_SHORTCUT` because its label/reasoning is anchored on one mobile robot while the remaining evidence is generic. These are single-scene diagnostics, not final shortcut conclusions.

## 13. Localization Evaluation
`polygon_iou`, region overlap, centroid error, boundary precision, and spatial coverage accuracy are all `NOT_AVAILABLE`. Reason: `MODEL_OUTPUT provided no region polygons`. No polygon or localization metric was fabricated.

## 14. Preliminary Descriptive Metrics
- reference_region_count: 5
- prediction_region_count: 5
- match_count: 2
- partial_match_count: 3
- no_match_count: 0
- unknown_count: 0
- unmatched_reference_count: 0
- unmatched_prediction_count: 0

Scope: `PRELIMINARY_REFERENCE_CANDIDATE_BASED`. Candidate agreement ratio was not computed because weighting partial matches would be arbitrary in one scene.

## 15. Validation / Gates
Reference/prediction JSON parse, ID existence, exact coverage, one-to-one assignment, duplicate full-match detection, provenance, zero-polygon localization guard, output parse, and protected-artifact hashes all PASS.

## 16. Representative Comparison Table
| Reference Region | Prediction Region | Match Status | Evidence Agreement | Notes |
|---|---|---|---|---|
| Storage Area (`region_storage`) | Inventory storage zone (`storage_area`) | MATCH | objects=SUBSTANTIAL, relations=PARTIAL | Both describe organized inventory storage centered on loaded shelving. |
| Work Station (`region_work`) | Tool and equipment maintenance area (`maintenance_station`) | PARTIAL_MATCH | objects=SUBSTANTIAL, relations=SUBSTANTIAL | The same workbench/chair area is identified, but the prediction specializes the broader Work Station as maintenance/configuration. |
| Loading Area (`region_loading`) | Vehicle and pallet loading area (`transport_and_loading_zone`) | PARTIAL_MATCH | objects=PARTIAL, relations=PARTIAL | Truck/pallet loading function agrees, but the prediction merges handcart/barrel material evidence and omits cone-based bay evidence. |
| No-Go Zone (`region_robot_work`) | Robotics operation zone (`robotic_workstation`) | MATCH | objects=SUBSTANTIAL, relations=SUBSTANTIAL | Both describe the manipulator/conveyor/fence cluster as a functional robot operation area. |
| Material Area (`region_material`) | Mobile robot operation area (`mobile_robot_zone`) | PARTIAL_MATCH | objects=PARTIAL, relations=PARTIAL | Pallet/drum transport context overlaps, but the prediction centers the mobile robot rather than material staging and omits the cart. |

## 17. Tests
- Reference/prediction coverage: PASS (5/5 each, exactly once).
- One-to-one and duplicate MATCH detection: PASS.
- Polygon count 0 ⇒ every localization metric `NOT_AVAILABLE`: PASS.
- Candidate-not-validated-performance terminology: PASS.
- Source traceability and protected-artifact hash comparison: PASS.

## 18. Regression
EXP001 plus all pre-existing EXP002 Step 01/02 files remained byte-identical before/after: PASS (26 protected files).

## 19. Files Created
- `evaluation/semantic_region_comparison.json`: authoritative structured preliminary comparison, evidence diagnostics, counts, localization status, and provenance.
- `REPORT_STEP03_SEMANTIC_REGION_EVALUATION.md`: human-readable method, table, gates, limitations, and next-step decision.

CSV was not created because it would duplicate the complete JSON without adding research value.

## 20. Files Modified
NONE.

## 21. Confirmed Facts
Both inputs contain five regions; prediction contains zero polygons. The evaluation accounts for every ID once and records two matches plus three partial matches. Source hashes are embedded in the JSON.

## 22. Assumptions
Manual correspondence assumes synonymous functional labels can match when evidence agrees. `mobile_robot_zone` is paired partially with Material Area to preserve one-to-one accounting, while its role reinterpretation remains explicit.

## 23. Unknowns
User/domain agreement, actual spatial overlap, whether maintenance is the intended work-station role, and whether mobile-robot operation deserves a separate reference region remain unknown.

## 24. Limitations
Reference is a candidate, not validated Ground Truth; prediction has no geometry; judgments are manual; only one scene is evaluated. Counts are diagnostic and must not be interpreted as final accuracy, precision, recall, or F1.

## 25. System Changes
NONE.

## 26. Final Status
`PASS` — evaluation mechanics, coverage, provenance, and regression gates pass. This status does not assert that Qwen predictions are correct.

## 27. Recommended Next Step
Priority **A: Reference candidate user/domain validation**. Three partial correspondences depend on debatable functional scope, so reference semantics should be validated before quantitative localization work. After that, priority B would be polygon/localization-targeted Qwen inference. Neither step was started.
