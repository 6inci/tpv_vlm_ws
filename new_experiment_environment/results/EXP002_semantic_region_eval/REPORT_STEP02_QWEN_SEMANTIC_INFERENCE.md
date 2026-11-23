# EXP002 Step 02 — Qwen3-VL Semantic Region Direct Inference

## 1. Step Result
`PARTIAL` — actual Qwen3-VL inference produced 5 structured semantic-region predictions; 0 include valid normalized polygons.

## 2. Scope
Generated blind semantic-region prediction only. No reference comparison, accuracy/IoU, shortcut judgment, traversability/map regeneration, metric conversion, ROS2/Nav2, or hardware execution occurred.

## 3. Input / Source of Truth
Authoritative input was the original image only. Step 01 reference artifacts were hashed for regression protection but never passed to the processor, prompt, or model.

## 4. Existing Contract Audit
Reused the local Qwen3-VL checkpoint, `qwen_vl` environment, `load_local_qwen`, `prepare_qwen_inputs`, 0..1000 full-image coordinates, and deterministic pixel conversion convention. A step-local wrapper was executed in memory; no persistent framework or script was added.

## 5. Exact Input Image
`/home/binci/vlm_ws/new_experiment_environment/approximate_environment.png` — 1448×1086px, SHA-256 `02a4e79589db3c15529b372768ef349b4ef2e4ebe2efb2454f605be43eb95a95`; existence, decode, dimensions, and expected hash PASS.

## 6. Model / Runtime
- Model: `Qwen/Qwen3-VL-4B-Instruct`
- Cache: `/home/binci/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct` (local files only)
- Environment: `/home/binci/miniconda3/envs/qwen_vl`
- Python 3.10.20, Torch 2.13.0+cu130, Transformers 5.16.1, OpenCV 5.0.0
- Device `cpu`, dtype `torch.bfloat16`, deterministic generation (`do_sample=False`)
- Model load: 0.474s

## 7. Exact Inference Prompt
Selected attempt 1 used exactly:

```text
Analyze only the attached image as an indoor industrial or logistics work environment. Identify functionally meaningful spatial regions directly from visual evidence.

Do not decide a region from one object alone. Consider combinations of visible objects, their spatial relationships, surrounding usable space, and functional context. Do not force regions when evidence is weak. Choose semantic labels yourself; no label list or target region count is provided.

Return JSON only, without markdown, using this schema:
{"scene_summary":"concise description","regions":[{"region_id":"short unique id","semantic_label":"model-chosen functional label","supporting_objects":["visible object descriptions"],"supporting_relations":["concise visible spatial relations"],"reasoning":"20 words or fewer","confidence":0.0,"polygon_normalized":[[x,y],[x,y],[x,y]]}]}

Coordinates are integers normalized from 0 to 1000 over the original full image. Use a polygon only when you can estimate a meaningful boundary; otherwise set polygon_normalized to null. Keep all fields concise and include only well-supported regions. Do not provide metric coordinates, paths, obstacle maps, or hidden objects.
```

The prompt contains no Step 01 labels, polygons, supporting evidence, region count, overlay, or reference content. Leakage check: PASS.

## 8. Inference Attempts
- Attempt 1: max_new_tokens=1600, generated=574, finish=eos_token, JSON=PASS, seconds=199.80, error=NONE

Selected attempt 1 because it was the first response with a complete JSON object and all required fields passing validation. No later attempt was run.

## 9. Selected Model Output
Raw semantic fields come only from attempt 1. Scene summary: Indoor logistics simulation with robotic systems, storage, and transport equipment

## 10. Raw Output Artifact
`model_output/semantic_region_raw.txt` preserves the first response unchanged except one terminal newline. Source type: `MODEL_OUTPUT`; non-empty: PASS.

## 11. Postprocessing
Source type: `POSTPROCESSED`. Only JSON extraction, parsing, schema checks, numeric/range checks, and deterministic normalized-to-pixel conversion were applied. No label, region, object evidence, relation, reasoning, confidence, or polygon was added or corrected.

## 12. Semantic Region Prediction Summary
- `robotic_workstation` — Robotics operation zone (0.95); polygon=NOT_AVAILABLE; objects=3.
- `storage_area` — Inventory storage zone (0.92); polygon=NOT_AVAILABLE; objects=3.
- `maintenance_station` — Tool and equipment maintenance area (0.85); polygon=NOT_AVAILABLE; objects=4.
- `transport_and_loading_zone` — Vehicle and pallet loading area (0.88); polygon=NOT_AVAILABLE; objects=4.
- `mobile_robot_zone` — Mobile robot operation area (0.87); polygon=NOT_AVAILABLE; objects=3.

## 13. Validation / Gates
Pre-inference image/model/environment gates PASS. Post-inference raw non-empty, selected attempt, complete JSON, unique IDs, finite confidence, coordinate bounds, provenance separation, and raw/structured separation PASS.

## 14. Representative Result
`robotic_workstation` / `Robotics operation zone` is supported by model-stated objects and relations in the raw response; no reference-derived evidence was appended.

## 15. Tests
Actual model execution evidence PASS; blind-prompt forbidden-term scan PASS; raw preservation PASS; JSON/schema validation PASS; semantic traceability policy PASS; coordinate validation PASS; visualization decode NOT_APPLICABLE.

## 16. Regression
Protected EXP001 files, Step 01 reference candidate, and Step 01 report have identical pre/post SHA-256 mappings: PASS (22 files).

## 17. Files Created
- `model_output/semantic_region_raw.txt`: unmodified MODEL_OUTPUT for attempt 1.
- `model_output/semantic_region_prediction.json`: validated POSTPROCESSED prediction and runtime/attempt evidence.
- `REPORT_STEP02_QWEN_SEMANTIC_INFERENCE.md`: scope, prompt, attempts, validation, regression, and handoff report.

## 18. Files Modified
NONE. Existing Step 01 and EXP001 artifacts were not modified.

## 19. Confirmed Facts
Qwen3-VL executed on `cpu` against exactly one image; generated 574 tokens in 199.80s; selected raw output parsed and validated.

## 20. Assumptions
Model confidence and semantic reasoning are treated as uncertain MODEL_OUTPUT. Image polygons are qualitative full-image coordinates, not physical boundaries.

## 21. Unknowns
Prediction correctness relative to the protected candidate, region precision/recall, IoU, shortcut behavior, and user agreement remain UNKNOWN because comparison is out of scope.

## 22. Limitations
Single oblique image, CPU execution, model-generated confidence, no metric calibration, and no user-validated ground truth. Polygon availability/quality does not imply navigation suitability.

## 23. System Changes
NONE. No download, installation, environment rebuild, CUDA/driver, calibration, ROS2, service, or shell change occurred.

## 24. Final Status
`PARTIAL` — prediction is structurally usable for the next comparison step; this status does not claim semantic accuracy.

## 25. Recommended Next Step
Qwen Semantic Region Prediction → Reference Candidate Comparison / Evaluation. This next step was not started.
