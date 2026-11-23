# Cleanup Candidates

Updated: 2026-09-29 (Asia/Seoul)  
This is a review list only. **Nothing was deleted, moved, merged, or overwritten.** A future cleanup requires explicit approval and should preserve a manifest or archive first.

## OBSOLETE_PLACEHOLDER

These three artifacts came from the embedded YOLO stage of the metric baseline before YOLO-World was actually available. They are not inference evidence and are replaced by the separate current YOLO integration run.

| Artifact | Evidence | Current handling |
|---|---|---|
| [`07_yolo_world/detections.json`](results/METRIC_VLM_PIPELINE_20260929T005848+0900/07_yolo_world/detections.json) | Status is BLOCKED and detection list is empty; no YOLO model was executed. | Retain in place, exclude from all current counts/comparisons. |
| [`07_yolo_world/overlay.png`](results/METRIC_VLM_PIPELINE_20260929T005848+0900/07_yolo_world/overlay.png) | Visualization contains only the BLOCKED diagnostic, not detections. | Retain in place, never present as model output. |
| [`07_yolo_world/RESULT.md`](results/METRIC_VLM_PIPELINE_20260929T005848+0900/07_yolo_world/RESULT.md) | Placeholder/blocker description only. | Retain in place as historical context. |

## Superseded drafts/intermediates

These may be candidates for a later archive after provenance review; none should be removed now.

| Artifact | Classification | Replacement |
|---|---|---|
| [`detection_comparison_attempt01.json`](results/AUTO_VLM_PIPELINE_20260928T225300+0900/05_semantic_analysis/detection_comparison_attempt01.json) | FAILED_DIAGNOSTIC | Corrected [`detection_comparison.json`](results/AUTO_VLM_PIPELINE_20260928T225300+0900/05_semantic_analysis/detection_comparison.json) |
| [`demo_response_attempt01.json`](results/METRIC_VLM_PIPELINE_20260929T005848+0900/09_mcp/demo/demo_response_attempt01.json) | FAILED_DIAGNOSTIC | Non-empty [`demo_response.json`](results/METRIC_VLM_PIPELINE_20260929T005848+0900/09_mcp/demo/demo_response.json) |
| [`metric_scene_graph_attempt01.json`](results/METRIC_VLM_PIPELINE_20260929T005848+0900/10_integration/metric_scene_graph_attempt01.json) | FAILED_DIAGNOSTIC | Final [`metric_scene_graph.json`](results/METRIC_VLM_PIPELINE_20260929T005848+0900/10_integration/metric_scene_graph.json) |
| [`RUN_REPORT_attempt01.md`](results/METRIC_VLM_PIPELINE_20260929T005848+0900/RUN_REPORT_attempt01.md) | SUPERSEDED | Final [`RUN_REPORT.md`](results/METRIC_VLM_PIPELINE_20260929T005848+0900/RUN_REPORT.md) |
| [`RUN_REPORT_attempt01.md`](results/YOLO_WORLD_INTEGRATION_20260929T012518+0900/RUN_REPORT_attempt01.md) | SUPERSEDED | Final [`RUN_REPORT.md`](results/YOLO_WORLD_INTEGRATION_20260929T012518+0900/RUN_REPORT.md) |
| [`RUN_REPORT_attempt02.md`](results/YOLO_WORLD_INTEGRATION_20260929T012518+0900/RUN_REPORT_attempt02.md) | SUPERSEDED | Final [`RUN_REPORT.md`](results/YOLO_WORLD_INTEGRATION_20260929T012518+0900/RUN_REPORT.md) |
| [`ARTIFACT_MANIFEST_attempt01.json`](results/YOLO_WORLD_INTEGRATION_20260929T012518+0900/ARTIFACT_MANIFEST_attempt01.json) | SUPERSEDED | Final [`ARTIFACT_MANIFEST.json`](results/YOLO_WORLD_INTEGRATION_20260929T012518+0900/ARTIFACT_MANIFEST.json) |

## Retain even though diagnostic

[`objects_attempt01.json`](results/METRIC_VLM_PIPELINE_20260929T005848+0900/01_qwen/objects_attempt01.json) records a real Qwen execution whose response was truncated. It is a **FAILED_DIAGNOSTIC**, not an obsolete placeholder, and should remain with the raw response/runtime provenance.

## Explicitly protected

Actual inference outputs, raw model responses, config snapshots, environment/runtime records, reference/validation material, final run reports, reproducibility scripts, and model-related provenance are not cleanup candidates.
