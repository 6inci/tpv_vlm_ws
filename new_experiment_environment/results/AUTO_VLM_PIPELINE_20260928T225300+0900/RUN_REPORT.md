# RUN REPORT

STEP 0 Environment Inventory: PASS

STEP 1 Object Detection: PARTIAL

STEP 2 Semantic Regions: PARTIAL

STEP 3 Traversability: PARTIAL

STEP 4 Grounding DINO: PASS

STEP 5 Semantic Analysis: PARTIAL

STEP 6 Image-plane 2D Map: PASS

STEP 6 Metric 2D Map: BLOCKED

STEP 7 End-to-End Pipeline: PASS

## Run identity

- Experiment/run: `AUTO_VLM_PIPELINE` / `AUTO_VLM_PIPELINE_20260928T225300+0900`
- Timestamp: `2026-09-28T22:53:00.615443+09:00`
- Git commit: `UNKNOWN (workspace is not a Git worktree)`
- Input: `/home/binci/vlm_ws/new_experiment_environment/approximate_environment.png`
- Input SHA-256: `02a4e79589db3c15529b372768ef349b4ef2e4ebe2efb2454f605be43eb95a95`
- Model evidence: Qwen/Qwen3-VL-4B-Instruct completed EXP001/EXP002; Grounding DINO `IDEA-Research/grounding-dino-tiny`
- Environment: `/home/binci/miniconda3/envs/qwen_vl/bin/python`, Python `3.10.20`, GPU `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- Command: `scripts/run_pipeline.py`
- Config snapshot: `config_snapshot.json`

## Execution and findings

- STEP 1 reused the completed Qwen pass instead of rerunning it. Sixteen complete object records were recovered previously; the preserved raw response was truncated, so status remains PARTIAL.
- STEP 2 reused both blind/open semantic inference and target-taxonomy region evidence. Open output had no polygons; target polygons remain approximate.
- STEP 3 replaced the old semantic-zone-as-obstacle behavior. It computed `floor - physical occupancy - explicit no-go - 8 px margin`; semantic labels alone do not restrict traversal.
- STEP 4 attempts: `[{"attempt": 1, "hypothesis": "local cache", "status": "BLOCKED", "error": "OSError(\"We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.\\nCheck your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.\")"}, {"attempt": 2, "hypothesis": "download with installed Transformers; no package change", "status": "PASS"}, {"attempt": 3, "hypothesis": "inference on cuda", "status": "PASS", "detections": 63}]`
- STEP 5 makes no accuracy claim. Comparison attempt 1 admitted one unreasonable IoU 0.003 pair; preserved attempt 2 adds a compatible-label IoU ≥ 0.10 guard and yields 14 matches (minimum retained IoU 0.298).
- STEP 6 generated an image-plane map. Metric conversion is BLOCKED because the exact image/camera pose has no validated pixel↔world evidence; unrelated homographies were not used.
- Pixel measurements were computed from masks, not manually invented: `{"workspace_floor_px": 1195846, "physical_occupancy_px": 473809, "inflated_occupancy_px": 550310, "explicit_no_go_px": 183223, "traversable_candidate_px": 605722, "unknown_px": 376682}`
- No package/system/CUDA/ROS2 change and no robot motion occurred.

## Failed or limited items

- Qwen evidence is a prior completed run with truncated primary JSON and approximate region polygons.
- Grounding DINO failure details, if any, are preserved in `04_grounding_dino/detections.json` and `RESULT.md`; work continued with Qwen evidence.
- Traversability is bbox/threshold based, perspective-distorted, and not safe for planning.
- Metric map is BLOCKED pending current-scene calibration and floor correspondences.

## Artifacts

- `00_inventory/ENVIRONMENT_INVENTORY.md`
- `00_inventory/inventory.json`
- `01_object_detection/PROMPT.txt`
- `01_object_detection/RESULT.md`
- `01_object_detection/object_detection_overlay.png`
- `01_object_detection/objects.json`
- `02_semantic_regions/MODEL_RAW.txt`
- `02_semantic_regions/PROMPT.txt`
- `02_semantic_regions/RESULT.md`
- `02_semantic_regions/open_semantic_analysis.json`
- `02_semantic_regions/semantic_regions.json`
- `02_semantic_regions/semantic_regions_overlay.png`
- `03_traversability/RESULT.md`
- `03_traversability/no_go_mask.png`
- `03_traversability/physical_occupancy_mask.png`
- `03_traversability/traversability.json`
- `03_traversability/traversability_mask.png`
- `03_traversability/traversability_overlay.png`
- `04_grounding_dino/PROMPT.txt`
- `04_grounding_dino/RESULT.md`
- `04_grounding_dino/detections.json`
- `04_grounding_dino/grounding_dino_overlay.png`
- `05_semantic_analysis/RESULT.md`
- `05_semantic_analysis/comparison_attempts.json`
- `05_semantic_analysis/detection_comparison.json`
- `05_semantic_analysis/detection_comparison_attempt01.json`
- `05_semantic_analysis/semantic_analysis_overlay.png`
- `05_semantic_analysis/semantic_scene_graph.json`
- `05_semantic_analysis/spatial_relations.json`
- `06_2d_map/RESULT.md`
- `06_2d_map/occupancy_image_plane.png`
- `06_2d_map/semantic_2d_map.json`
- `06_2d_map/semantic_2d_map.png`
- `config_snapshot.json`

## WHAT WORKS NOW

- One command creates the complete object → semantic region → layered traversability → Grounding DINO attempt → comparison/scene graph → image-plane 2D map chain.
- Existing Qwen/open-semantic evidence is reused with source paths and unmodified raw output references.
- Physical occupancy, explicit No-Go, semantic regions, traversable candidate, and mobile-robot state are separate layers.
- All JSON and PNG outputs are machine-readable/decodable and hashes are recorded.

## WHAT TO DO NEXT

1. Human-review the object boxes, target-region polygons, and floor mask; complete the existing validation form before claiming semantic performance.
2. Capture calibration/control points for this exact camera scene (minimum 4, recommended 8–12) and validate reprojection error before metric map generation.
3. If Grounding DINO is BLOCKED, resolve only the recorded model-access/runtime cause in an isolated environment; do not modify the validated Qwen environment packages.
4. Replace bbox occupancy with validated segmentation/footprints before any visualization, dry-run, or robot planning gate.
