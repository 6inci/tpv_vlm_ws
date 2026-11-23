# RUN REPORT

- Environment inventory: PASS
- Synthetic geometry/homography: PARTIAL
- Qwen: BLOCKED
- Grounding DINO: PASS
- YOLO-World: BLOCKED
- SAM2: PASS
- Metric map: PARTIAL
- Semantic/geometric integration: PARTIAL
- MCP: PARTIAL
- End-to-end: PARTIAL

## Input and geometry

- Input: `/home/binci/vlm_ws/new_experiment_environment/warehouse_robotics_1920x1080.png`
- SHA-256: `5e1cdda9fec9a3840396dd60cb0743dd2d519fd0884587c1c0156d89a565e331`
- Tile geometry: 3×2 square mats, each 1.05 m; nominal workspace 3.15×2.10 m.
- World coordinates: bottom-left `(0,0)`, X right, Y up, metres.
- Control points: 12 total (8 fit, 4 holdout).
- Homography image pixel → world XY: `[[0.00381231257012703, 0.0020863699953941866, -2.3527524010414385], [0.0002858152850570536, -0.0036917180262037125, 3.5241862305590628], [0.00012945233307578034, 0.0009772692977421055, 1.0]]`
- Calibration RMSE: `0.070252 m`; max `0.121891 m`.
- Holdout RMSE: `0.091300 m`; max `0.114344 m`.
- Geometry claim: synthetic software/pipeline feasibility only.

## Model and stage evidence

- Qwen attempts: `[{"attempt": 1, "purpose": "fresh object detection", "status": "PARTIAL", "seconds": 71.55070551300014, "objects": 0}, {"attempt": 2, "purpose": "fresh semantic regions/relations", "status": "PASS", "seconds": 42.20023730699904, "regions": 6, "relations": 0}]`
- Grounding DINO attempts: `[{"attempt": 1, "hypothesis": "local cache", "status": "PASS"}, {"attempt": 2, "hypothesis": "inference on cuda", "status": "PASS", "detections": 58}]`
- YOLO-World checks: `[{"attempt": 1, "check": "qwen_vl import ultralytics/mmdet", "available": false}, {"attempt": 2, "check": "env_isaaclab import ultralytics/mmdet", "available": false}, {"attempt": 3, "check": "workspace YOLO-World model cache", "available": false, "paths": []}]`
- SAM2 attempts: `[{"attempt": 1, "hypothesis": "local cache", "status": "BLOCKED", "error": "OSError(\"Can't load image processor for 'facebook/sam2.1-hiera-tiny'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'facebook/sam2.1-hiera-tiny' is the correct path to a directory containing a preprocessor_config.json file\")"}, {"attempt": 2, "hypothesis": "download through installed Transformers; no package change", "status": "PASS"}, {"attempt": 3, "hypothesis": "bbox-prompt inference on cuda", "status": "PASS", "masks": 39}]`
- Map resolution: `0.01 m/px`, raster `[315, 210]`. This visualization resolution does not imply 1 cm calibration accuracy.
- Bbox occupancy/traversability and SAM2 occupancy/traversability are both preserved.
- No package/system/CUDA/ROS2 change and no robot motion occurred.

## Failures and limits

- YOLO-World is BLOCKED because no installed implementation/model cache exists and project rules require approval before package installation.
- Geometry is PARTIAL because the generated tile grid is not perfectly projective and points were manually audited.
- Qwen, Grounding DINO, and SAM2 outputs lack ground-truth annotation; no accuracy claim is made.
- MCP is a locally tested read-only prototype, not externally client-validated infrastructure.

## Artifacts

- `00_inventory/ENVIRONMENT_INVENTORY.md`
- `00_inventory/inventory.json`
- `01_qwen/OBJECT_PROMPT.txt`
- `01_qwen/RESULT.md`
- `01_qwen/SEMANTIC_PROMPT.txt`
- `01_qwen/object_overlay.png`
- `01_qwen/objects.json`
- `01_qwen/objects_raw.txt`
- `01_qwen/runtime.json`
- `01_qwen/semantic_overlay.png`
- `01_qwen/semantic_regions.json`
- `01_qwen/semantics_raw.txt`
- `03_metric_map/RESULT.md`
- `03_metric_map/map_metadata.json`
- `03_metric_map/metric_map.json`
- `03_metric_map/metric_map.png`
- `03_metric_map/metric_map_bbox.png`
- `03_metric_map/occupancy_metric.png`
- `03_metric_map/occupancy_metric_bbox.png`
- `03_metric_map/semantic_metric.png`
- `03_metric_map/traversability_metric.png`
- `03_metric_map/traversability_metric_bbox.png`
- `04_grounding_dino/PROMPT.txt`
- `04_grounding_dino/RESULT.md`
- `04_grounding_dino/detections.json`
- `04_grounding_dino/grounding_dino_overlay.png`
- `06_detector_comparison/RESULT.md`
- `06_detector_comparison/detector_comparison.json`
- `07_yolo_world/RESULT.md`
- `07_yolo_world/detections.json`
- `07_yolo_world/overlay.png`
- `08_sam2/RESULT.md`
- `08_sam2/masks/gd_001.png`
- `08_sam2/masks/gd_002.png`
- `08_sam2/masks/gd_003.png`
- `08_sam2/masks/gd_004.png`
- `08_sam2/masks/gd_005.png`
- `08_sam2/masks/gd_006.png`
- `08_sam2/masks/gd_007.png`
- `08_sam2/masks/gd_008.png`
- `08_sam2/masks/gd_009.png`
- `08_sam2/masks/gd_010.png`
- `08_sam2/masks/gd_011.png`
- `08_sam2/masks/gd_012.png`
- `08_sam2/masks/gd_013.png`
- `08_sam2/masks/gd_014.png`
- `08_sam2/masks/gd_015.png`
- `08_sam2/masks/gd_016.png`
- `08_sam2/masks/gd_017.png`
- `08_sam2/masks/gd_018.png`
- `08_sam2/masks/gd_019.png`
- `08_sam2/masks/gd_020.png`
- `08_sam2/masks/gd_021.png`
- `08_sam2/masks/gd_022.png`
- `08_sam2/masks/gd_023.png`
- `08_sam2/masks/gd_024.png`
- `08_sam2/masks/gd_026.png`
- `08_sam2/masks/gd_027.png`
- `08_sam2/masks/gd_029.png`
- `08_sam2/masks/gd_030.png`
- `08_sam2/masks/gd_031.png`
- `08_sam2/masks/gd_032.png`
- `08_sam2/masks/gd_034.png`
- `08_sam2/masks/gd_035.png`
- `08_sam2/masks/gd_037.png`
- `08_sam2/masks/gd_039.png`
- `08_sam2/masks/gd_040.png`
- `08_sam2/masks/gd_043.png`
- `08_sam2/masks/gd_044.png`
- `08_sam2/masks/gd_046.png`
- `08_sam2/masks/gd_049.png`
- `08_sam2/segmentation.json`
- `08_sam2/segmentation_overlay.png`
- `09_mcp/README.md`
- `09_mcp/RESULT.md`
- `09_mcp/demo/demo_response.json`
- `09_mcp/server/server.py`
- `09_mcp/tool_schema.json`
- `10_integration/RESULT.md`
- `10_integration/metric_scene_graph.json`
- `config_snapshot.json`
- `geometry/RESULT.md`
- `geometry/control_points.json`
- `geometry/control_points_overlay.png`
- `geometry/homography.json`
- `geometry/reprojection_validation.png`

## WHAT WORKS NOW

- Fresh Qwen and Grounding DINO inference on the new 1920×1080 image.
- 8-point homography fit with 4 independent holdout points and numeric residuals.
- Grounding DINO bbox → SAM2 masks → metric occupancy/traversability conversion.
- Bbox and SAM2 metric maps, integrated metric scene graph, and read-only scene-query prototype.

## WHAT IS ONLY A SYNTHETIC/FEASIBILITY TEST

- Every homography-derived XY, footprint, occupancy area, traversability layer, and metric map in this run.
- The 3.15×2.10 m scale is imposed from the synthetic mat specification, not measured from a real camera scene.
- Detector/SAM2 comparison is diagnostic only and does not establish accuracy.

## WHAT MUST BE REPEATED WITH THE REAL CAMERA

- Capture the fixed camera's undistorted 1920×1080 frame and confirm current intrinsic calibration/camera pose.
- Re-click or automatically measure 8–12 real floor pixel↔world points and repeat holdout residual validation.
- Re-run all detectors/SAM2 on the real frame and human-review object masks, no-go regions, and robot footprint inflation.
- Validate the metric map against physical measurements before visualization, dry-run, planning, or Nav2 use.

## WHAT TO DO NEXT

1. Review control-point and reprojection overlays; replace any disputed synthetic point and rerun once.
2. Provide approval for an isolated YOLO-World environment if the three-detector comparison is still required.
3. Annotate a small real-image detection/segmentation reference set before threshold tuning or performance claims.
4. Repeat calibration and metric-map validation with the real undistorted camera frame.
