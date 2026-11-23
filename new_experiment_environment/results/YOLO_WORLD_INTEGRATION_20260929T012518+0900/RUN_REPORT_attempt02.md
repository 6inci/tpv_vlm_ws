# YOLO-World Integration RUN REPORT

## Status

YOLO-World: PARTIAL

- Technical inference execution: `PASS`; required-class coverage is `PARTIAL`.

- Model: `yolov8s-world.pt`
- Ultralytics: `8.4.164`
- Isolated environment: `/home/binci/miniconda3/envs/yolo_world`
- CUDA used: `True` (`NVIDIA GeForce RTX 5070 Ti Laptop GPU`)
- Detections: `39`
- Detected classes: `{"cardboard box": 25, "cart": 2, "storage rack": 5, "traffic cone": 3, "truck": 3, "workbench": 1}`
- Conveyor detected: `False`
- Manipulator detected: `False`
- Left manipulators (x-center < 650 px): `0`
- Central manipulators (650 <= x-center < 1000 px): `0`

## Detector comparison

- Ground truth is unavailable; no accuracy claim is made.
- Qwen: `31` detections.
- Grounding DINO: `58` detections.
- YOLO-World: `39` detections.
- Qwen ↔ YOLO compatible one-to-one matches: `19`.
- Grounding DINO ↔ YOLO compatible one-to-one matches: `27`.
- YOLO duplicate pairs (same canonical label, bbox IoU >= 0.50): `0`.
- Qwen previously had conveyor=`False` and left manipulator count=`0` by the same spatial rule.

## SAM2 integration

- Existing Grounding DINO → SAM2 artifacts remained unchanged and were read only.
- New YOLO-World → SAM2 masks: `38` using `facebook/sam2.1-hiera-tiny` on `cuda`.
- Grounding-DINO-mask ↔ YOLO-mask comparable pairs: `26`.
- Mask IoU is diagnostic only because there is no segmentation ground truth.

## Synthetic metric integration

- Status: `PARTIAL`.
- Projected detections: `38`.
- YOLO bbox occupancy: `1.8979 m²`.
- YOLO→SAM2 occupancy: `1.1687 m²`.
- The reused homography is synthetic feasibility geometry, not real-camera calibration and not Nav2-ready.

## Environment protection

- `qwen_vl` conda package-list hash unchanged: `True`.
- `qwen_vl` conda metadata hash unchanged: `True`.
- `qwen_vl` Torch/Transformers after run: `2.13.0+cu130` / `5.16.1`.
- No sudo, apt, driver, CUDA, ROS2, or kernel changes were made.
- Ultralytics first import created `/home/binci/.config/Ultralytics/settings.json`.
- YOLO weight is run-local (`07_yolo_world/yolov8s-world.pt`, SHA-256 `095f5266bb9b654bd5ad9e21e9cdeda78e0f2c8460f5d652eaf04bab7ee251cf`).
- The first inference config fallback stored CLIP at `/home/binci/weights/clip/ViT-B-32.pt` (SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`); this did not alter `qwen_vl` but was not fully run-local.

## WHAT WORKS NOW

- Isolated YOLO-World CUDA inference with the requested 12-class vocabulary.
- Three-detector class/count/IoU diagnostic comparison.
- Separate Grounding DINO→SAM2 and YOLO-World→SAM2 paths.
- YOLO bbox/mask projection through the preserved synthetic homography.

## WHAT TO DO NEXT

1. Human-review YOLO boxes and both SAM2 mask paths on the source image.
2. Add ground-truth annotations before reporting detector or mask accuracy.
3. Repeat pixel→metric validation with real undistorted camera data and measured control points.
