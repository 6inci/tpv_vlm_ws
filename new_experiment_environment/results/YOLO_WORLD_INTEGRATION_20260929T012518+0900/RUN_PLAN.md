# YOLO-World Integration Run Plan

1. Record `qwen_vl` package/GPU state without modifying it.
2. Clone `qwen_vl` to isolated conda environment `yolo_world`.
3. Install only Ultralytics requirements inside `yolo_world`, preserving cloned Torch/CUDA.
4. Run `yolov8s-world.pt` on the fixed 1920x1080 source image with the requested 12-class vocabulary.
5. Save detections, overlay, prompt, runtime, and result under this new run.
6. Compare YOLO-World with the existing Qwen and Grounding DINO artifacts without accuracy claims.
7. Use YOLO-World boxes as a separate SAM2 prompt path in `qwen_vl`.
8. Project YOLO boxes and masks through the existing synthetic homography and report feasibility limits.
9. Verify `qwen_vl` package state is unchanged and write a new `RUN_REPORT.md`.

Each distinct failure is limited to three practical attempts. Existing runs are read-only.
