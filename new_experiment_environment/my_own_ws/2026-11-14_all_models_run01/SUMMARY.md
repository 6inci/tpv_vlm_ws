# 2026-11-14 all-models detection (exploratory)

- input: `/home/binci/vlm_ws/new_experiment_environment/final_experiment_picture.png` sha256 `a543977af78ef4c92d9ed62f7c5058d14eb96913efeeb16805e6d6cf645d7a74`
- 1922x1119 RGBA screenshot of the OpenCV 'UNDISTORTED' window (title bar + borders included). Not a raw camera frame.
- coordinates: pixels of the input screenshot (x right, y down, origin top-left). Undistortion state as displayed in that window; not usable for homography.
- git commit: NO_COMMIT
- latency = one timed call after warmup (Florence OVD = sum over 12 class prompts); VRAM = torch peak allocated in that process stage.

| panel | n det | unique labels | latency s | peak VRAM GiB | top labels |
|---|---|---|---|---|---|
| qwen | 0 | 0 | 59.492 | 9.00 |  |
| qwen SALVAGED | 5 | 3 | 59.492 | 9.00 | parse_error; regex-recovered unique items |
| qwen_attempt02 | 0 | 0 | 58.194 | 9.00 |  |
| qwen_attempt02 SALVAGED | 9 | 4 | 58.194 | 9.00 | parse_error; regex-recovered unique items |
| qwen_attempt03 | 0 | 0 | 59.466 | 9.00 |  |
| qwen_attempt03 SALVAGED | 10 | 4 | 59.466 | 9.00 | parse_error; regex-recovered unique items |
| dino | 26 | 14 | 0.219 | 2.00 | traffic cone×11, safety fence×2, pallet×2, truck×1, mobile robot×1, workbenchor belt×1, robot arm×1, mobile robot robot arm×1 |
| sam2_on_dino | 26 | 14 | 0.087 | 1.38 | traffic cone×11, safety fence×2, pallet×2, truck×1, mobile robot×1, workbenchor belt×1, robot arm×1, mobile robot robot arm×1 |
| yolo_world | 0 | 0 | 0.016 | 0.75 |  |
| yoloe_pf | 57 | 31 | 0.034 | 0.56 | clip art×9, remove×5, extrude×5, clip×5, assemble×4, combination lock×2, fax×2, toy×2 |
| florence_base_OD | 3 | 2 | 0.122 | 0.64 | wheel×2, land vehicle×1 |
| florence_base_DENSE_REGION_CAPTION | 3 | 2 | 0.138 | 0.64 | wheel×2, 3D model of forklift truck in parking lot×1 |
| florence_base_OPEN_VOCABULARY_DETECTION | 13 | 12 | 1.159 | 0.64 | mobile robot×2, truck×1, pallet×1, cardboard box×1, storage rack×1, workbench×1, cart×1, drum×1 |
| florence_large_OD | 3 | 2 | 0.198 | 1.80 | wheel×2, land vehicle×1 |
| florence_large_DENSE_REGION_CAPTION | 3 | 2 | 0.225 | 1.80 | wheel×2, white truck with cardboard box on back×1 |
| florence_large_OPEN_VOCABULARY_DETECTION | 12 | 12 | 1.979 | 1.80 | truck×1, pallet×1, cardboard box×1, storage rack×1, workbench×1, cart×1, drum×1, mobile robot×1 |
