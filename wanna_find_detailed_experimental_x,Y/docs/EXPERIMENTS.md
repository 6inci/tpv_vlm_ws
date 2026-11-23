# Experiments

결과가 존재하지 않는 실험은 `PLANNED`로만 표기한다. 실행되지 않은 실험에 metric이나 PASS/FAIL 값을 기록하지 않는다.

## Registry

| ID | Purpose | State | Raw Result Path | Processed Result Path | Result |
|---|---|---|---|---|---|
| EXP001 | Independent geometry validation | PLANNED | 미생성 | 미생성 | 미실행 |
| EXP002 | Robot world localization | PLANNED | 미생성 | 미생성 | 미실행 |
| EXP004 | VLM semantic-metric grounding | PLANNED | 미생성 | 미생성 | 미실행 |
| EXP012 | Approach and grasp integration | PLANNED | 미생성 | 미생성 | 미실행 |
| EXP013 | Object transport | PLANNED | 미생성 | 미생성 | 미실행 |
| EXP014 | Release and final verification | PLANNED | 미생성 | 미생성 | 미실행 |
| EXP016 | End-to-end task | PLANNED | 미생성 | 미생성 | 미실행 |

## Run Record Template

```yaml
experiment_id: EXP001
run_id: run01
purpose: ""
date: ""
hardware_configuration: ""
software_commit: ""
dataset: ""
parameters: ""
raw_result_path: ""
processed_result_path: ""
metrics: {}
result: PASS_OR_FAIL
notes: ""
```

각 run은 `results/EXP001_run01/`처럼 새 디렉터리에 저장하고 덮어쓰지 않는다. config snapshot, raw measurement, 처리 코드/version, summary, figure, log를 함께 둔다.

## Legacy Measurements (Reference Only)

다음은 새 Workspace 실험이 아니며 논문 최종 결과로 등록하지 않는다.

- calibration RMS 0.205716 px: `/home/binci/camera_calibration_1920x1080/calibration_report.txt`
- undistorted homography 12점 적합: mean 5.34 mm, RMSE 5.74 mm, max 8.80 mm
- 같은 12점 LOO: mean 9.19 mm, RMSE 10.73 mm, max 19.94 mm

독립 reference point, current installation 증거, run provenance가 없으므로 EXP001 결과가 아니다.

## Software Verification Records (Not Paper Experiments)

### VER001 — Legacy Offline Geometry Reproduction — PASS

- Date: 2026-09-17 KST
- Input: `/home/binci/qwen_vision_test/photo_04.png`
- Reference image: `/home/binci/qwen_vision_test/photo_04_undistorted.png`
- Calibration: `/home/binci/camera_calibration_1920x1080/calibration_result.json`
- Geometry: `/home/binci/qwen_vision_test/homography_12pt_undistorted.json`
- Command: `python3 scripts/verify_legacy_geometry.py`
- Results: `/home/binci/vlm_ws/results/geometry_legacy_regression/`
- Image reproduction: mean absolute difference 0, max difference 0, changed pixel ratio 0
- 12-point reproduction: undistortion max difference 0 px, world max difference 0 m
- Legacy fitting error recalculation: mean 5.34 mm, RMSE 5.74 mm, max 8.80 mm
- Determinism: repeated command produced byte-identical PNG, two JSON files and CSV
- Interpretation: `LEGACY_OFFLINE_GEOMETRY` software reproduction PASS; not independent accuracy validation and not an EXP001 result

### VER002 — Qwen Single-Object Metric Grounding Integration — PASS

- Date: 2026-09-17 KST
- Input: `/home/binci/qwen_vision_test/photo_04.png`; target query `red cup`
- Model: `Qwen/Qwen3-VL-4B-Instruct`, existing local cache only
- Environment: `qwen_vl`; Python 3.10.20; Torch 2.13.0+cu130; Transformers 5.16.1; NVIDIA GeForce RTX 5070 Ti Laptop GPU
- Command: `/home/binci/miniconda3/envs/qwen_vl/bin/python scripts/run_qwen_object_grounding.py --image /home/binci/qwen_vision_test/photo_04.png --target "red cup"`
- Results: `/home/binci/vlm_ws/results/qwen_grounding/`
- Stored bbox integration: `[447,527,483,595]` → verified legacy world `(0.267293,0.657320) m`; PASS
- Stored rounded legacy world comparison: `(0.265,0.662) m`; new-minus-stored `(0.002293,-0.004680) m`. 기존 값은 다른 distorted-image Homography와 반올림을 사용했다.
- Fresh inference: 3/3 runs returned `[448,525,483,595]` and `(0.269240,0.657628) m`
- Repeatability: bbox, bottom-center and world X/Y peak-to-peak variation all zero in these three runs
- Timing: model load 3.514 s; inference 2.717/2.051/2.054 s
- Interpretation: semantic→pixel→metric software integration PASS; no independent object ground-truth, not final paper accuracy and not an EXP004 result

### VER003 — Multi-Entity Semantic + Metric Scene Grounding — PARTIAL

- Date: 2026-09-17 KST
- Input: `/home/binci/qwen_vision_test/photo_04.png`
- Model/environment: existing local-only `Qwen/Qwen3-VL-4B-Instruct` in `qwen_vl`; no install/download
- Command: `/home/binci/miniconda3/envs/qwen_vl/bin/python scripts/run_qwen_scene_grounding.py --image /home/binci/qwen_vision_test/photo_04.png`
- Results: `/home/binci/vlm_ws/results/multi_entity_grounding/`
- Fresh inference: 3 runs; 11/11/11 entities; inference 23.126/22.633/22.696 s
- Geometry status per run: `VALID` 3, `OUTSIDE_LEGACY_WORKSPACE` 8, `INVALID` 0
- Repeatability: 11 common entities; no appearance/disappearance; labels/classifications/bboxes/world X/Y identical across runs
- Legacy spatial comparison: red cup IoU 0.9452, white bag 0.9100, trophy-like object 0.8730. 이는 ground-truth accuracy가 아닌 bbox overlap 관찰이다.
- Limitation: fresh prompt는 legacy 세 물체 위치를 다시 찾았지만 실제 물체 이름 대신 generic `workspace_element` label/classification을 반환했다.
- Interpretation: multi-entity schema, parsing, geometry integration and visualization succeeded; semantic specificity와 workspace coverage 때문에 PARTIAL. No semantic ground truth, no obstacle footprint, not final paper accuracy and not an EXP004 result.

### VER004 — Semantic Label Specificity + Partial Reference Matching — PARTIAL

- Date: 2026-09-17 KST
- Input: `/home/binci/qwen_vision_test/photo_04.png`
- Partial reference: `/home/binci/vlm_ws/datasets/offline_dev/annotations/photo_04_semantic_reference.json`; three legacy-reviewed objects only
- Command: `/home/binci/miniconda3/envs/qwen_vl/bin/python scripts/run_qwen_semantic_refinement.py --image /home/binci/qwen_vision_test/photo_04.png`
- Results: `/home/binci/vlm_ws/results/semantic_refinement/`
- Fresh inference: 3 runs; 11/11/11 entities; all outputs identical; inference 19.972/19.011/18.896 s
- Label specificity per run: `SPECIFIC` 11, `GENERIC` 0, `UNKNOWN` 0
- Reference comparison per run: red cup→`red mug`, IoU 0.9444, specific but alias mismatch; white bag→no bbox match, best IoU 0.0652; gold trophy-like object→`trophy`, IoU 0.9153, alias PASS
- Reference gate: 1/3 PASS in all runs; result PARTIAL
- Active workspace per run: `INSIDE` 3, `OUTSIDE` 8, `INVALID_GEOMETRY` 0; OUTSIDE entities preserved
- Architecture: navigation/target/obstacle/path roles removed from new VLM output and left to future task/workspace layers
- Interpretation: generic ontology labels were removed, but all three known-reference semantic gates did not pass. Partial reference only, no full semantic ground truth, not final paper accuracy and not an EXP004 result.

### VER005 — Task-Conditioned Target Grounding — PASS

- Date: 2026-09-17 KST
- Input: `/home/binci/qwen_vision_test/photo_04.png`
- Partial reference: `/home/binci/vlm_ws/datasets/offline_dev/annotations/photo_04_semantic_reference.json`; canonical label은 query에 사용했지만 reference bbox/pixel/world coordinate는 prompt에 제공하지 않음
- Model/environment: existing local-only `Qwen/Qwen3-VL-4B-Instruct` in `qwen_vl`; Python 3.10.20; Torch 2.13.0+cu130; Transformers 5.16.1; NVIDIA GeForce RTX 5070 Ti Laptop GPU; no install/download
- Command: `/home/binci/miniconda3/envs/qwen_vl/bin/python scripts/run_qwen_task_grounding.py --image /home/binci/qwen_vision_test/photo_04.png`
- Results: `/home/binci/vlm_ws/results/task_conditioned_grounding/`
- Red cup: 3/3 `TARGET_FOUND`; label `red cup`; reference IoU 0.9452; world `(0.269240, 0.657628) m`; `INSIDE`
- White bag: 3/3 `TARGET_FOUND`; label `white bag`; reference IoU 0.9341; world `(1.387814, 0.584694) m`; `OUTSIDE`; broad scene `NO_BBOX_MATCH`에서 복구
- Gold trophy-like object: 3/3 `TARGET_FOUND`; label `gold trophy-like object`; reference IoU 0.9234; world `(0.699674, 1.467030) m`; `INSIDE`
- Repeatability: target별 세 run의 label/bbox/world가 동일; bbox IoU vs run01 1.0; world X/Y peak-to-peak 0 m
- Timing: model load 1.852 s; red cup inference 2.965/2.445/2.431 s; white bag 2.402/2.404/2.389 s; trophy 2.434/2.482/2.487 s
- Candidate safety: 0개/다중/invalid output test에서 각각 `TARGET_NOT_FOUND`/`TARGET_AMBIGUOUS`/`INVALID_OUTPUT`; 다중 후보 좌표 미선택
- Interpretation: known target phrase를 이용한 task-conditioned recovery와 기존 geometry integration PASS. Partial reference only, task-conditioned evaluation, legacy geometry, not final paper semantic accuracy, no full task parser yet; not an EXP004 result.

### VER006 — Offline Metric Goal Zone Definition + Verification — PASS

- Date: 2026-09-17 KST
- Zone config: `/home/binci/vlm_ws/config/zones.json`
- Inputs: legacy geometry and `/home/binci/qwen_vision_test/photo_04_undistorted.png`; STEP 08 run01 JSON three files read-only
- Command: `python3 scripts/verify_goal_zones.py`
- Results: `/home/binci/vlm_ws/results/goal_zones/`
- Definition: `LEGACY_WORKSPACE_COORDINATE`; meter; X 0.0–1.2, Y 0.0–1.8; grid 0.6 m; `zone_A`~`zone_F`; `OFFLINE_DEVELOPMENT_ZONE_SET`
- Config validation: 6 unique IDs; 30 unique configured aliases; all polygon vertices within bounds; all centers strictly inside; polygon interior overlap 0 m²; shared edges allowed
- Alias validation: all 30 configured aliases resolved to their expected zone. `저쪽`, `왼쪽`, `저기`, `가까운 곳` returned `ZONE_NOT_RESOLVED`.
- Boundary policy: `(0.6, 0.3)` returned `BOUNDARY` with `zone_A`, `zone_B`; no silent selection
- STEP 08 integration: red cup `(0.269240, 0.657628) m`→`zone_C`; gold trophy-like object `(0.699674, 1.467030) m`→`zone_F`; white bag `(1.387814, 0.584694) m`→`OUTSIDE_WORKSPACE`
- Visualization: undistorted-image polygon overlay and metric top-down diagram generated. Inverse Homography was used for visualization only; raw geometry unchanged.
- Tests: existing 27 plus 11 new zone tests; 38/38 PASS in default and `qwen_vl` Python environments
- Interpretation: deterministic offline zone definition/lookup/integration PASS. Offline development zones only; no physical validation, robot/object footprint, feasibility, task parser or final-paper configuration.

### VER007 — Natural-Language Task Parser + Resolution — PARTIAL

- Date: 2026-09-17 KST
- Dataset: `/home/binci/vlm_ws/datasets/offline_dev/task_instructions/task_parser_cases.json`; 6 valid, 5 invalid/incomplete cases
- Model: existing local-only `Qwen/Qwen3-VL-4B-Instruct`; text-only mode; no install/download
- Environment: `qwen_vl`; Python 3.10.20; Torch 2.13.0+cu130; Transformers 5.16.1; CUDA unavailable during final parser run
- Command: `/home/binci/miniconda3/envs/qwen_vl/bin/python scripts/run_task_parser.py`
- Results: `/home/binci/vlm_ws/results/task_parser/`; prompt development attempts preserved in `task_parser_attempt01/` and `task_parser_attempt02/`
- Final valid metrics: parse/integration 3/6; action field 6/6; target phrase 3/6; destination zone 6/6
- Final invalid metrics: correct rejection 4/5. Missing target, missing destination, unknown destination and ambiguous target passed; rotate command was misclassified as `reposition` then rejected as missing destination instead of `UNSUPPORTED_ACTION`.
- Language split: English valid 3/3 PASS. Korean valid 0/3 integration PASS because model preserved Korean target phrases rather than canonical English phrases; action and destination zone remained correct.
- Repeatability: Korean red cup, Korean trophy and English white bag each ran 3 times; action, target phrase, destination phrase and zone ID were identical within each instruction. Deterministic repetition did not correct the Korean normalization error.
- Integration evidence: English red cup→STEP 08 `TARGET_FOUND`, world `(0.269240, 0.657628) m`, `INSIDE`; destination `Zone A`→STEP 09 `zone_A`, center `(0.3, 0.3) m`; feasibility `NOT_EVALUATED`.
- Unit/regression tests: existing 38 plus 8 new parser tests; 46/46 PASS. Unit success does not override observed model integration failures.
- Interpretation: schema validation, deterministic destination resolution and English offline integration work, but multilingual normalization and unsupported-action classification require rework. Offline development only; not a final language benchmark.

### VER008 — Deterministic Task Canonicalization + Unsupported Action Guard — PASS

- Date: 2026-09-17 KST
- Resumed state: usage-limit interruption 전에 생성된 `config/task_language.json`, `task_parser.json` 연결과 English rotate case를 보존하고 parser/CLI/tests/integration부터 이어서 완료
- Dataset: existing 6 valid + existing 5 invalid core cases; additional `Rotate the red cup.` regression. 기존 expected result는 변경하지 않음.
- Model: existing local-only `Qwen/Qwen3-VL-4B-Instruct`; text-only surface extraction; no install/download
- Execution environment: `qwen_vl`; Python 3.10.20; Torch 2.13.0+cu130; Transformers 5.16.1; CUDA unavailable, CPU inference
- Command: `/home/binci/miniconda3/envs/qwen_vl/bin/python scripts/run_task_parser.py`
- Results: `/home/binci/vlm_ws/results/task_parser_rework/`; first PARTIAL attempt preserved at `/home/binci/vlm_ws/results/task_parser_rework_attempt01/`
- Core metrics: valid integration 6/6; canonical action 6/6; canonical target phrase 6/6; destination zone 6/6; invalid rejection 5/5
- Additional regression: English rotate 1/1 `UNSUPPORTED_ACTION`
- Korean integration: red cup→`red cup`/`zone_A`/`TARGET_FOUND`/`INSIDE`; white bag→`white bag`/`zone_B`/`TARGET_FOUND`/`OUTSIDE`; trophy→`gold trophy-like object`/`zone_F`/`TARGET_FOUND`/`INSIDE`
- Action guard: `빨간 컵을 회전시켜`와 `Rotate the red cup.`은 raw instruction guard로 차단됐으며 target grounding, zone resolution과 path status는 `NOT_RUN`
- Repeatability: Korean red cup, Korean trophy, English white bag 각각 3회; action surface/canonical action/target surface/grounding phrase/destination surface/zone 모두 동일
- Tests: previous 46 plus 4 new deterministic resolver/guard tests; 50/50 PASS
- Interpretation: STEP 10 두 failure의 deterministic rework PASS. Offline development only, limited human-reviewed aliases, not a language benchmark, no feasibility or planning.

### VER009 — Unified Workspace / Task State Construction — PASS

- Date: 2026-09-17 KST
- Inputs: saved STEP 11 Korean valid run01 tasks, STEP 08 target run01 results, STEP 07 semantic run01 entities, STEP 09 `zones.json`; no new Qwen inference, geometry estimation or parsing
- Command: `python3 scripts/build_workspace_state.py`
- Results: `/home/binci/vlm_ws/results/workspace_state/`
- State results: red cup→`zone_A`/`INSIDE`, white bag→`zone_B`/`OUTSIDE`, gold trophy-like object→`zone_F`/`INSIDE`; all `VALID_REPRESENTATION`
- Scene: STEP 07 PARTIAL source의 11 entities와 semantic/geometry/reference status를 보존
- Association: red cup→`entity_007`, IoU 0.9437; trophy→`entity_009`, IoU 0.9717; white bag→null, best IoU 0.0632. Association은 same-visual-entity candidate이며 semantic ground truth나 persistent identity가 아님.
- Validation: invalid task, target-not-found, unresolved zone, frame mismatch와 non-finite coordinate reject; workspace `LEGACY_WORKSPACE_COORDINATE`/`m`, bounds X 0.0–1.2, Y 0.0–1.8
- Deferred state: feasibility `NOT_EVALUATED`, planning `NOT_RUN`, robot `NOT_AVAILABLE`; no robot goal pose, path, footprint or obstacle role
- Determinism: 동일 saved inputs를 두 번 build한 semantic dict가 동일
- Tests: previous 50 plus 10 new state tests; system Python and `qwen_vl` both 60/60 PASS
- Interpretation: offline unified representation integration PASS. Scene semantics remain PARTIAL; legacy geometry and development zones are not physical/final-paper validation.

### VER010 — Physical Geometry + Planning-Relevant Footprint Layer — PASS

- Date: 2026-09-17 KST
- Audit roots: `/home/binci/vlm_ws`, `/home/binci/qwen_vision_test`, read-only `/home/binci/mecanum_ros2`
- Reference repository: commit `3c710ab27209eaa196facb6a88507c42447837a8`; modified by this step: false
- Confirmed robot source: `mecanum_bringup/config/nav2_params.yaml` local/global footprint `[[0.145,0.123],[0.145,-0.123],[-0.145,-0.123],[-0.145,0.123]] m`; `base_link`; extent 0.290×0.246 m
- Additional audited reference: wheel radius 0.04 m and CAD wheel-center spacing 0.192091×0.207001 m. These values were recorded but not used to replace the configured footprint.
- URDF simplified chassis collision box 0.25×0.20×0.08 m was recorded but not selected as the full robot footprint; the explicit larger Nav2 planning footprint remains authoritative.
- Excluded: Nav2 padding 0.01 m, inflation radius 0.55 m and all Qwen bboxes; none are treated as physical footprint dimensions.
- Unknown: gripper geometry/offset, red cup/white bag/trophy footprints, selected chair/desk polygons
- Role integration: per red/trophy state, 1 `TASK_TARGET`, 2 `DYNAMIC_ENTITY`, 4 `STATIC_OBSTACLE_CANDIDATE`, 4 `UNASSIGNED`; white bag has no scene association and therefore 5 `UNASSIGNED`
- Output: three separate physical planning states; STEP 12 inputs unchanged; representation `VALID`; robot `READY`; gripper/object/static obstacles `NOT_READY`; overall `PARTIAL`
- Deferred: collision checking, inflation, costmap, combined robot-object footprint, grasp/approach pose and planning all `NOT_RUN` or unimplemented
- Visualization: confirmed robot footprint in a separate `base_link` local-frame panel; workspace shows reference points only because robot pose and obstacle polygons are unavailable
- Command: `python3 scripts/audit_physical_geometry.py`
- Results: `/home/binci/vlm_ws/results/physical_geometry/`
- Tests: previous 60 plus 12 new physical-geometry tests; system Python and `qwen_vl` both 72/72 PASS
- Interpretation: representation-layer PASS with planning geometry readiness `PARTIAL`; missing physical measurements must be acquired before collision validation.

### VER011 — Static Obstacle + Robot Collision Geometry — PASS

- Date: 2026-09-17 KST
- Inputs: `zones.json`, `physical_geometry.json`, STEP 13 readiness, `static_obstacles.json`; no VLM inference or geometry estimation
- Command: `python3 scripts/build_collision_geometry.py`
- Results: `/home/binci/vlm_ws/results/collision_geometry/`
- Robot: confirmed `base_link` rectangle 0.290×0.246 m, half extents X 0.145 m/Y 0.123 m, fixed yaw 0
- Workspace: raw X 0.0–1.2 m/Y 0.0–1.8 m; robot-center valid bounds X 0.145–1.055 m/Y 0.123–1.677 m
- Obstacle: one explicit `DEVELOPMENT_SYNTHETIC` rectangle, raw X 0.45–0.65 m/Y 0.65–1.15 m; not measured and not VLM-derived
- Expanded collision bounds: X 0.305–0.795 m/Y 0.527–1.273 m; safety margin/padding/inflation 0
- Policy: obstacle and workspace contact are `COLLISION`; continuous metric query, no occupancy grid
- Checks: obstacle interior collision, outside free, exact obstacle contact collision, exact workspace contact collision; invalid frame/unit/non-finite/geometry rejected
- Readiness: robot-only static planning geometry `READY`; overall manipulation geometry `PARTIAL`
- Deferred: path planning `NOT_RUN`; A*, dynamic replanning, robot execution and manipulation unimplemented
- Determinism: identical saved inputs build equal semantic world/summary data; STEP 12/13 artifact hashes unchanged
- Tests: previous 72 plus 19 collision-geometry tests; system and `qwen_vl` Python both 91/91 PASS
- Interpretation: deterministic offline robot-only collision representation PASS. Synthetic development geometry only; not real-scene obstacle validation or final-paper planning evidence.

### VER012 — Binary Occupancy + Safety-Cost 8-Connected A* — PASS

- Date: 2026-09-17 KST
- Inputs: saved STEP 14 `static_collision_world.json`, `planner_baseline.json`; no VLM inference, live camera, robot pose or ROS execution
- Command: `python3 scripts/run_path_planning_baseline.py`
- Results: `/home/binci/vlm_ws/results/path_planning_baseline/`
- Grid: raw workspace cell-center convention; 0.02 m; 90 rows × 60 columns; FREE 2,638; OCCUPIED 2,762; no second robot-footprint inflation
- Safety layer: clearance from expanded obstacles and valid-center boundary; linear risk radius 0.20 m; weight 2.0; occupied cells null/blocked; offline development representation, not Nav2 Costmap2D
- Development endpoints: input start `(0.20,0.30) m`→cell `[15,10]`/center `(0.21,0.31) m`; input goal `(0.95,1.50) m`→cell `[75,47]`/center `(0.95,1.51) m`
- Movement: fixed `N, NE, E, SE, S, SW, W, NW`; cardinal 0.02 m; diagonal `sqrt(2)×0.02 m`; adjacent orthogonal cells required for diagonal motion
- Distance-only: PATH_FOUND; 1.740833 m; 4.401 ms observed; 1,286 nodes; minimum clearance 0.015 m; accumulated risk 1.355386; 81 waypoints
- Safety-cost: PATH_FOUND; 1.846274 m; 5.868 ms observed; 1,765 nodes; minimum clearance 0.065 m; accumulated risk 0.487105; 90 waypoints
- Validation: both paths collision-free; invalid endpoints rejected; corner cutting blocked; NO_PATH explicit; no smoothing
- Determinism: occupancy, safety layer and images byte-identical; both path semantic results and comparison summary identical after excluding volatile planning time/output paths; STEP 12/13/14 hashes unchanged
- Tests: previous 91 plus 31 planner tests; system and `qwen_vl` Python both 122/122 PASS
- Interpretation: offline static planning baseline PASS. A* is a reference baseline, not research novelty; single synthetic scenario, no robustness claim, dynamic replanning, real robot, or manipulation.

### VER013 — Static Path Stress Test + Parameter Sensitivity — PASS

- Date: 2026-09-18 KST
- Inputs: `planner_stress_scenarios.json`, existing STEP 14 collision builder, STEP 15 planner/config/results; all added geometry `DEVELOPMENT_SYNTHETIC`
- Command: `python3 scripts/run_path_planning_stress.py`
- Results: `/home/binci/vlm_ws/results/path_planning_stress/`
- Scenario set: S1 baseline; S2 shifted left; S3 two obstacles with overlapping expanded regions; S4 0.030 m robot-center narrow passage; S5 full-width blocked barrier; S6 alternate start/goal
- Aggregate: 6 scenarios; PATH_EXISTS expected 5; NO_PATH expected 1; 12 planner runs; 12 expectation matches; 10 PATH_FOUND independently collision-free; 2 NO_PATH confirmations
- S1 regression: rebuilt collision world, distance path and safety path semantically identical to STEP 15 saved artifacts
- S1 metrics: distance 1.740833 m/0.015 m clearance/1.355386 risk; safety 1.846274 m/0.065 m/0.487105
- S2 metrics: distance 1.623675 m/0.015 m/0.855080; safety 1.764264 m/0.065 m/0.084824
- S3 metrics: distance 1.705685 m/0.005 m/1.395878; safety 1.797401 m/0.045 m/0.760800
- S4 metrics: both planners 1.200000 m, 0.005 m clearance, 1.093267 risk; narrow geometry constrained both to the same corridor
- S5: both planners `NO_PATH`; no partial path
- S6 metrics: distance 1.729117 m/0.015 m/1.211339; safety 1.834558 m/0.065 m/0.482326
- Resolution sweep 0.01/0.02/0.04 m: grid cells 21,600/5,400/1,350; safety nodes 6,911/1,765/426. Path metrics changed with discretization; no resolution was selected as best.
- Safety-radius sweep 0.10/0.20/0.30 m: safety path lengths 1.799411/1.846274/1.787696 m; minimum clearance remained 0.065 m because the fixed start endpoint constrained the metric; raw risk values are radius-definition-dependent.
- Safety-weight sweep 0/0.5/1/2/4: path lengths 1.740833/1.740833/1.811127/1.846274/1.874558 m; clearance 0.015/0.022672/0.065/0.065/0.065 m; accumulated risk 1.355386/0.608850/0.515521/0.487105/0.474359
- Hard-layer checks: safety radius/weight occupancy SHA-256 unchanged; all additional inflation 0; weight 0 objective equals distance-only objective
- Repeatability: each scenario rebuilt twice with equal occupancy, risk and semantic paths; separate full rerun produced 10 equal semantic JSON artifacts and 14 byte-identical deterministic PNGs. Planning time and its chart were excluded.
- Tests: previous 122 plus 31 stress tests; system and `qwen_vl` Python both 153/153 PASS
- Interpretation: tested static development cases passed robustness/sensitivity gates. This does not optimize parameters, certify physical safety, or establish a general planner-performance claim.

### VER014 — Offline Dynamic State Update + Current Path Invalidation — PASS

- Date: 2026-09-18 KST
- Inputs: `dynamic_obstacle_scenarios.json`, existing STEP 14 collision builder and STEP 15 grid/safety/A*; all event geometry `DEVELOPMENT_SYNTHETIC`
- Command: `python3 scripts/run_dynamic_path_invalidation.py`
- Results: `/home/binci/vlm_ws/results/dynamic_path_invalidation/`
- Current path policy: six scenarios each generated one initial `SAFETY_COST_ASTAR` path; all event revisions reused the exact stored path; post-event planner runs 0
- Events: 8/8 expectations matched. D1 ADD away `VALID`; D2 ADD on path `BLOCKED`; D3 MOVE onto path `BLOCKED`; D4 ADD/REMOVE `BLOCKED→VALID`; D5 MOVE onto/away `BLOCKED→VALID`; D6 ADD near path `VALID`
- Blocking result: D2/D3/D4/D5 blocking revisions reported `WAYPOINT_COLLISION`, first blocked waypoint 50, first blocked edge 49 and the responsible dynamic obstacle ID
- Soft-cost result: D6 remained `VALID`; minimum clearance changed 0.065→0.035 m and accumulated safety risk 0.487105→0.705794
- Rebuild policy: raw metric state→existing collision builder→full occupancy/safety grid; robot expansion once; grid additional inflation 0
- Edge validation: explicit tests covered between-waypoint cardinal edge intersection and diagonal corner-cut invalidation in addition to waypoint/workspace checks
- Performance observations: event collision-world rebuild 0.234–0.312 ms, grid rebuild 7.379–8.517 ms, path validation 0.090–0.280 ms in this run; volatile observations, not a benchmark
- Determinism: all six event sequences were semantically identical on immediate replay after excluding timing; revisions, states, grids, validity, blocked indices, IDs and metrics matched
- Tests: previous 153 plus 36 new tests; system and `qwen_vl` Python both 189/189 PASS; STEP 12–16 result artifacts unchanged
- Interpretation: offline dynamic state update and current-path invalidation PASS. This is not live perception or real-time replanning.

### VER015 — Event-Driven Offline Automatic Replanning — PASS

- Date: 2026-09-18 KST
- Inputs: `automatic_replanning_scenarios.json`, STEP 17 dynamic state/path validation, existing STEP 14 collision builder and STEP 15 `SAFETY_COST_ASTAR`; all event geometry `DEVELOPMENT_SYNTHETIC`
- Command: `python3 scripts/run_automatic_replanning.py`
- Results: `/home/binci/vlm_ws/results/automatic_replanning/`
- Scenario/event totals: R1–R7, 9 events; 7 initial plans; post-event planner runs 6; replan success 5; replan failure 1; valid-path keep 3; stop required 1
- R1: away obstacle left revision 0 `ACTIVE`; planner rerun 0; `CONTINUE_CURRENT_PATH`
- R2/R3: ADD/MOVE blocked future waypoint 50 after robot progress index 20. Replan started at `(0.61, 0.33) m`, retained goal `(0.95, 1.50) m`, produced revision 1; 2.092548 m, clearance 0.075 m, risk 0.707693, 1,682 nodes
- R4: full-width barrier produced `NO_PATH`; 1,130 nodes; `REPLAN_FAILED / STOP`; no partial/new active path
- R5: remaining-path clearance 0.105→0.035 m and risk 0.443781→0.662470 while `VALID`; planner rerun 0; revision 0 retained
- R6: event 1 produced revision 1; event 2 moved the blocker onto revision 1 and produced revision 2. Final path 1.637990 m, clearance 0.105 m, risk 0.446781, 1,579 nodes; lineage `0→1→2`
- R7: event 1 produced revision 1; blocker removal left revision 1 valid, so planner rerun 0 and revision 0 was not restored
- Endpoint policy tests: obstacle on current pose returned `ROBOT_POSE_INVALID / NOT_RUN / STOP`; obstacle on original goal returned `GOAL_BLOCKED / NOT_RUN / STOP`
- Validation: every successful replan passed current-pose start cell, original-goal cell, waypoint/workspace, adjacency, diagonal corner and continuous edge checks before activation
- Timing observations: collision rebuild 0.229–0.251 ms; grid rebuild 7.356–8.120 ms; current-path validation 0.118–0.211 ms; actual A* replan 3.591–5.466 ms; total event pipeline 7.882–14.425 ms. Volatile observations, not a real-time guarantee.
- Determinism: separate replay produced 7/7 timing-excluded semantic JSON matches and 16/16 byte-identical PNGs
- Tests: previous 189 plus 40 new tests; system and `qwen_vl` Python both 229/229 PASS; STEP 12–17 artifacts unchanged
- Interpretation: event-driven offline automatic replanning baseline PASS. No temporal obstacle policy, live perception, localization or robot execution was tested.

### VER016 — Dynamic Obstacle Persistence + Stop / Wait / Resume / Replan Policy — PASS

- Date: 2026-09-18 KST
- Inputs: `dynamic_obstacle_policy.json`, STEP 17 validation, STEP 18 automatic replanning, existing STEP 14 collision builder and STEP 15 `SAFETY_COST_ASTAR`; all temporal observations and dynamic geometry are synthetic development inputs
- Command: `python3 scripts/run_dynamic_obstacle_policy.py`
- Results: `/home/binci/vlm_ws/results/dynamic_obstacle_policy/`
- Parameters: logical observation interval 0.20 s; presence confirmation 2 observations; absence confirmation 2 observations; persistence threshold 1.00 s; comparison tolerance `1e-9`; all `DEVELOPMENT_PARAMETER`
- Scenario set: required P1–P10 plus optional P11; 73 logical observations; 10 blocking episodes; 5 temporary and 5 persistent classifications
- Temporary results: P1/P4/P7 resumed revision 0 after confirmed clear without A*; P9 remained stopped because blocker B still blocked; P11 episode 1 resumed and reset correctly
- Persistent results: P2/P8/P10 and P11 episode 2 each invoked the STEP 18 planner exactly once and activated revision 1; P3 invoked it once, returned `NO_PATH`, and left no partial/new active path
- Replan geometry: successful cases started at stopped revision-0 waypoint 20, `(0.61, 0.33) m`, retained goal `(0.95, 1.50) m`, and produced a 2.092548 m path with 0.075 m minimum clearance, 0.707693 accumulated risk and 1,682 expanded nodes
- Non-blocking results: P5 remained `MOVING` despite persistent off-path presence. P6 remained `VALID` while clearance changed 0.105→0.035 m and risk 0.443781→0.662470; neither scenario stopped or replanned
- Aggregate: temporary resumed 4; persistent replan triggers/planner runs 5; failed replans 1; unnecessary replans 0; stops 10; wait observations 29
- Provenance: stable `episode_NNN` identity, logical blocking start/duration/classification/clear time, path revision before/after, replan status and STEP 18 path lineage are stored in each scenario JSON
- Determinism: P1–P11 immediate replay matched semantically after excluding volatile timing; state, action, confirmation, episode identity, planner decision, revision and paths matched
- Visualization: 11 timeline PNGs and four spatial PNGs (P1/P2/P3/P10) were generated and visually inspected; machine-readable timelines are embedded in all scenario JSON files
- Tests: previous 229 plus 46 new tests; system and `qwen_vl` Python both 275/275 PASS; STEP 12–18 result artifacts unchanged
- Interpretation: DEVELOPMENT TEMPORAL POLICY PASS for deterministic offline synthetic simulation. No live camera timing, localization, ROS/Nav2 execution, hardware stop, real robot or manipulation was tested.

### VER017 — Fixed-Camera Fast Scene Monitor Interface — SOFTWARE PASS / LIVE NOT AVAILABLE

- Date: 2026-09-18 KST
- Inputs: `live_scene_monitor.json`, programmatic 160×120 BGR fixture frames, installed OpenCV/NumPy; no external dataset, Qwen, Homography, planner, ROS or robot execution
- Commands: `python3 scripts/verify_scene_monitor_offline.py`; `python3 scripts/run_live_scene_monitor.py --max-frames 30`
- Results: `/home/binci/vlm_ws/results/live_scene_monitor/`
- Camera audit: no `/dev/video*` or `/dev/v4l/by-id/*` nodes were visible. Configured `usb-HD_Web_Camera_HD_Web_Camera_Ucamera001-video-index0` was absent; no built-in fallback or capture-open attempt occurred
- Interface: strict external `CameraSource` exposes bounded open/read/close/info behavior and live frame metadata. The unavailable run terminated with `CAMERA_NOT_FOUND`, 0 captured frames and `live_external_camera_validation=NOT_AVAILABLE`
- Fast method: fixed reference → grayscale → 5×5 Gaussian blur → absdiff → threshold 25 → 3×3 close → connected components; minimum area 80 px and 0.0002 ratio; global threshold 0.35
- Tracking: deterministic greedy association sorted by IoU descending, centroid distance ascending, track ID lexical; confirmation 2 frames; retirement after more than 2 missing frames; synchronous latest-observation interface
- F1/F2: unchanged frames emitted no candidate/trigger; one-frame local change created only a tentative track and no trigger
- F3/F4/F5: persistent rectangle triggered on frame 2; moving rectangle retained `track_001`; disappeared candidate progressed through missing counts and retired
- F6: full-frame illumination change produced `GLOBAL_CHANGE`, no candidate and no semantic trigger
- F7/F8: two independent rectangles produced deterministic `track_001/track_002`; merge/split completed without error while preserving the documented identity limitation
- Aggregate: 8 scenarios, 24 frames, 15 local-change frames, 1 global-change frame and 9 semantic-trigger frames; F1–F8 timing-excluded semantic replay matched
- Performance observation: offline median 0.182 ms/frame, mean 0.685 ms, P95 0.242 ms and max 12.222 ms due to a cold-start outlier; volatile CPU observations, not a real-time guarantee
- Outputs: config snapshot, camera audit, offline/live summaries, eight machine-readable observation timelines and eight annotated PNGs
- Tests: previous 275 plus 42 new tests; system and `qwen_vl` Python both 317/317 PASS; STEP 12–19 result artifacts unchanged
- Interpretation: STEP 20 software interface and deterministic offline validation PASS; live external camera validation is independently `NOT_AVAILABLE`. Pixel observations are not metric/planner obstacles.

## Dataset Separation

- `/home/binci/qwen_vision_test/photo_01.png`~`photo_08.png`과 파생 파일은 `OFFLINE_DEVELOPMENT_DATASET`이다.
- 공통 reference는 `photo_04.png`; 대응 legacy undistorted image는 `photo_04_undistorted.png`다.
- 이 dataset과 legacy Homography로 생성한 수치는 개발 검사용이며 논문 최종 실험 결과가 아니다.
- `FINAL_PAPER_EXPERIMENTAL_DATASET`은 최종 camera pose와 workspace coordinate를 freeze한 뒤 별도로 촬영·등록한다.

### VER018 — Stored-Image Geometry Audit + Offline Real-Scene Replay — PASS

- Date: 2026-09-18 KST
- Inputs: `/home/binci/qwen_vision_test` stored images, `photo_04.png` reference, stored calibration/Homography provenance, unchanged STEP 20 scene monitor
- Commands: `python3 scripts/audit_offline_image_geometry.py`; `python3 scripts/run_offline_real_image_replay.py`
- Results: `/home/binci/vlm_ws/results/offline_image_replay/`
- Inventory: 15 images; raw-like 9, undistorted derived 1, annotated derived 2, visualization derived 3; every record includes absolute path, SHA-256, bytes, dimensions/channels/dtype, category and relation to `photo_04`
- Method: grayscale ORB 3,000 features, KNN Hamming ratio 0.75, OpenCV RNG seed 2104, diagnostic RANSAC threshold 3 px; development thresholds accepted matches 100, inliers 80, ratio 0.80, mean reprojection error 2 px, maximum corner displacement 10 px
- Registration: `photo_01/02/03/05/06/07/08.png` all passed as development candidates. Accepted matches 1,735–2,390; inliers 1,713–2,376; inlier ratio 0.983386924–0.994142259; mean reprojection error 0.084511392–0.221448243 px; maximum corner displacement 0.067397259–0.666570008 px
- Exclusion: `qwen_frame.jpg` was 640×480 `RESOLUTION_MISMATCH / NOT_ALLOWED`; six derived images were not registration/replay sources. No resize, pixel scaling or legacy-H scaling was used.
- Replay manifest: `photo_01.png`~`photo_08.png`, deterministic filename order only; this order is not claimed as the actual capture timeline
- STEP 20 real-image observation: 8 images; 7 `LOCAL_CHANGE`, 1 reference `NO_SIGNIFICANT_CHANGE`, 0 `GLOBAL_CHANGE`, 5 semantic triggers. Geometry classification and scene state are stored as independent evidence.
- Outputs: four JSON audit/manifest artifacts, generated config manifest, 28 diagnostic registration PNGs, 8 observation JSONs, 8 annotated replay PNGs and one summary JSON
- Determinism: repeated geometry audit and repeated replay were semantically equal
- Tests: previous 317 plus 32 new tests; system and `qwen_vl` Python both 349/349 PASS; STEP 12–20 protected artifacts unchanged
- Boundaries: no Qwen/model load, metric coordinate, physical obstacle geometry, planner injection, ROS/Nav2, camera driver or robot execution. `OFFLINE STORED-IMAGE REPLAY IS NOT LIVE REAL-TIME EVIDENCE`.

### VER019 — Offline Semantic + Metric Point Grounding — SOFTWARE PASS / PIPELINE PARTIAL

- Date: 2026-09-18 KST
- Inputs: STEP 21 `legacy_geometry_offline_group_01`, STEP 20 stored-image observations, existing local Qwen config/helpers and `LegacyGeometry`
- Command: `/home/binci/miniconda3/envs/qwen_vl/bin/python scripts/run_offline_semantic_metric_grounding.py`; strict V2 retry used `--image-id image_012_photo_06 --runs 1 --output-dir results/offline_semantic_metric_grounding/retry_v2_photo_06`
- Results: `/home/binci/vlm_ws/results/offline_semantic_metric_grounding/`
- Selection: 5/8 replay frames triggered; 58 confirmed candidates; `photo_04` remained non-triggered reference/control and was not a primary inference input
- Runtime: local `Qwen/Qwen3-VL-4B-Instruct`, Python 3.10.20, PyTorch 2.13.0+cu130, Transformers 5.16.1, CUDA unavailable, CPU; 8 inference runs with raw text preserved
- Primary semantics: 46 `NO_VISIBLE_ENTITY`, 1 `VISIBLE_ENTITY_FOUND`, 0 `AMBIGUOUS`; association counts 0 associated, 0 weak, 1 not-associated; metric points 0
- Rejections: `photo_02` repeated the same candidate bbox mismatch twice. `photo_06` V1 emitted a 0–1 bbox and V2 emitted raw pixel coordinates in a normalized 0–1000 field; strict validation prevented both from reaching geometry.
- Repeatability: `photo_02` raw SHA-256 was identical across two rejected runs. `photo_03` raw SHA-256 and all five parsed `NO_VISIBLE_ENTITY` results were identical across two runs.
- Boundary: scene-change bbox was never used as the metric reference. Only a validated associated VLM bbox bottom-center could call raw → stored undistortion → legacy Homography; no result satisfied that gate.
- Tests: previous 349 plus 36 new tests; system and `qwen_vl` Python both 385/385 PASS. STEP 12–21 protected artifacts are hash-checked by the new suite.
- Interpretation: implementation/software PASS, semantic metric pipeline PARTIAL. No physical footprint, planner role, planner injection, A*, replanning, ROS/Nav2 or robot execution occurred.

#### VER019 Representative Continuation — PARTIAL

- Policy: original 2016×1133 `RAW_LIKE_SOURCE`, one confirmed candidate per Qwen run, maximum three attempts; annotated overlay was audit-only and never the model source image
- Attempt 1: `photo_06 / track_011`, bbox `[1430,222,2016,1092]`, person candidate → parsed `NO_VISIBLE_ENTITY`, 97.410 s
- Attempt 2: `photo_08 / track_039`, bbox `[1017,545,1375,684]`, tennis-racket candidate → parsed `NO_VISIBLE_ENTITY`, 74.089 s
- Attempt 3: `photo_07 / track_036`, bbox `[699,451,745,518]`, paper-cup candidate → parsed `NO_VISIBLE_ENTITY`, 118.937 s
- All three raw outputs preserved exact image/track/candidate identity and used null visible bbox. Association remained `NOT_EVALUATED`; bottom-center, undistortion, Homography and world XY remained `NOT_RUN`.
- Existing exhaustive directory remained 73 files with aggregate content digest `1de4c7cb70351bfd51b4ab52e69e32548e6aecc9da3e109b324179e0498a32d9` before and after continuation.
- Result: actual inference and strict safety gates PASS; representative semantic-to-metric integration PARTIAL; valid metric points 0. No further Qwen inference was run after Attempt 3.

### VER020 — Candidate-Conditioned Observation Audit — PASS

- Date: 2026-09-18 KST
- Command: `python3 scripts/audit_candidate_conditioning.py`; no Qwen model load, processor preparation or generation call
- Inputs: three preserved representative observations, original stored images, frame results, raw outputs and current prompt constructor
- Prompt audit: raw candidate pixels were identified, but image dimensions, explicit min/max bbox order and half-open convention were absent; all three reconstructed prompts classified `LIKELY_CONFUSING`
- Bbox semantics: SceneMonitor uses OpenCV component `[x,y,x+width,y+height]`, confirming half-open bounds. `photo_06` `[1430,222,2016,1092]` is valid and touches only the right border.
- Deterministic normalized prototypes: photo_06 `[709,196,1000,964]`; photo_08 `[504,481,682,604]`; photo_07 `[347,398,370,457]`, with `ROUND_HALF_UP`
- Visual audit: photo_06 `OBJECT_PLUS_BACKGROUND`; photo_08 racket and photo_07 cup `FULL_OBJECT_LIKE`. Bbox area ratios were 22.320%, 2.179% and 0.135% respectively.
- Existing-output comparison: exhaustive photo_06 returned a visible person with invalid 0–1 bbox, whereas representative photo_06 returned `NO_VISIBLE_ENTITY` with a boundary note. Confirmed inputs differ in prompt cardinality/version and overlay versus original source; causal explanation remains `UNKNOWN`.
- Representation result: A raw-pixel-only has high coordinate ambiguity and low region salience; B normalized-only removes domain ambiguity; C overlay+normalized preserves context and adds salience with low complexity; D secondary crop adds salience but introduces medium crop-coordinate confusion/complexity.
- Recommendation: Method C for a future, separately authorized STEP 22C. Overlay/crop remain semantic-conditioning assets only.
- Evidence protection: Qwen raw outputs stayed at 11 files with identical pre/post digest `92175504f6f814c4925cc7ea96bec9734b57a630612270dfd2decf1f15433e00`; Qwen inference runs 0.

### VER021 — Method C Candidate-Conditioned VLM Revalidation — PARTIAL

- Date: 2026-09-18 KST
- Command: `/home/binci/miniconda3/envs/qwen_vl/bin/python scripts/run_candidate_conditioned_vlm_revalidation.py`
- Results: `/home/binci/vlm_ws/results/candidate_conditioned_vlm_revalidation/`
- Input: `photo_08 / image_014_photo_08 / track_039`, raw candidate `[1017,545,1375,684]`, audited normalized candidate `[504,481,682,604]`, existing full-resolution Method C highlight overlay
- Prompt policy: normalized integer 0–1000 input/output domain, explicit min/max order, full-object bbox request; raw candidate pixels omitted from the semantic prompt
- Runtime: local `Qwen/Qwen3-VL-4B-Instruct`, CPU, exactly 1 generation, 80.478 s, no retry/fallback
- Actual output: valid JSON `NO_VISIBLE_ENTITY`, null object bbox; strict parser `ACCEPTED`
- Gates: association `NOT_EVALUATED`; ground reference, LegacyGeometry and world XY `NOT_RUN`; visualization not generated because no metric result existed
- Protection: source image and all 435 protected STEP 12–22B/result/geometry artifacts retained identical before/after digest; no physical geometry or planner call
- Tests: 18 new tests; pre-inference system and `qwen_vl` Python both 425/425 PASS
- Interpretation: Method C execution and safety gates worked, but it did not yield a validated visible-object bbox. STEP 22C and STEP 22 integration remain `PARTIAL`; causality remains `UNKNOWN`.

### VER022 — Method D Secondary-Crop VLM Revalidation — PASS

- Date: 2026-09-19 KST
- Command: `/home/binci/miniconda3/envs/qwen_vl/bin/python scripts/run_secondary_crop_vlm_revalidation.py`; deterministic recovery only: `--finalize-existing`
- Results: `/home/binci/vlm_ws/results/candidate_conditioned_vlm_revalidation_method_d/`
- Inputs: IMAGE 1 original `photo_08` 2016×1133; IMAGE 2 STEP 22B context crop 466×181 from original bbox `[963,524,1429,705]`; `track_039` candidate normalized `[504,481,682,604]`
- Runtime: local `Qwen/Qwen3-VL-4B-Instruct`, CPU, actual generation exactly 1, no retry/fallback. Exact model/inference timing was not persisted before a deterministic visualization postprocess exception and is intentionally reported as `NOT_RECOVERED`.
- Actual output: `VISIBLE_ENTITY_FOUND`, `tennis racket`, `movable_object`, IMAGE 1 bbox `[504,481,682,604]`, explicit `bbox_reference_image=IMAGE_1`; strict parser `ACCEPTED`
- Association: raw object bbox `[1016.064,544.973,1374.912,684.332]`, IoU `0.9945790206`, both centroid containment checks true, `ASSOCIATED`
- Geometry: raw bottom-center `[1195.488,684.332]`, undistorted `[1098.0327,752.1130]`, world `[0.764672,0.713486] m`, frame `LEGACY_WORKSPACE_COORDINATE`, workspace `INSIDE`
- Postprocess correction: existing `annotate_metric_results` read `item["track_id"]` although the contract stores it at `scene_observation.track_id`; iteration was corrected and tested. Preserved Qwen raw SHA-256 `2d706906...fddf70` was reused without model load/generation.
- Protection: 442 protected result/geometry files retained digest `4d9e5057...60eb2c7`; STEP 22C, source image and crop remained unchanged. No footprint/planner work occurred.
- Tests: 20 new tests; pre-inference system and `qwen_vl` Python both 445/445 PASS.
- Interpretation: representative Method D integration PASS and STEP 22 final semantic-metric integration PASS. This is one development case, not a Method C/D benchmark or accuracy claim.

### VER023 — Physical Obstacle Geometry Contract — PASS

- Date: 2026-09-19 KST
- Command: `python3 scripts/audit_physical_obstacle_geometry.py`; no Qwen model load/generation and no planning call
- Input: preserved STEP 22D semantic/metric result for `photo_08 / track_039 / tennis racket`
- Source audit: no explicit tennis-racket length, width, ground footprint or observed-bottom-center anchor measurement was found in `/home/binci/vlm_ws` or `/home/binci/qwen_vision_test`; internet/general specifications were not used
- Representative result: metric point `AVAILABLE`; physical source/dimensions/anchor `UNKNOWN`; footprint `NOT_AVAILABLE`; readiness `NOT_READY_MEASUREMENT_REQUIRED`
- Development verification: reused STEP 14 `dev_obstacle_01` as `DEVELOPMENT_SYNTHETIC`, generating local rectangle `[[-0.1,-0.25],[0.1,-0.25],[0.1,0.25],[-0.1,0.25]]` and translated bounds `[0.45,0.65]×[0.65,1.15] m`
- Separation: physical geometry, uncertainty, safety margin and planning representation remain distinct; planning geometry `NOT_GENERATED`, planner injection `NOT_RUN`
- Protection: 453 protected STEP 12–22D/result/legacy-geometry artifacts retained identical digest `10e68938...bfd8`
- Tests: 26 new tests; system and `qwen_vl` Python full suites both 471/471 PASS
- Interpretation: STEP 23 software/policy is PASS; the observed racket requires physical measurement registration before any real footprint or planner adapter can exist

### VER024 — Representative Object Measurement Registration — PASS / WAITING

- Date: 2026-09-19 KST
- Command: `python3 scripts/register_representative_object_measurement.py`; no Qwen/model load and no planning call
- Audit: no actual `config/representative_object_measurement.json` or explicit `USER_PHYSICAL_MEASUREMENT` evidence was found
- Intake: null-only template created; empty READY-only registry retained with 0 objects
- Status: `WAITING_FOR_USER_MEASUREMENT`; representative physical geometry `NOT_READY`; world placement `NOT_RUN`
- Validation fixtures only: rectangle, circle and polygon; mm/cm/m conversion; invalid/non-finite dimensions; anchor/orientation gates; raw/canonical provenance and READY-only registration
- Separation: planning geometry `NOT_GENERATED`, planner injection `NOT_RUN`, Qwen inference 0, STEP 17/18/19/A*/costmap `NOT_RUN`
- Protection: 461 protected STEP 12–23/result/geometry artifacts retained identical digest `f1d56b83...e1c1`
- Tests: 22 new tests; system and `qwen_vl` Python full suites both 493/493 PASS
- Interpretation: software/schema is complete; no real physical footprint exists until the user supplies direct measurements

### VER025 — Approximate Navigation Obstacle Adapter — PASS

- Date: 2026-09-19 KST
- Command: `python3 scripts/build_navigation_obstacle_observation.py`; no Qwen, geometry recomputation or planner execution
- Input: preserved STEP 22D `image_014_photo_08 / track_039 / tennis racket`, world reference `[0.764672,0.713486] m`
- Result: `OBSTACLE_CANDIDATE / NAVIGATION_APPROXIMATION`; existing STEP 15 grid projection row 35, column 38, cell center `[0.77,0.71] m`
- Separation: physical geometry remains `NOT_READY`; no physical radius/shape/size, safety margin, occupancy mutation or planner injection was generated
- Contract audit: STEP 15 provides deterministic metric-to-grid projection; STEP 17/18 require rectangle collision geometry; STEP 19 additionally requires obstacle identity and logical observation time. STEP 25N must define this bridge.
- Navigation experiment flow: fixed third-person camera → scene observation → VLM semantic validation → metric reference point → navigation approximation → future occupancy/cost representation → future safe path → future invalidation → future replanning
- `FUTURE_EXPERIMENT_METRIC`: path length, planning time, replan count, planning success/failure, obstacle clearance and path-change magnitude. None was measured in STEP 24N.
- Experiment A/B/C/D execution: `NOT_RUN`; visualization: `NOT_GENERATED` because a point/cell JSON record is sufficient and no physical outline may be implied
- Tests: 32 adapter-contract tests passed; full regression results are recorded in the STEP 24N report

### VER026 — Navigation Dynamic-World Integration — PASS

- Date: 2026-09-19 KST
- Command: `python3 scripts/integrate_navigation_dynamic_world.py`; existing STEP 17 state/collision/grid APIs only
- Input: preserved STEP 24N observation `image_014_photo_08_track_039`, reference `[0.764672,0.713486] m`, cell `(35,38)`
- Bridge: exact raw grid-cell AABB `[0.76,0.78]×[0.70,0.72] m`, `NAVIGATION_DISCRETIZATION_CELL`; no physical size, radius, anchor or orientation
- Registration: deterministic ID `dynamic_image_014_photo_08_track_039`, ADD through `apply_obstacle_event`, revision 0→1; static obstacle state unchanged
- Occupancy: source cell was already occupied and stayed 1→1. Existing robot-footprint collision transform added 78 occupied cells around the exact cell rectangle; no cells changed 1→0.
- Interpretation: zero source-cell delta does not erase the registered dynamic observation, while the wider collision-grid delta belongs to the existing robot collision layer rather than inferred object extent.
- Execution boundary: collision world and occupancy grid rebuilt for STEP 17 state materialization; path validation, planning, STEP 18/19, A*, replanning and experiment metrics `NOT_RUN`
- Tests: 32 focused tests; final dual-environment totals are recorded in the STEP 25N report

### VER027 — Dynamic Observation Existing-Path Validity — PASS

- Date: 2026-09-19 KST
- Command: `python3 scripts/evaluate_navigation_path_validity.py`; no planner search or replanning
- Path source: preserved STEP 17 `D1.current_path`, 90 waypoints, 1.8462741699796958 m, `SAFETY_COST_ASTAR`; path digest `17646cb8...dcdc7`
- Comparison integrity: identical path grid/world sequence used in STEP 25N revisions 0 and 1
- Existing validity behavior: waypoint collision/grid checks, continuous segment/AABB intersection, transition adjacency and corner-cut validation
- Result: `VALID → VALID / NO_PATH_EFFECT`; no blocking waypoint, edge or obstacle attribution exists for this representative case
- Context: STEP 25N source cell was already occupied and its existing robot collision transform added 78 occupied cells, but they did not intersect the canonical D1 path under the existing validator
- Boundary: no path movement, obstacle movement, A*, STEP 18/19, replacement path, replanning or physical geometry
- Tests: 32 focused tests; final dual-environment totals are recorded in the STEP 26N report

### VER028 — Controlled Observation–Path Effect — PASS

- Date: 2026-09-19 KST
- Command: `python3 scripts/run_controlled_navigation_path_effect.py`; existing STEP 17 state/collision/grid/validity APIs only
- CASE A: preserved STEP 26N observed control, path `17646cb8...dcdc7`, `VALID → VALID / NO_PATH_EFFECT`
- CASE B: existing documented STEP 17 D2 `D2_add_on_path`, same stored path before/after, `VALID → BLOCKED / PATH_BLOCKED`
- Blocking provenance: waypoint 50, edge 49, cell `[36,46]`, world `[0.93,0.73] m`, occupancy `0→1`, blocker `dynamic_box_02`
- Evidence separation: CASE A `OBSERVED_NAVIGATION_APPROXIMATION`; CASE B `DOCUMENTED_DEVELOPMENT_FIXTURE`, explicitly not camera/VLM-derived
- Protection: STEP 26N nine-file digest stayed `f697b444...2c830`; 487 protected artifact digest stayed `33c57532...4a32`
- Boundary: physical geometry, Qwen, Homography, STEP 18/19, A*, path generation, replanning and ROS/Nav2 `NOT_RUN`
- Tests: 28 focused tests; system and `qwen_vl` Python full suites both 617/617 PASS

### VER029 — Path-Block Persistence Policy Integration — PASS

- Date: 2026-09-19 KST
- Input: preserved STEP 27N CASE A `VALID / NO_PATH_EFFECT` and CASE B D2 `VALID → BLOCKED / dynamic_box_02`
- Contract: reused existing STEP 19 evaluator and config; presence 2, absence 2, interval 0.2 s, persistence threshold 1.0 s
- CASE A: `MOVING / CONTINUE_CURRENT_PATH`; obstacle presence did not create a false blocking-policy trigger
- CASE B: STEP 19 P2 documented logical sequence remapped only by obstacle identity to `dynamic_box_02`; 0.2 s presence confirmation produced `STOPPED_FOR_OBSTACLE / STOP_AND_WAIT`, followed by `WAITING_FOR_CLEARANCE`
- Persistence result: blocking duration 1.0 s produced `PERSISTENT_BLOCKING / REPLAN_REQUIRED`; this is a decision output, not replanning execution
- Resume boundary: no CASE B disappearance was fabricated; existing STEP 19 P1 `RESUME_CURRENT_PATH` remains read-only `DEVELOPMENT_POLICY_FIXTURE` evidence
- Execution boundary: `execute_replanning=False`, planner runs 0, active path revision 0, STEP 18/A*/new path/ROS/Nav2/Qwen/Homography `NOT_RUN`; hardware STOP not claimed
- Physical branch: STEP 23M `WAITING_FOR_USER_MEASUREMENT`, physical geometry `NOT_READY_NOT_USED`
- Protection: STEP 27N ten-file digest stayed `ba6e3c91...9f46e`; 500 protected artifact digest stayed `1999dad5...af9628`
- Tests: 32 focused tests PASS; system Python 649/649 PASS (7.199 s), `qwen_vl` 649/649 PASS (7.064 s)

### VER030 — Persistent-Block Existing-Replanner Integration — PASS

- Date: 2026-09-20 KST
- Command: `python3 scripts/integrate_persistent_block_replanning.py`; CASE B only invokes existing STEP 18
- Gate: requires `BLOCKED / PATH_BLOCKED / PERSISTENT_BLOCKING / REPLAN_REQUIRED` plus valid `dynamic_box_02` blocking and `EXISTING_STEP19_POLICY` provenance
- CASE A: `OBSERVED_NAVIGATION_APPROXIMATION / VALID / CONTINUE_CURRENT_PATH`; STEP 18 calls 0, planner runs 0, revision 0, no replacement
- CASE B: `DOCUMENTED_DEVELOPMENT_FIXTURE`; exact STEP 17 D2 world revision 1 and obstacle, preserved path digest `17646cb8...dcdc7`, STEP 28N progress waypoint 20 at `[0.61,0.33] m`, original goal `[0.95,1.5] m`
- Existing STEP 18 result: one invocation and one planner run, `REPLAN_SUCCESS / SWITCH_TO_REPLANNED_PATH`; old revision 0 `SUPERSEDED`, accepted replacement revision 1
- Path data: old 90 waypoints / 1.8462741699796958 m; replacement 99 waypoints / 2.092548339959391 m. Timing is retained only as existing volatile STEP 18 output, not an experimental performance claim.
- Validation: existing validator returned `VALID`; path validity, current start mapping and original goal mapping checks all passed in world revision 1
- Failure contract: existing STEP 18 R4 historical evidence remains `REPLAN_FAILED / STOP`, no replacement path
- Protection: STEP 28N nine-file digest stayed `09501603...8de3e7`; previous results 500-file digest stayed `9b03d7a7...e68bc6`; planner config unchanged
- Boundary: Qwen 0; Homography, ROS/Nav2, hardware execution `NOT_RUN`; STEP 23M remains `WAITING_FOR_USER_MEASUREMENT / NOT_READY`
- Tests: 32 focused tests PASS; system Python 681/681 PASS (11.760 s), `qwen_vl` 681/681 PASS (11.173 s)

### VER031 — Offline Navigation Chain System Integration — PASS

- Date: 2026-09-20 KST
- Command: `python3 scripts/audit_offline_navigation_chain.py`; read-only aggregation and cross-step consistency assertions only
- TRACK A: STEP22D actual `image_014_photo_08 / track_039 / tennis racket`, associated bottom-center world reference `[0.764672,0.713486] m` → STEP24N cell `(35,38)` → STEP25N dynamic record → STEP26N `VALID / NO_PATH_EFFECT` → `CONTINUE_CURRENT_PATH`; STEP18 calls 0, no replacement
- TRACK B: documented STEP17 D2 `dynamic_box_02` → STEP27N `BLOCKED` → STEP28N `PERSISTENT_BLOCKING / REPLAN_REQUIRED` → STEP29N existing STEP18 call 1 → `REPLAN_SUCCESS`; `D2_path_r1` validation `VALID`, revision 0→1
- Evidence separation: TRACK A `OBSERVED_NAVIGATION_APPROXIMATION / camera-VLM-derived=true`; TRACK B `DOCUMENTED_DEVELOPMENT_FIXTURE / camera-VLM-derived=false`; tracks were not collapsed
- Physical branch: approximate navigation works without supplied dimensions, but STEP23M stays `WAITING_FOR_USER_MEASUREMENT`; physical footprint and real collision accuracy remain unsupported
- Execution boundary: no prior runner, Qwen, Homography, calibration, planner, ROS/Nav2 or hardware operation was executed in STEP30N
- Protection: 510 prior result files retained digest `3d64c51c...0606034`
- Tests: 24 focused tests PASS; final system Python 705/705 PASS (12.007 s), `qwen_vl` 705/705 PASS (11.303 s)
