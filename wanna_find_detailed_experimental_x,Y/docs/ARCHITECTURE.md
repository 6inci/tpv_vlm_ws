# Architecture

## End-to-End Data Flow

```text
Natural-language instruction
  → Task Parser: structured task
Camera frame
  → Fast Scene Monitor: pixel change candidates + stable semantic trigger
  → Later Semantic/VLM Consumer: trigger-driven, latest observation only
Stored research images
  → Inventory + raw/derived classification
  → ORB image-to-image geometry compatibility diagnostic
  → Compatible raw-image manifest
  → Existing Fast Scene Monitor: offline real-image pixel observations
  → One representative confirmed candidate + original full-resolution image
  → Triggered Qwen semantic interpretation + validated full-image bbox
  → Candidate/entity association gate
  → Associated bbox bottom-center → legacy metric point
Candidate-conditioning audit (no model execution)
  → Raw half-open candidate bbox + original image + preserved Qwen output
  → Deterministic normalized candidate + overlay/crop prototypes
  → STEP 22C representation decision only
Method C representative revalidation
  → Full-resolution candidate overlay + normalized candidate metadata
  → One strict Qwen semantic result
  → Valid full-object bbox only → original full-image raw bbox
  → Association only → bottom-center → LegacyGeometry → optional world XY
Method D representative revalidation
  → IMAGE 1 original full scene + IMAGE 2 secondary context crop
  → Crop-assisted semantic identity + explicit IMAGE 1 full-object bbox
  → Original-domain association → bottom-center → LegacyGeometry → world XY
Camera frame
  → VLM Perception: semantic label + bbox + original response
  → Vision Geometry: raw pixel → undistorted pixel → workspace X/Y
Camera frame
  → ArUco Localization: marker pose → robot-center workspace X/Y/yaw
Task + robot + objects + configured zones/obstacles/gripper
  → Validated Workspace State
  → Grasp/Approach Pose
  → Physical Feasibility Validator
  → Path Planner (robot footprint, then robot+object footprint)
  → External-Vision Controller → ROS2 velocity interface
  → Gripper close → transport → open → retreat
  → Fresh external observation → deterministic final verification
```

## Module Contracts

| Module | Input | Output | Must Not Do |
|---|---|---|---|
| `camera` | exact external device/config | timestamped raw frame | fallback selection/semantic inference |
| `fast scene monitor` | fixed reference + raw frame | pixel candidate/track/semantic trigger | semantic identity/metric geometry/planner obstacle |
| `stored-image audit/replay` | research image directory + photo_04 reference | inventory, development compatibility evidence, replay manifest/observations | resize/legacy-H scaling/metric transform/live claim |
| `offline semantic metric grounding` | triggered compatible observation + one representative candidate + original image | semantic status, association evidence, optional metric point | exhaustive inference, candidate crop as sole input, candidate bbox as object/footprint, planner injection |
| `candidate-conditioning audit` | preserved candidate/prompt/output + original image | coordinate audit, salience artifacts, future representation decision | Qwen load/generation, model-output repair, geometry/planner changes |
| `Method C revalidation` | one full-resolution overlay + normalized candidate metadata | strict semantic result, optional associated metric point | retries/fallback candidates, overlay-to-Homography, footprint/planner injection |
| `Method D revalidation` | original full scene + provenance-bound secondary crop | strict IMAGE 1 bbox, association evidence, optional metric point | crop-coordinate remap, crop-to-Homography, retry/benchmark, footprint/planner injection |
| `navigation obstacle adapter` | validated STEP 22D semantic-metric observation + existing STEP 15 grid contract | obstacle candidate with reference point and deterministic containing cell | infer physical size/center, add safety margin, mutate occupancy, execute planner |
| `navigation dynamic bridge` | STEP 24N point/cell observation + STEP 17 contract | observed-provenance dynamic record with exact cell AABB | relabel as synthetic/physical geometry, invent identity/time/size, validate paths or replan |
| `navigation path-validity comparison` | preserved STEP 17 path + STEP 25N before/after world | two validity records and causal effect classification | generate/alter a path, infer blocking from AFTER alone, call persistence or replanning |
| `controlled path-effect experiment` | preserved STEP 26N observed control + documented STEP 17 blocker fixture | provenance-separated VALID→VALID and VALID→BLOCKED contract evidence | invent paths/obstacles, merge observed and development evidence, call persistence or replanning |
| `path-block persistence bridge` | STEP 27N validity evidence + existing STEP 19 logical-time policy | CONTINUE, STOP/WAIT, persistence classification, decision-only REPLAN_REQUIRED | treat obstacle presence as block, alter thresholds, execute STEP 18/A*/hardware stop |
| `persistent-block replanning gate` | STEP 28N BLOCKED + persistent decision + valid provenance | existing STEP 18 result, separately preserved old/replacement paths and revision lineage | replan VALID/temporary evidence, alter planner/config/obstacle/goal, claim camera or hardware execution |
| `offline navigation chain audit` | immutable STEP22D–29N artifacts | two separate lineage records, consistency matrix and system capability boundary | rerun prior stages, duplicate large artifacts, merge observed control with development positive evidence |
| `perception` | raw frame, target text | validated label/bbox, raw model text | metric coordinate/control |
| `geometry` | raw pixel, camera/H config | undistorted pixel, world X/Y | target selection |
| `localization` | frame, marker/config | robot X/Y/yaw, timestamp, validity | motion command |
| `task` | instruction | action/target/destination/constraints | path/motor generation |
| `world` | validated module outputs | unified workspace state | infer missing physical values |
| `planning` | state, footprints/config | approach pose, feasibility, paths | direct hardware actuation |
| `control` | validated pose/goal/path | bounded vx/vy/omega, gripper command | continue on stale/invalid input |
| `evaluation` | saved raw/processed data | metrics/figures/PASS evidence | fabricate/manual metric values |

## Coordinate Systems

다음 값은 섞지 않고 이름과 단위를 보존한다.

1. `raw_image_px`: distorted camera frame의 `(u,v)` pixel.
2. `undistorted_image_px`: 명시된 new camera matrix/ROI의 `(u,v)` pixel.
3. `workspace_m`: 고정 origin과 +X/+Y를 가진 floor-plane metric 좌표.
4. `marker_frame`: ArUco marker center/orientation.
5. `robot_base`: marker-to-robot-center transform이 적용된 robot reference pose.
6. ROS2 `map`, `odom`, `base_link`: TF ownership과 변환 timestamp를 명시한다.

Homography에는 `undistorted_image_px → workspace_m`만 입력한다. raw pixel은 같은 calibration/new camera matrix 정책으로 먼저 변환한다.

`H_image_to_image`는 stored-image camera-view compatibility 진단에만 사용한다. 이는 `undistorted_image_px → workspace_m` Homography가 아니며 world coordinate 또는 planner input을 만들지 않는다. Stored replay의 filename order와 pixel tracking은 각각 실제 capture timeline 및 live temporal tracking을 의미하지 않는다.

STEP 22 connects triggered pixel observations to VLM semantics and metric point grounding only. Scene-change candidate geometry is never used directly as physical obstacle geometry. Metric grounding uses the bottom-center of a validated VLM visible-object bbox through the existing deterministic legacy geometry pipeline. A metric point is not a physical obstacle footprint. Planner injection remains prohibited until obstacle footprint geometry has an explicit source.

STEP 20 candidate boxes use half-open raw pixels. Candidate-conditioning may normalize or annotate them for semantic attention, but overlay/crop coordinates never become geometry inputs. The geometry source remains a validated full-object bbox expressed in the original full-image domain.

STEP 22C uses candidate overlay and normalized candidate metadata solely for semantic attention. Metric coordinates are derived only from a validated full-object bbox mapped back to the original full-image domain. Overlay pixels are never used as Homography input. One representative actual inference was executed; its `NO_VISIBLE_ENTITY` result left association and metric geometry unexecuted.

Method D uses a secondary crop only to improve semantic visibility. All metric coordinates continue to originate from a full-object bbox expressed in the original full-image coordinate system. Crop coordinates are never mapped through Homography. STEP 22D executes exactly one actual Qwen inference; the crop is never a geometry or planner input.

STEP 23 adds a source-gated physical obstacle geometry layer after semantic-metric grounding and before any planning representation. A semantic metric point is not itself a physical obstacle footprint. Physical object geometry requires an explicit dimensional source. The VLM bounding box is not used to infer metric width or length. Observed bottom-center is not automatically the physical footprint center. Physical geometry, uncertainty, safety margin, and planner representation are separate layers. Unknown dimensions or anchor keep the representative record non-placeable; development synthetic geometry verifies transforms without becoming observed-object evidence.

STEP 23M adds a user-measurement intake gate in front of the real geometry registry. The intake preserves raw `mm`/`cm`/`m` values, canonicalizes them to meters, and reuses the STEP 23 shape, anchor, orientation and world-placement validators. Only a complete `USER_PHYSICAL_MEASUREMENT` record may enter the registry. Missing input remains `WAITING_FOR_USER_MEASUREMENT`; incomplete dimensions, anchor or orientation never create a world footprint. Even a READY physical record remains separate from planning geometry and cannot trigger planner injection in STEP 23M.

STEP 24N creates an independent navigation branch: validated semantic-metric observation → `OBSTACLE_CANDIDATE` → `NAVIGATION_POINT_OBSTACLE` → existing STEP 15 containing-cell projection. Physical geometry and navigation approximation are separate layers, so missing STEP 23M measurements do not block this observation. The point remains a bbox-bottom-center development reference, not a physical center, and VLM bbox dimensions never produce object size. The adapter adds no radius, footprint, inflation or safety margin; clearance remains the existing navigation collision/cost layer's responsibility. Because STEP 17 currently accepts only development-synthetic axis-aligned rectangles, dynamic-world insertion requires a separately approved STEP 25N bridge.

STEP 25N minimally extends STEP 17 geometry provenance to accept `NAVIGATION_APPROXIMATION` while preserving `DEVELOPMENT_SYNTHETIC`. The bridge converts the recorded containing cell into its exact metric AABB and registers it through the existing obstacle-event and dynamic collision-world APIs. This rectangle is navigation discretization, not physical geometry. The original reference point, cell identity, observation-scoped track and unavailable timestamp remain traceable. Static occupancy and dynamic state remain distinct: the representative source cell was already occupied, but the dynamic record still exists; existing robot-footprint collision processing may affect surrounding cells without constituting object-size inference. No path validity or replanning stage is part of STEP 25N.

STEP 26N evaluates one preserved path against both STEP 25N world revisions using the existing STEP 17 validator. Path identity is proven by the same grid/world waypoint digest before and after. Observation-induced blocking is a transition claim requiring valid-before, blocked-after and defensible location provenance; an after-only block is insufficient. The representative D1 path remains valid in both worlds, so the result is `NO_PATH_EFFECT` despite the recorded occupancy changes. No replacement path, persistence decision or replanning is produced.

STEP 27N keeps the STEP 26N observed result as a read-only non-blocking control and reproduces the existing STEP 17 D2 blocker fixture as a positive software-contract case. Both evaluate an unchanged stored path with the existing validity API. Their provenance classes remain separate: observed navigation approximation versus documented development fixture. The experiment verifies both `VALID→VALID` and `VALID→BLOCKED` handling without generating a path or invoking persistence/replanning.

STEP 28N consumes STEP 27N validity as input evidence and leaves persistence classification to the existing STEP 19 state machine. `PATH_BLOCKED`, `PERSISTENT_BLOCKING`, `REPLAN_REQUIRED` and replanning execution are distinct. CASE A remains `CONTINUE_CURRENT_PATH`; CASE B uses documented development logical time to reach STOP/WAIT and then decision-only `REPLAN_REQUIRED`. `execute_replanning=False` prevents the STEP 18 handoff, so planner runs and new paths remain zero. STOP/WAIT are offline policy semantics, not hardware-stop verification; STEP 23M physical geometry remains independently `NOT_READY`.

STEP 29N is the first stage in this branch that actually invokes existing STEP 18. A strict gate rejects VALID, NO_PATH_EFFECT, temporary, missing-decision and malformed-provenance inputs. CASE A therefore stays revision 0 with no planner call. CASE B keeps its `DOCUMENTED_DEVELOPMENT_FIXTURE` provenance and passes the preserved D2 world, active path, STEP 28N current waypoint, goal and unchanged planner config to `evaluate_event_and_replan`. Existing STEP 18 alone rebuilds world/grid, runs the existing planner, validates the replacement and owns the revision transition. The actual result is accepted revision 1 after `REPLAN_SUCCESS`; old and replacement paths remain separate artifacts. This is offline software integration, not camera-derived replanning, ROS/Nav2 execution or physical robot validation.

STEP 30N audits the complete offline navigation chain as two parallel evidence tracks. TRACK A is camera/VLM-derived observed control: STEP22D semantic-metric evidence reaches STEP24N navigation approximation, STEP25N dynamic state and STEP26N path validity, then remains `VALID / NO_PATH_EFFECT / CONTINUE_CURRENT_PATH` without STEP18. TRACK B is an independent documented development positive: STEP17 D2 reaches STEP27N block evidence, STEP28N persistence decision and STEP29N existing replanning with a valid replacement. Cross-step IDs, coordinates, path digest and revisions are asserted, but the tracks never join into an observed replanning claim. STEP30N performs artifact-level verification only and introduces no perception, world, policy or planner execution.

## Process Placement

### Laptop

camera capture/undistortion, ArUco workspace localization, Qwen inference, grounding, task parser, workspace state, validator/planner, visualization, logging/evaluation.

### Jetson Orin Nano

기존 ROS2 mecanum controller, serial motor interface, encoder/IMU/EKF, velocity execution, gripper I/O, emergency stop.

기존 topic과 TF frame은 변경하지 않는다. Laptop은 검증된 명령만 기존 velocity interface로 전달한다.

## Safety Boundary

```text
fresh valid pose + valid state + feasible plan + communication healthy
                            │
                 yes ──────┴────── no
                  │                  │
             bounded command     cmd_vel = 0
```

hardware motion은 별도 사용자 실행 단계이며 offline/simulation/visualization/dry-run 결과를 먼저 요구한다.

## Configuration Ownership

- `experiment.yaml`: frozen run-level hardware/environment 선택
- `camera.yaml`: device, resolution/FPS, calibration path, undistortion policy
- `workspace.yaml`: origin/axes/boundary/zones/obstacles
- `robot.yaml`: footprint, marker→base transform, velocity limits
- `object.yaml`: type, footprint, mass, grasp interface/offset
- `gripper.yaml`: geometry, offset, limits/state interface
- `aruco.yaml`: dictionary, ID, measured marker size, validity timeout
- `vlm.yaml`: model, prompt/schema, inference limits
- `planner.yaml`: grid, margins, footprint modes, A* parameters

아직 값이 확정되지 않았으므로 config 파일은 Phase 0에서 측정 근거와 함께 생성한다.
