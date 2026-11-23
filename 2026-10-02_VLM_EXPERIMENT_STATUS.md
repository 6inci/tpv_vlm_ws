# VLM Experiment Status Report

Date: 2026-10-02

This document reconstructs the actual status of the workspace as of 2026-10-02 using the files, artifacts, and result records present in the repository. It is a mid-term research status report, not a code-change log or a new experiment run.

Important constraints applied during this review:
- No existing file was modified.
- No experiment was re-run.
- No package / CUDA / ROS2 / Conda environment changes were made.
- This report uses the preserved evidence in the workspace as the source of truth.

---

## 1. Workspace structure and what each part is doing

### 1.1. Top-level structure

The active research area is the folder [new_experiment_environment](new_experiment_environment), not the root README alone. The repository also contains a separate historical/legacy branch under [wanna_find_detailed_experimental_x,Y](wanna_find_detailed_experimental_x,Y) that contains offline planning, geometry, and earlier VLM experiments.

Major top-level directories:

- [AGENTS.md](AGENTS.md): project rules and scope constraints.
- [README.md](README.md): high-level research description and system pipeline.
- [new_experiment_environment](new_experiment_environment): current active experimental workspace.
- [wanna_find_detailed_experimental_x,Y](wanna_find_detailed_experimental_x,Y): earlier historical research branch with offline task/geometry/planning artifacts.

### 1.2. Active experimental workspace: [new_experiment_environment](new_experiment_environment)

Key folders and their roles:

- [new_experiment_environment/scripts](new_experiment_environment/scripts): execution scripts for Qwen, Grounding DINO, SAM2, YOLO, metric map generation, MCP prototype, and result assembly.
- [new_experiment_environment/results](new_experiment_environment/results): the main evidence store for experiments and artifacts.
- [new_experiment_environment/config](new_experiment_environment/config): not strongly used in this run, but the experiment package is configured around synthetic image + metric pipeline definitions.
- [new_experiment_environment/experiment_config.json](new_experiment_environment/experiment_config.json): run-level configuration for the first baseline.
- [new_experiment_environment/metric_pipeline_config.json](new_experiment_environment/metric_pipeline_config.json): configuration snapshot for the synthetic metric pipeline.
- [new_experiment_environment/EXPERIMENT_INDEX.md](new_experiment_environment/EXPERIMENT_INDEX.md): master index of run-level experiments and current status.
- [new_experiment_environment/PROJECT_STATUS.md](new_experiment_environment/PROJECT_STATUS.md): project status across phases.
- [new_experiment_environment/RUN_PLAN.md](new_experiment_environment/RUN_PLAN.md): current run plan and experiment sequencing.

### 1.3. Active results and artifacts

The strongest current evidence is under:

- [new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900](new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900)
- [new_experiment_environment/results/YOLO_WORLD_INTEGRATION_20260929T012518+0900](new_experiment_environment/results/YOLO_WORLD_INTEGRATION_20260929T012518+0900)
- [new_experiment_environment/results/SEMANTIC_REASONING_20260929T094013+0900](new_experiment_environment/results/SEMANTIC_REASONING_20260929T094013+0900)

These contain the main evidence for:
- Qwen scene/object generation
- Grounding DINO and YOLO World detection diagnostics
- SAM2 masks
- synthetic image-to-world homography
- metric map generation
- cross-detector comparison
- semantic reasoning and target grounding attempt

### 1.4. Legacy / historical workspace: [wanna_find_detailed_experimental_x,Y](wanna_find_detailed_experimental_x,Y)

This folder contains earlier offline experiments and system design documents. It records many phases of workspace state, task parsing, geometry, navigation policy, and path validation. Its most relevant files are:

- [wanna_find_detailed_experimental_x,Y/docs/ARCHITECTURE.md](wanna_find_detailed_experimental_x,Y/docs/ARCHITECTURE.md): architecture and module boundaries.
- [wanna_find_detailed_experimental_x,Y/docs/PROJECT_STATUS.md](wanna_find_detailed_experimental_x,Y/docs/PROJECT_STATUS.md): historical phase-by-phase status.
- [wanna_find_detailed_experimental_x,Y/results](wanna_find_detailed_experimental_x,Y/results): a large set of offline result folders.
- [wanna_find_detailed_experimental_x,Y/scripts](wanna_find_detailed_experimental_x,Y/scripts): earlier task/parser and grounding scripts.

The historical branch does not currently represent a validated robot-execution pipeline; it is better interpreted as a design and offline validation trail.

---

## 2. Research goal reconstructed from code and documents

The project is trying to build a system for:

- fixed third-person camera observation of a workspace
- VLM-based semantic understanding of the scene and objects
- object and region grounding from natural language
- map-level semantic understanding, not only raw object detection
- metric grounding through camera geometry
- target region or target coordinate generation from language and context
- downstream delivery to robot motion and/or task planning

The underlying objective matches this description from the root README and the experiment reports:

> fixed third-person camera + VLM → workspace semantics → geometry → target grounding → robot task execution / navigation

### 2.1. Difference from generic object detection

This is not the same as a standard object detector pipeline.

- Simple object detection asks: what objects are in the image?
- This project asks: what does the scene mean, which objects/regions are relevant, and where should the robot go for a meaningful task?

### 2.2. Difference from generic navigation

- Generic navigation: goal is usually a pose, a waypoint, or a costmap.
- This project aims to interpret natural-language task meaning and grounding into a semantic region or goal coordinate using workspace semantics.

### 2.3. Intended distinction

- Simple Goal Coordinate Input: direct metric point or waypoint
- This project: natural-language instruction → semantic interpretation → object/region grounding → metric target or region candidate

This means the real research question is not just detection accuracy; it is semantics-to-geometry grounding under a controlled workspace.

---

## 3. Reconstructed current system pipeline

The actual pipeline that exists in the current project is best approximated as:

```text
Third-Person Camera
    ↓
Input image / synthetic workspace image
    ↓
Camera geometry / synthetic homography setup
    ↓
Undistortion / image-space processing
    ↓
Object detection (Qwen, Grounding DINO, YOLO-World diagnostics)
    ↓
Segmentation (SAM2, prompt-based masks)
    ↓
Semantic understanding (Qwen structured context)
    ↓
Object / relation / semantic region generation
    ↓
Scene graph / region representation
    ↓
Natural-language instruction grounding
    ↓
Target region / target metric point generation
    ↓
Metric map / traversability / downstream planning prototype
```

### 3.1. Stage-by-stage status

| Stage | Purpose | Used technology | Input | Output | Status |
|---|---|---|---|---|---|
| Camera input | fixed observation | fixed camera image + synthetic workspace image | 1920×1080 image | image artifact | PASS (for image availability) |
| Geometry setup | metric projection basis | synthetic homography from visible grid | 12 control points | homography matrix | PARTIAL |
| Undistortion | image rectification | synthetic calibration procedure | raw image pixel data | undistorted pixel data | PARTIAL |
| Qwen perception | semantic object and region understanding | Qwen/Qwen3-VL-4B-Instruct | image | object list, relations, regions | PARTIAL |
| Grounding DINO | detector diagnostic and object localization | IDEA-Research/grounding-dino-tiny | image | bbox detections | PASS |
| YOLO-World | detector diagnostic and cross-detector comparison | Ultralytics YOLO-World | image + class vocabulary | bbox detections | PARTIAL / BLOCKED in environment but run completed |
| SAM2 | mask generation from boxes | facebook/sam2.1-hiera-tiny | DINO/YOLO boxes | masks | PASS (diagnostic) |
| Semantic reasoning | task grounding + scene meaning | structured context + Qwen | image + task prompts | region/target predictions | PARTIAL |
| Scene graph / metadata | object-relation integration | pipeline JSON | object + region + relation outputs | scene graph-like records | PARTIAL |
| Metric map | occupancy/traversability map | synthetic homography + masks | object/bbox/mask intersections | map artifacts | PARTIAL |
| Motion interface | robot command path | not yet active | no real robot data | none | NOT STARTED |

### 3.2. Current state labels

The project uses several state labels across the experimental artifacts. The most important to preserve are:

- PASS: validated pipeline evidence exists for the given artifact
- PARTIAL: evidence exists but is limited, synthetic, or not ground-truth validated
- BLOCKED: implementation or environment prevents completion
- NOT STARTED: not yet implemented
- UNKNOWN: no recoverable evidence

The current active run is best described as: PARTIAL with strong pipeline evidence, but not a mature end-to-end research claim.

---

## 4. Model-by-model role and evidence

### 4.1. Qwen3-VL-4B-Instruct

This is the main semantic model used in the active run.

Relevant evidence:
- [new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/RUN_REPORT.md](new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/RUN_REPORT.md)
- [new_experiment_environment/results/SEMANTIC_REASONING_20260929T094013+0900/SEMANTIC_REASONING_RUN_REPORT.md](new_experiment_environment/results/SEMANTIC_REASONING_20260929T094013+0900/SEMANTIC_REASONING_RUN_REPORT.md)
- [new_experiment_environment/results/EXP001_run01/FINAL_REPORT.md](new_experiment_environment/results/EXP001_run01/FINAL_REPORT.md)

Observed role:
- model used under the `qwen_vl` Conda environment
- used for object recognition and relation understanding
- used for semantic region generation and structured scene context
- used as the language-grounding module, but the source-specific Qwen path was explicitly blocked after three attempts in the semantic-reasoning run
- Qwen output is treated as model-derived semantics, not as ground truth

Actual evidence:
- METRIC_VLM_PIPELINE run: Qwen attempts recovered 31 complete objects via deterministically reconstructed JSON
- The same run also produced region and relation JSON artifacts
- Semantic reasoning run: 15 structured tasks were completed, with 13 answered and 2 abstained
- Source-specific Qwen output remained blocked, and no direct source-specific accuracy claim was made

Current assessment:
- Strength: semantic understanding and scene description were produced in a reproducible pipeline
- Limitation: source-specific Qwen semantic generation is not yet stable enough to be treated as a validated metric-grounding source
- Downstream use: object IDs, region candidates, semantic target generation, scene graph-like structured context

### 4.2. YOLO-World

Evidence:
- [new_experiment_environment/results/YOLO_WORLD_INTEGRATION_20260929T012518+0900/RUN_REPORT.md](new_experiment_environment/results/YOLO_WORLD_INTEGRATION_20260929T012518+0900/RUN_REPORT.md)
- [new_experiment_environment/scripts/run_yolo_world_integration.py](new_experiment_environment/scripts/run_yolo_world_integration.py)

Actual model details:
- model: `yolov8s-world.pt`
- library: Ultralytics 8.4.164
- environment: `/home/binci/miniconda3/envs/yolo_world`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- classes used: truck, pallet, cardboard box, storage rack, workbench, cart, drum, mobile robot, robot arm, conveyor belt, safety fence, traffic cone

Actual detection counts:
- total detections: 39
- detected classes: cardboard box 25, cart 2, storage rack 5, traffic cone 3, truck 3, workbench 1
- conveyor: false
- manipulator: false

Important diagnostics:
- Qwen: 31 detections
- Grounding DINO: 58 detections
- YOLO-World: 39 detections
- high-overlap disagreement examples involving wrong-class localization were recorded:
  - q_030 (Qwen manipulator) vs YOLO cart with IoU 0.9311
  - q_029 (Qwen mobile robot) vs YOLO truck with IoU 0.9231

Current assessment:
- YOLO-World is technically runnable in the separate environment and produced real detections.
- The project does not claim detector accuracy because no ground-truth annotation exists.
- It is useful as a diagnostic cross-detector comparison layer, not as a final scientific detector claim.

### 4.3. SAM2

Evidence:
- [new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/08_sam2/segmentation.json](new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/08_sam2/segmentation.json)
- [new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/08_sam2/RESULT.md](new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/08_sam2/RESULT.md)

Actual model details:
- `facebook/sam2.1-hiera-tiny`
- run used after Grounding DINO box prompts; not a free-standing scene understanding model in this flow

Actual role:
- initial attempt failed because the model processor was missing locally
- later run succeeded via installed Transformers path
- produced 39 masks in the synthetic metric pipeline and 38 masks in the YOLO integration path

Current assessment:
- SAM2 is a segmentation layer, not a final semantic decision layer
- it reduces bbox over-coverage and contributes to metric occupancy and traversability overlays
- it is used as a diagnostic segmentation tool rather than as a proof of physical occupancy

### 4.4. Grounding DINO

Evidence:
- [new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/04_grounding_dino/RESULT.md](new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/04_grounding_dino/RESULT.md)
- [new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/06_detector_comparison/detector_comparison.json](new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/06_detector_comparison/detector_comparison.json)

Actual usage:
- `IDEA-Research/grounding-dino-tiny`
- executed in the same environment as Qwen and used as a detector diagnostic
- produced 58 detections in the reference metric pipeline

Current assessment:
- It is a real, working detector system in the current research path.
- It is not positioned as a final scientific claim; it is a benchmark and cross-detector comparison component.
- It provides bounding boxes that feed SAM2 mask generation.

### 4.5. Other models and helpers

The project also includes a suite of helper and integration structures:
- local JSON-based scene graph representations
- MCP prototype server for scene-query style interfaces
- read-only synthetic geometry and metric map generation scripts
- manual or semi-heuristic region labeling in the early EXP001 and EXP002 runs

These are useful as pipeline scaffolding, but not as end-to-end proof of physical robot competence.

---

## 5. Geometry and coordinate system status

This is a critical part of the project and is the main basis for metric target grounding.

### 5.1. Actual input image

The current synthetic metric pipeline uses:

- [new_experiment_environment/warehouse_robotics_1920x1080.png](new_experiment_environment/warehouse_robotics_1920x1080.png)
- resolution: 1920×1080
- SHA-256: `5e1cdda9fec9a3840396dd60cb0743dd2d519fd0884587c1c0156d89a565e331`

The earlier approximate-image baseline uses:
- [new_experiment_environment/approximate_environment.png](new_experiment_environment/approximate_environment.png)
- resolution: 1448×1086
- SHA-256: `02a4e79589db3c15529b372768ef349b4ef2e4ebe2efb2454f605be43eb95a95`

### 5.2. Workspace geometry in the current synthetic metric pipeline

From [new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/geometry/control_points.json](new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/geometry/control_points.json) and the run report:

- workspace size: 3.15 m × 2.10 m
- tile size: 1.05 m square tiles
- layout: 3×2
- coordinate frame: `synthetic_workspace_bottom_left_xy_m`
- origin: bottom-left `(0,0)`
- world plane: X right, Y up
- map resolution: 0.01 m per pixel
- 12 control points total: 8 calibration + 4 holdout validation

### 5.3. Homography

The model uses a synthetic software homography fit, not a real camera calibration.

Evidence:
- [new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/geometry/homography.json](new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/geometry/homography.json)
- [new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/RUN_REPORT.md](new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/RUN_REPORT.md)

Numerical values recorded:
- calibration RMSE: 0.070252 m
- holdout RMSE: 0.091300 m
- max error: 0.114344 m

The run explicitly states: this is a synthetic software/pipeline feasibility homography, not real-camera calibration.

### 5.4. Pixel-to-world transformation path reconstructed from evidence

The intended geometry chain is:

```text
Raw camera image
    ↓
Calibration / visible grid points
    ↓
Undistortion procedure
    ↓
Raw pixel (u, v)
    ↓
Undistorted pixel
    ↓
Homography projection
    ↓
Workspace metric coordinates (X, Y) in metres
```

The current project is using a synthetic, manually audited calibration and does not claim physical camera calibration quality.

### 5.5. Important limitation

The geometry is strong as a proof-of-pipeline concept, but weak as a real-world robot source of truth. The run report says the geometry remains synthetic and not Nav2-ready.

---

## 6. Current experiment environment summary

### 6.1. Input image and workspace

The main recent experiments use the synthetic warehouse scene image:

- [new_experiment_environment/warehouse_robotics_1920x1080.png](new_experiment_environment/warehouse_robotics_1920x1080.png)

This image is interpreted as a warehouse-like work area with objects such as:
- truck
- pallet
- cardboard box
- storage rack
- workbench
- cart
- drum
- mobile robot
- robot arm/manipulator
- conveyor belt
- safety fence
- traffic cone

The vocabulary used in the YOLO-World script is explicit and matches the expected knowledge set in the detector pipeline.

### 6.2. Active environment details

The key environments named in the artifacts are:

- Qwen model environment: `/home/binci/miniconda3/envs/qwen_vl`
- YOLO environment: `/home/binci/miniconda3/envs/yolo_world`

This means the workspace is intentionally not using a single environment for every detector. The split is explicit and important for reproduction.

---

## 7. Experiment history reconstructed from the files

### 7.1. EXP001_run01 (early baseline)

Path: [new_experiment_environment/results/EXP001_run01](new_experiment_environment/results/EXP001_run01)

What it did:
- first Qwen-based object recognition and image-plane map baseline
- input: approximate_environment.png
- output: 16 recovered objects and 5 relations
- semantic/traversability products were preliminary and non-metric

Status from the report:
- object recognition: PARTIAL
- relation inference: PARTIAL
- scene context: PASS
- semantic region: PARTIAL
- traversability: PARTIAL
- semantic map: PASS

This is a valid early provenance artifact, but not the main current scientific claim.

### 7.2. EXP002_semantic_region_eval

Path: [new_experiment_environment/results/EXP002_semantic_region_eval](new_experiment_environment/results/EXP002_semantic_region_eval)

What it did:
- prepared a human-review validation packet for semantic regions
- defined five candidate reference regions: Storage Area, Work Station, Loading Area, No-Go Zone, Material Area
- recorded explicit ambiguity conditions: overlap policy, region naming, No-Go vs Robotics Operation, Material vs Mobile Robot Operation

Status from the packet:
- review packet generation: PASS
- user validation form was created and left blank
- the reference material remained a candidate, not ground truth

This is a useful semantic evaluation structure, but it remains user-reviewed and not a final metric authoritative region set.

### 7.3. METRIC_VLM_PIPELINE_20260929T005848+0900

Path: [new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900](new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900)

This is the current strongest active baseline on the newer 1920×1080 image.

Key evidence:
- Qwen reconstructed 31 complete objects
- Grounding DINO produced 58 detections
- SAM2 produced 39 masks
- synthetic homography achieved 0.0913 m holdout RMSE
- metric map was generated in a synthetic metric coordinate system

Stage status from the run report:
- environment inventory: PASS
- geometry: PARTIAL
- Qwen: PARTIAL
- Grounding DINO: PASS
- YOLO-World: BLOCKED
- SAM2: PASS
- metric map: PARTIAL
- integration: PARTIAL
- MCP: PARTIAL
- end-to-end: PARTIAL

This run is the clearest current reference, but it is not yet a robot-ready or physically validated system.

### 7.4. YOLO_WORLD_INTEGRATION_20260929T012518+0900

Path: [new_experiment_environment/results/YOLO_WORLD_INTEGRATION_20260929T012518+0900](new_experiment_environment/results/YOLO_WORLD_INTEGRATION_20260929T012518+0900)

This run adds YOLO-World and cross-detector comparison.

Key numbers:
- 39 detections
- 38 YOLO→SAM2 masks
- no conveyor/maneipulator detection in the YOLO run
- Qwen and Grounding DINO both produced stronger or different class coverage depending on the query vocabulary

Status:
- technical execution with detector inference: PASS
- class coverage and detector comparison: PARTIAL
- no ground-truth accuracy claim made

### 7.5. SEMANTIC_REASONING_20260929T094013+0900

Path: [new_experiment_environment/results/SEMANTIC_REASONING_20260929T094013+0900](new_experiment_environment/results/SEMANTIC_REASONING_20260929T094013+0900)

This is the latest semantic-reasoning result and the current detector diagnostic run.

Key facts:
- 15 tasks were evaluated
- 13 answered, 2 abstained
- source-specific Qwen comparison was BLOCKED after three attempts
- manipulator, conveyor, mobile robot, and pallet behaviors were vocabulary-dependent or ambiguous
- value of structured-context output is strong as an evidence trail, not as ground truth

This run is best interpreted as an active human-review and diagnostic stage.

---

## 8. Detector diagnostic status

The detector diagnostic focus is a key issue because the project is trying to separate several failure types:

1. vocabulary problem
2. synonym problem
3. detector limitation
4. threshold problem
5. object appearance problem
6. localization/context bias
7. VLM semantic matching issue

### 8.1. Current diagnosis

From the semantic reasoning and detector comparison artifacts:

- manipulator: `VOCABULARY_DEPENDENT`
- conveyor: `AMBIGUOUS`
- mobile_robot: `VOCABULARY_DEPENDENT`
- pallet: `VOCABULARY_DEPENDENT`

This means the current bottleneck is not simply straight detection failure; it is that category naming and semantics strongly affect whether the detector sees the object.

### 8.2. Validated disagreement examples

The files record actual disagreement examples:

- q_030 (Qwen manipulator) vs YOLO cart at IoU 0.9311
- q_029 (Qwen mobile robot) vs YOLO truck at IoU 0.9231

These are useful because they prove that the detector may find the right location but assign the wrong class label. This is a meaningful insight for the research, since the project’s ultimate target is semantics + function, not only bounding boxes.

---

## 9. Current detection and segmentation status matrix

| Model | Target | Result | Evidence status |
|---|---|---|---|
| Qwen3-VL | objects / relations / semantic regions | produced 31 object records in current metric pipeline; 13 answered / 2 abstained in structured reasoning | PARTIAL |
| Grounding DINO | object boxes | 58 detections in the metric run | PASS |
| YOLO-World | object boxes | 39 detections; specific classes mostly matched to box/cart/rack/cone/truck/workbench | PARTIAL |
| SAM2 | masks from boxes | 39 masks (metric run), 38 masks (YOLO integration) | PASS (diagnostic only) |
| Qwen source-specific ground truth comparison | semantic inference | blocked after three attempts | BLOCKED |
| End-to-end robot task execution | full motion pipeline | not started | NOT STARTED |

---

## 10. Scene graph and semantic reasoning state

The project contains scene-graph-like and relation-based outputs, but the files do not support a claim that a final, validated scene graph is already a downstream planner input.

### 10.1. What exists

- object IDs and relations from Qwen output
- structured-context output with object support and relation support
- semantic region proposals in image space
- target predictions in pixel space and projected synthetic metric targets

### 10.2. What is not yet proven

- a validated scene graph feeding a deterministic planner
- a firm relation-to-goal mapping backed by human-verified ground truth
- a final semantic region policy that is metric, mutually exclusive, and physically meaningful

This is why the scene graph remains a useful intermediate artifact, not a final integrated world model.

---

## 11. Semantic reasoning state

The research question is essentially: can an agent combine object identity, spatial relations, and functional context to choose a meaningful place to move?

### 11.1. What is currently established

- The structured-context flow executes on the image and returns task-level semantic region predictions.
- The project can produce region candidates and synthetic metric target points from the image geometry.
- The target generation process still relies on synthetic-only homography and is not physical ground truth.

### 11.2. What remains open

- whether the semantic region is correct
- whether the target point is within the real robot workspace
- whether relations are stable under vocabulary or prompt variation
- whether the region semantics are valid beyond a single synthetic or approximate image

This is why the project is currently in a semantic reasoning / diagnostic phase rather than a robot execution phase.

---

## 12. Traversability and metric map status

The current metric map is generated from synthetic homography and then used for occupancy or traversability overlays.

Data from [new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/03_metric_map/metric_map.json](new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900/03_metric_map/metric_map.json):

- workspace size: 3.15 × 2.10 m
- resolution: 0.01 m/px
- raster size: 315 × 210 px
- bbox occupancy area: 1.8602 m²
- SAM2 occupancy area: 1.0322 m²
- bbox traversable area: 3.9919 m²
- SAM2 traversable area: 4.4968 m²

Important: the run report says this is not Nav2-ready and not a real-camera claim.

---

## 13. Nav2 and robot motion interface status

The project has not reached a validated Nav2 or robot-execution core.

The evidence indicates:
- VLM object/semantic outputs exist
- synthetic metric geometry exists
- a scene-query or MCP prototype exists
- but no validated path is handed to a real Nav2 stack or to robot motion

The historical design docs explain the intended modular contracts, but current active evidence is still in perception + geometry + semantic target generation stages.

---

## 14. Core achievements confirmed by the artifacts

The strongest current positive statements that are supported by the evidence are:

- Qwen inference on the new 1920×1080 scene is reproducible at least in the source-artifact sense
- Grounding DINO produced valid detector outputs in the current metric run
- SAM2 masks can be generated from detector boxes in this pipeline
- a synthetic 12-point homography and validation residuals were produced
- a metric occupancy/traversability map was generated from synthetic geometry
- a detector comparison and semantic reasoning diagnostic is operating on the same image
- an initial scene-graph-like and target-generation structure exists

These are valuable research prototypes, but they are not yet end-to-end robot-validated evidence.

---

## 15. Current problems classified

### A. Must be solved before claiming a strong research result

1. Semantic region ↔ metric coordinate connectivity is still weak and synthetic-only.
2. Natural-language grounding remains partially blocked or user-reviewed.
3. Semantic relations and target generation must be validated against human-reviewed ground truth.
4. Detector vocabulary sensitivity remains a serious issue for manipulator, conveyor, mobile robot, and pallet classes.
5. Real camera calibration and real world coordinate validation are still missing.

### B. Good to improve next

1. Controlled vocabulary trials across classes
2. More robust synonym handling for detector prompts
3. Better region naming and semantic discretization
4. Cross-detector agreement and filtering strategy
5. Human validation of semantic regions and target generation

### C. Not a current priority

1. Full robot hardware integration
2. ROS2 Nav2 tuning
3. real-time deployment optimization
4. large-scale end-to-end motion benchmark

---

## 16. Research stage evaluation

### 16.1. Current implementation status

The project is currently a perception-and-geometry prototype with semantic reasoning diagnostics. It is not a mechanically validated robot system.

### 16.2. Current validation status

What is validated:
- artifact generation and pipeline execution order
- synthetic geometric residual values
- Qwen/Grounding DINO/SAM2 output generation on the provided image
- detector comparison structure and vocabulary sensitivity analysis

What is not validated:
- real-world camera calibration
- object-class accuracy against ground truth
- semantic region correctness
- target-point success rate
- trajectory or robot movement validity
- end-to-end planning robustness

### 16.3. Bottleneck

The biggest bottleneck is not model inference itself; it is the handoff from image semantics to validated, metric, human-reviewed goal generation.

The pipeline produces many candidate artifacts, but the validated step that turns semantics into a robot-safe metric goal still remains incomplete.

### 16.4. Current phase

The current phase is best described as:

```text
Perception / detector studies
→ synthetic geometry + metric projection
→ semantic reasoning diagnostics
→ semantic region and target grounding prototype
→ real-world metric validation still pending
```

Not yet at: full integration, physical validation, or publication-grade end-to-end proof.

---

## 17. Recommended next experiments

The best next work is not to add a large number of disconnected features. It should follow the existing evidence chain.

### Priority 1: human-reviewed semantic grounding

Need to answer:
- Which semantic regions are valid?
- Which target assignments are correct?
- Which classes are truly ambiguous?

Required evidence:
- user-reviewed semantic regions
- manual QA for target grounding
- review of object labels and detector disagreements

### Priority 2: real geometry validation

Need to answer:
- does the homography actually correspond to the real fixed camera?
- does the metric map match the physical workspace?
- are the object points in the correct world coordinates?

Required evidence:
- measured real-world calibration points
- holdout validation on real camera data
- explicit statement of uncertainty and projected RMSE

### Priority 3: semantic target generation and validation

Need to answer:
- can a natural-language instruction be grounded to a region/goal with human-reviewed correctness?
- is the resulting target inside the valid workspace?
- does semantic region status map into a safe robot operation point?

---

## 18. Reproducible current-state overview

### Main workspace paths

- root: `/home/binci/vlm_ws`
- active experiment folder: `/home/binci/vlm_ws/new_experiment_environment`
- historical branch: `/home/binci/vlm_ws/wanna_find_detailed_experimental_x,Y`

### Model references

- Qwen: `Qwen/Qwen3-VL-4B-Instruct`
- Grounding DINO: `IDEA-Research/grounding-dino-tiny`
- YOLO-World: `yolov8s-world.pt`
- SAM2: `facebook/sam2.1-hiera-tiny`

### Key scripts

- [new_experiment_environment/scripts/run_metric_pipeline.py](new_experiment_environment/scripts/run_metric_pipeline.py)
- [new_experiment_environment/scripts/run_yolo_world_integration.py](new_experiment_environment/scripts/run_yolo_world_integration.py)
- [new_experiment_environment/scripts/run_qwen_semantics.py](new_experiment_environment/scripts/run_qwen_semantics.py)
- [new_experiment_environment/scripts/run_vlm.py](new_experiment_environment/scripts/run_vlm.py)
- [new_experiment_environment/scripts/run_semantic_reasoning.py](new_experiment_environment/scripts/run_semantic_reasoning.py)

### Key result directories

- [new_experiment_environment/results/EXP001_run01](new_experiment_environment/results/EXP001_run01)
- [new_experiment_environment/results/EXP002_semantic_region_eval](new_experiment_environment/results/EXP002_semantic_region_eval)
- [new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900](new_experiment_environment/results/METRIC_VLM_PIPELINE_20260929T005848+0900)
- [new_experiment_environment/results/YOLO_WORLD_INTEGRATION_20260929T012518+0900](new_experiment_environment/results/YOLO_WORLD_INTEGRATION_20260929T012518+0900)
- [new_experiment_environment/results/SEMANTIC_REASONING_20260929T094013+0900](new_experiment_environment/results/SEMANTIC_REASONING_20260929T094013+0900)

### Last recorded active status

The most recent and most relevant run is the semantic reasoning run in:

- [new_experiment_environment/results/SEMANTIC_REASONING_20260929T094013+0900](new_experiment_environment/results/SEMANTIC_REASONING_20260929T094013+0900)

The state recorded there is: PARTIAL.

---

## 19. Explicit Unknown / uncertain records

The project is careful to separate facts from assumptions. Several points are still uncertain and should not be overclaimed:

- whether the detected regions are real physical semantic zones or just image-space approximations
- whether the vocabulary/class naming corresponds to a canonical robot task ontology
- whether the synthetic homography maps to a real fixed camera with acceptable accuracy
- whether semantic relations are stable under prompt changes and scene variation
- whether natural-language instruction grounding can be turned into a safe robot motion goal without human review

The rule applied here is: if it is not present in the actual artifact and run report, it is not treated as a confirmed result.

---

## 20. Summary

# 2026-10-02 기준 연구 상태 Summary

## 연구 목표

현재 작업은 고정 카메라에서 작업환경을 이해하고, 자연어 지시를 의미 기반으로 해석하며, 물체와 장소를 metric 좌표로 연결해 이동로봇의 목표 지점 또는 작업 영역을 생성하는 VLM 기반 작업환경 인식 연구를 목표로 한다. 현재 단계는 신호와 산출물 중심의 perception/geometry/semantic-grounding prototype에 머물러 있다.

## 현재 구현 완료

- Qwen object/scene semantic inference artifact generation
- Grounding DINO detector inference
- YOLO-World detector diagnostic execution
- SAM2 mask generation from boxes
- synthetic homography and metric-map generation
- detector comparison, structured context, and semantic reasoning output generation
- pass/fail documentation for artifact-level provenance and validation

## 현재 검증 완료

- synthetic image + detector + mask pipeline execution on the current image
- image-to-world homography residual logs and metric map generation in synthetic space
- artifact existence and provenance recorded in the run reports
- cross-detector vocabulary sensitivity and disagreement analysis

## 부분 구현 / 부분 검증

- semantic region generation
- natural-language grounding to target region/point
- object-to-metric mapping under synthetic geometry
- scene graph-like integration of object and relation outputs
- end-to-end movement or Nav2 interface

## 현재 BLOCKED

- real-world camera calibration and physical geometry validation
- fully validated semantic region ground truth
- source-specific Qwen semantic comparison after three attempts
- robot execution / motion control integration
- end-to-end planner validation against real workspace constraints

## 가장 중요한 미해결 문제

1. semantic region과 metric coordinate의 연결이 아직 synthetic-only이며 검증이 부족하다
2. natural-language grounding이 실제 목표 좌표로 수렴하는 절차가 아직 불완전하다
3. detector class ambiguity and vocabulary sensitivity remain critical for key object classes such as manipulator, conveyor, pallet, and mobile robot

## 현재 연구 단계

```text
Perception + detector study
→ synthetic geometry / metric projection
→ semantic reasoning diagnostics
→ semantic region and target-grounding prototype
→ real geometry validation and end-to-end integration remain pending
```

## 마지막으로 진행 중이던 작업

최종으로 확인된 가장 최신 작업은 semantic reasoning run:

- [new_experiment_environment/results/SEMANTIC_REASONING_20260929T094013+0900](new_experiment_environment/results/SEMANTIC_REASONING_20260929T094013+0900)

이 run은 structured context와 detector diagnostic을 수행했으며, 대체로 PARTIAL 상태로 종결되었다. 가장 최근의 핵심 사실은 source-specific Qwen comparison이 차단되었고, semantic grounding의 이슈가 vocabulary, ambiguity, and human review 문제로 분리되었다는 점이다.

## 다음 작업 추천

### Priority 1

- human-reviewed semantic region validation
- human-reviewed target grounding and region correctness
- detector vocabulary normalization for key classes

### Priority 2

- real image calibration and real-world metric validation
- homography validation against measured control points
- real-camera scene rerun with the same pipeline

### Priority 3

- natural-language grounding success evaluation
- target-to-task adaptation
- end-to-end integration after real geometry is validated

---

This status document is based on actual workspace files and artifact records. Where there was a conflict between plans and results, the preserved artifact and run report were treated as authoritative.
