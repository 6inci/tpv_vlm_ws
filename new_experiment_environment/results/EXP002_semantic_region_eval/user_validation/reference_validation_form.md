# EXP002 Semantic Reference User Validation Form

This form is for human review of the existing `REFERENCE_GT_CANDIDATE`. It does not create or approve Ground Truth. The authoritative annotation source remains `/home/binci/vlm_ws/new_experiment_environment/results/EXP002_semantic_region_eval/reference/reference_annotation_candidate.json`.

Reference evidence is shown first. Qwen/Step 03 information is isolated in **Model comparison context (review aid only)** and must not determine the user's answer.

## Global semantic-layer decision
Functional region != navigation restriction. The same physical space may be a Robotics Operation Area while also being No-Go in a navigation layer.

Choose one only if a project-wide policy is intended:
- [ ] Semantic regions must be mutually exclusive
- [ ] Overlapping semantic layers are allowed
- [ ] UNSURE

Global comment: ________________________________________________

## region_storage — Storage Area

### Reference evidence — candidate annotation
- Current semantic label: `Storage Area`
- Confidence: `0.88`
- Approximate polygon (normalized 0..1000): `[[268,25],[590,25],[590,275],[268,275]]`
- Supporting objects:
  - `obj_001` — left storage rack with boxes and bins
  - `obj_002` — right storage rack with boxes and bins
- Supporting relations:
  - `rel_001` — obj_001 (left storage rack with boxes and bins) adjacent_to obj_002 (right storage rack with boxes and bins)
- Current interpretation: Two adjacent loaded racks, repeated boxes/bins, and the surrounding shelf-access space jointly indicate storage use.
- Important uncertainty: Boundary is an approximate image-coordinate review polygon, not a metric or user-validated boundary. Reference boundary is preliminary and prediction has no polygon.
- Review image: `regions/region_storage.png`

### Model comparison context — review aid only
- Qwen prediction: `storage_area` — Inventory storage zone
- Step 03 match status: `MATCH`
- Diagnostic note: Both describe organized inventory storage centered on loaded shelving.

### Critical review question
이 랙·박스·빈 주변을 실제 실험에서 보관 구역으로 정의하는 것이 맞습니까?

### User decision — leave all unchecked until reviewed
- [ ] KEEP
- [ ] RENAME
- [ ] BOUNDARY_ADJUST
- [ ] RENAME_AND_BOUNDARY_ADJUST
- [ ] SPLIT
- [ ] MERGE
- [ ] REMOVE
- [ ] UNSURE

User label: ________________________________________________

User comment:

________________________________________________________________


## region_work — Work Station

### Reference evidence — candidate annotation
- Current semantic label: `Work Station`
- Confidence: `0.84`
- Approximate polygon (normalized 0..1000): `[[590,60],[760,60],[760,300],[590,300]]`
- Supporting objects:
  - `obj_004` — tool workbench
  - `obj_019` — black workbench chair
- Supporting relations:
  - `rel_002` — obj_019 (black workbench chair) in_front_of obj_004 (tool workbench)
  - `rel_003` — obj_004 (tool workbench) adjacent_to obj_002 (right storage rack with boxes and bins)
- Current interpretation: A tool/laptop workbench, its chair, adjacency to storage, and usable operator space jointly indicate a work station.
- Important uncertainty: Boundary is an approximate image-coordinate review polygon, not a metric or user-validated boundary. The image alone cannot confirm whether maintenance is the intended primary function.
- Review image: `regions/region_work.png`

### Model comparison context — review aid only
- Qwen prediction: `maintenance_station` — Tool and equipment maintenance area
- Step 03 match status: `PARTIAL_MATCH`
- Diagnostic note: The same workbench/chair area is identified, but the prediction specializes the broader Work Station as maintenance/configuration.

### Critical review question
이 작업대·도구·의자 영역은 일반 작업 구역입니까, 유지보수/정비 구역입니까, 또는 다른 기능입니까?

### User decision — leave all unchecked until reviewed
- [ ] KEEP
- [ ] RENAME
- [ ] BOUNDARY_ADJUST
- [ ] RENAME_AND_BOUNDARY_ADJUST
- [ ] SPLIT
- [ ] MERGE
- [ ] REMOVE
- [ ] UNSURE

User label: ________________________________________________

User comment:

________________________________________________________________


## region_loading — Loading Area

### Reference evidence — candidate annotation
- Current semantic label: `Loading Area`
- Confidence: `0.87`
- Approximate polygon (normalized 0..1000): `[[745,85],[998,85],[998,510],[745,510]]`
- Supporting objects:
  - `obj_003` — white delivery truck
  - `obj_005` — left traffic cone
  - `obj_006` — upper-right traffic cone
  - `obj_007` — lower-right traffic cone
  - `obj_016` — right pallet with cardboard boxes
- Supporting relations:
  - `rel_004` — obj_016 (right pallet with cardboard boxes) near obj_003 (white delivery truck)
  - `rel_005` — obj_005 (left traffic cone) near obj_003 (white delivery truck)
  - `rel_006` — obj_006 (upper-right traffic cone) near obj_003 (white delivery truck)
  - `rel_007` — obj_007 (lower-right traffic cone) near obj_016 (right pallet with cardboard boxes)
- Current interpretation: Truck, nearby boxed pallet, three boundary cones, and open handling space form a coherent loading bay.
- Important uncertainty: Boundary is an approximate image-coordinate review polygon, not a metric or user-validated boundary. Without polygons, the extent of the predicted merged loading/transport region is unknown.
- Review image: `regions/region_loading.png`

### Model comparison context — review aid only
- Qwen prediction: `transport_and_loading_zone` — Vehicle and pallet loading area
- Step 03 match status: `PARTIAL_MATCH`
- Diagnostic note: Truck/pallet loading function agrees, but the prediction merges handcart/barrel material evidence and omits cone-based bay evidence.

### Critical review question
트럭·박스 팔레트·콘과 주변 취급 공간을 하나의 하역 구역으로 정의하는 것이 맞습니까?

### User decision — leave all unchecked until reviewed
- [ ] KEEP
- [ ] RENAME
- [ ] BOUNDARY_ADJUST
- [ ] RENAME_AND_BOUNDARY_ADJUST
- [ ] SPLIT
- [ ] MERGE
- [ ] REMOVE
- [ ] UNSURE

User label: ________________________________________________

User comment:

________________________________________________________________


## region_robot_work — No-Go Zone

### Reference evidence — candidate annotation
- Current semantic label: `No-Go Zone`
- Confidence: `0.81`
- Approximate polygon (normalized 0..1000): `[[10,140],[420,185],[420,480],[250,480],[170,840],[20,840]]`
- Supporting objects:
  - `obj_008` — upper-left robotic manipulator
  - `obj_009` — upper-center robotic manipulator
  - `obj_010` — lower-left robotic manipulator
  - `obj_017` — red safety fence
  - `obj_018` — mobile conveyor platform
- Supporting relations:
  - `rel_008` — obj_008 (upper-left robotic manipulator) grouped_with obj_009 (upper-center robotic manipulator)
  - `rel_009` — obj_018 (mobile conveyor platform) adjacent_to obj_008 (upper-left robotic manipulator)
  - `rel_010` — obj_008 (upper-left robotic manipulator) adjacent_to obj_017 (red safety fence)
- Current interpretation: Multiple manipulators, adjacent conveyor equipment, and a safety fence define a preliminary restricted robot-work area.
- Important uncertainty: Boundary is an approximate image-coordinate review polygon, not a metric or user-validated boundary. Prediction emphasizes operation while reference emphasizes restriction; both concern the same functional cluster.
- Review image: `regions/region_robot_work.png`

### Model comparison context — review aid only
- Qwen prediction: `robotic_workstation` — Robotics operation zone
- Step 03 match status: `MATCH`
- Diagnostic note: Both describe the manipulator/conveyor/fence cluster as a functional robot operation area.

### Critical review question
로봇팔·컨베이어·펜스 주변은 작업 기능 구역입니까, 이동로봇 접근 금지 구역입니까, 아니면 겹치는 두 layer입니까?

### User decision — leave all unchecked until reviewed
- [ ] KEEP
- [ ] RENAME
- [ ] BOUNDARY_ADJUST
- [ ] RENAME_AND_BOUNDARY_ADJUST
- [ ] SPLIT
- [ ] MERGE
- [ ] REMOVE
- [ ] UNSURE

User label: ________________________________________________

User comment:

________________________________________________________________

### Required ambiguity decision — No-Go / Robotics Operation
The same physical area may have different functional and navigation meanings. Choose none until reviewed.
- [ ] A. No-Go Zone
- [ ] B. Robotics Operation Area
- [ ] C. Robotics Operation Area and simultaneously No-Go in the navigation layer
- [ ] D. Split into multiple regions/layers

Ambiguity comment: ____________________________________________


## region_material — Material Area

### Reference evidence — candidate annotation
- Current semantic label: `Material Area`
- Confidence: `0.87`
- Approximate polygon (normalized 0..1000): `[[742,535],[998,535],[998,965],[740,965]]`
- Supporting objects:
  - `obj_012` — blue handcart with box
  - `obj_014` — pallet with gray containers
  - `obj_015` — pallet with blue and gray drums
- Supporting relations:
  - `rel_011` — obj_012 (blue handcart with box) near obj_014 (pallet with gray containers)
  - `rel_012` — obj_014 (pallet with gray containers) adjacent_to obj_015 (pallet with blue and gray drums)
- Current interpretation: Loaded cart plus adjacent container/drum pallets and surrounding handling space jointly indicate material staging.
- Important uncertainty: Boundary is an approximate image-coordinate review polygon, not a metric or user-validated boundary. No polygon is available to determine whether the prediction covers the right-side material area or a broader central robot area.
- Review image: `regions/region_material.png`

### Model comparison context — review aid only
- Qwen prediction: `mobile_robot_zone` — Mobile robot operation area
- Step 03 match status: `PARTIAL_MATCH`
- Diagnostic note: Pallet/drum transport context overlaps, but the prediction centers the mobile robot rather than material staging and omits the cart.

### Critical review question
카트·컨테이너 팔레트·드럼 주변은 자재 적치 구역입니까, 이동로봇 작업 구역입니까, 또는 겹치는 별도 layer입니까?

### User decision — leave all unchecked until reviewed
- [ ] KEEP
- [ ] RENAME
- [ ] BOUNDARY_ADJUST
- [ ] RENAME_AND_BOUNDARY_ADJUST
- [ ] SPLIT
- [ ] MERGE
- [ ] REMOVE
- [ ] UNSURE

User label: ________________________________________________

User comment:

________________________________________________________________

### Required ambiguity decision — Material / Mobile Robot
Choose none until reviewed.
- [ ] A. Material Area only
- [ ] B. Mobile Robot Operation Area only
- [ ] C. Distinct but overlapping semantic layers
- [ ] D. Qwen role reinterpretation; keep Material Area
- [ ] E. UNSURE

Ambiguity comment: ____________________________________________

