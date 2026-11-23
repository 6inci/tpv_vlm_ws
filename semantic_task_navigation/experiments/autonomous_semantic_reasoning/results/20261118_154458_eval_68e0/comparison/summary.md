# 조건 비교 요약 — run `20261118_154458_eval_68e0` (prompt v3, split eval)

> No accuracy is computed. dest_iou compares the model's own boxes with each other (consistency).

| 이미지 | 조건 | 상태 | 결정 | 목적지 기능 이름 | 신뢰도 | 목적지 박스(px) | 이미지 대비 면적 | 트럭을 근거로 인용 | 영역/객체 수 | 토큰 | 시간(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | A | OK | proposed | truck unloading zone | high | [1209.6, 0.0, 1766.4, 302.4] | 0.0812 | True | 6/14 | 1611 | 48.982 |
| E1 | B | OK | proposed | truck unloading zone | high | [1209.6, 0.0, 1728.0, 302.4] | 0.0756 | True | 6/14 | 1638 | 49.655 |
| E1 | C1 | OK | proposed | loading/unloading zone for truck | high | [1209.6, 0.0, 1728.0, 302.4] | 0.0756 | True | 6/14 | 1606 | 48.132 |
| E2 | A | OK | proposed | truck unloading zone | high | [1200.0, 0.0, 1728.0, 302.4] | 0.077 | True | 6/11 | 1407 | 41.809 |
| E2 | B | OK | proposed | unloading zone | high | [1344.0, 0.0, 1728.0, 302.4] | 0.056 | True | 6/10 | 1325 | 39.655 |
| E2 | C1 | OK | proposed | loading/unloading zone for truck | high | [1200.0, 0.0, 1766.4, 302.4] | 0.0826 | True | 6/11 | 1406 | 42.478 |

## 목적지 박스끼리의 겹침 (IoU, 일관성 지표이며 정확도가 아님)

| 종류 | 대상 | a | b | IoU |
|---|---|---|---|---|
| same_image_different_condition | E1 | A | B | 0.931 |
| same_image_different_condition | E1 | A | C1 | 0.931 |
| same_image_different_condition | E1 | B | C1 | 1.0 |
| same_image_different_condition | E2 | A | B | 0.727 |
| same_image_different_condition | E2 | A | C1 | 0.932 |
| same_image_different_condition | E2 | B | C1 | 0.678 |
| same_condition_different_image | A | E1 | E2 | 0.915 |
| same_condition_different_image | B | E1 | E2 | 0.741 |
| same_condition_different_image | C1 | E1 | E2 | 0.915 |

## 모델 응답 요약 (원문 그대로)

### E1 / A
- decision: `proposed`, 필요한 장소: unloading zone for the truck
- 이유: The instruction asks to place a red cube at the truck's unloading zone, which is clearly marked by the white truck, orange cones, and designated parking lines in area_2.
- 목적지 `area_2` = truck unloading zone (high)
  - 관찰 근거: ['white truck model', 'orange cones marking a designated area', 'white dashed lines indicating parking or loading zone']
  - 추론: This area is designated for truck operations, likely for unloading or loading cargo.
- 제안한 영역들: ['area_1: conveyor belt production zone (high)', 'area_2: truck unloading zone (high)', 'area_3: central storage area (medium)', 'area_4: material storage and handling zone (medium)', 'area_5: charging or power station (medium)', 'area_6: instruction or control station (medium)']
- 관찰한 객체: ['robotic arm x3 (top-left and bottom-left corners)', 'truck x1 (top-right)', 'conveyor belt x1 (left side, near robotic arms)', 'stack of boxes x1 (center of the floor)', 'orange cone x12 (scattered across the floor)', 'green pathway x1 (top-center)', 'black cylindrical containers x4 (bottom-right and bottom-center)', 'blue cylindrical containers x4 (bottom-right)', 'white pallet x1 (bottom-right)', 'black vehicle x1 (bottom-right)', 'whiteboard x1 (bottom-center)', 'gray metal grate x1 (bottom-center)', 'green scaffolding x2 (top-center and bottom-left)', 'red object x1 (bottom-right)']
- 빨간 큐브 보고: 없음
- 확인 불가로 적은 것: ['The location of the red cube on the conveyor belt', 'Whether the red cube is currently being produced or transported', 'The exact size or shape of the red cube']
- 물리 검증이 필요하다고 적은 것: ['Whether the red cube is physically present and accessible on the conveyor belt', 'Whether the truck is stationary and ready to receive the cube', 'Whether the path from the conveyor belt to the truck is clear and unobstructed']
- 목적지와 겹치는 다른 제안 영역: []
- 트럭을 근거로 든 영역: ['area_2']
- 일관성 문제: ['observed_objects has 14 items, more than the requested 12']

### E1 / B
- decision: `proposed`, 필요한 장소: a designated unloading area for a truck
- 이유: The instruction asks for a place where a truck is located, and area_2 is the only area with a truck and marked boundaries consistent with an unloading zone.
- 목적지 `area_2` = truck unloading zone (high)
  - 관찰 근거: ['white truck model', 'orange cones marking a designated area', 'white dashed lines indicating parking or loading zone']
  - 추론: This area is designated for truck unloading, as indicated by the truck's position and surrounding cones.
- 제안한 영역들: ['area_1: conveyor belt production zone (high)', 'area_2: truck unloading zone (high)', 'area_3: central storage or staging area (medium)', 'area_4: material storage zone (medium)', 'area_5: charging or power station (medium)', 'area_6: instruction or labeling area (medium)']
- 관찰한 객체: ['robotic arm x3 (top-left and bottom-left corners)', 'truck x1 (top-right)', 'conveyor belt x1 (left side, near robotic arms)', 'stack of boxes x1 (center of the floor)', 'orange cone x12 (distributed across the floor)', 'green pathway x1 (top-center)', 'black cylindrical containers x4 (bottom-right and bottom-center)', 'blue cylindrical containers x6 (bottom-right)', 'white rectangular container x1 (bottom-right)', 'black vehicle x1 (bottom-right)', 'whiteboard x1 (bottom-center)', 'green scaffolding x2 (top-center and bottom-left)', 'gray grated floor x1 (bottom-center)', 'lightning symbol x1 (bottom-right)']
- 빨간 큐브 보고: 없음
- 확인 불가로 적은 것: ['The location of the red cube that needs to be moved', "The exact function of the whiteboard's writing", 'The current state of the conveyor belt (whether it is active or not)', 'Whether the robotic arms are currently operating or idle']
- 물리 검증이 필요하다고 적은 것: ['Whether the red cube is present and located on the conveyor belt', 'Whether the truck is currently available for unloading', 'Whether the area is clear and accessible for the robot to move the cube']
- 목적지와 겹치는 다른 제안 영역: []
- 트럭을 근거로 든 영역: ['area_2']
- 일관성 문제: ['observed_objects has 14 items, more than the requested 12']

### E1 / C1
- decision: `proposed`, 필요한 장소: unloading area for the truck
- 이유: The instruction asks to place a red cube at the truck's unloading zone, which is clearly marked by the truck's position and surrounding cones.
- 목적지 `area_2` = loading/unloading zone for truck (high)
  - 관찰 근거: ['truck', 'orange traffic cones', 'white dashed lines']
  - 추론: The truck is parked here, and the cones and dashed lines suggest this is a designated area for loading or unloading goods.
- 제안한 영역들: ['area_1: conveyor belt work cell (high)', 'area_2: loading/unloading zone for truck (high)', 'area_3: storage area (medium)', 'area_4: central staging area (medium)', 'area_5: emergency or power zone (low)', 'area_6: unloading zone for truck (medium)']
- 관찰한 객체: ['robotic arm x3 (top-left and bottom-left corners)', 'truck x1 (top-right)', 'conveyor belt x1 (left side, near the robotic arms)', 'stack of boxes x1 (center of the floor)', 'orange traffic cones x12 (scattered across the floor)', 'green path x1 (top-center)', 'black barrels x4 (bottom-right, near a black vehicle)', 'blue barrels x6 (bottom-right, near a black vehicle)', 'white pallet x1 (bottom-right, near black barrels)', 'black vehicle x1 (bottom-right)', 'whiteboard x1 (bottom-center)', 'green scaffolding x2 (top-center and bottom-left)', 'gray grated floor x1 (bottom-center)', 'lightning symbol x1 (bottom-right)']
- 빨간 큐브 보고: 없음
- 확인 불가로 적은 것: ['The location of the red cube', "The exact position of the conveyor belt's output point", 'The current state of the conveyor belt (whether it is active or not)']
- 물리 검증이 필요하다고 적은 것: ['Whether the red cube is present and located on the conveyor belt', 'Whether the truck is stationary and ready to receive the cube', 'Whether the unloading zone is clear of obstructions']
- 목적지와 겹치는 다른 제안 영역: []
- 트럭을 근거로 든 영역: ['area_2', 'area_6']
- 일관성 문제: ['observed_objects has 14 items, more than the requested 12']

### E2 / A
- decision: `proposed`, 필요한 장소: unloading zone for the truck
- 이유: The instruction specifies placing the red cube at the truck's unloading location, which is clearly marked by the truck and surrounding cones in area_2.
- 목적지 `area_2` = truck unloading zone (high)
  - 관찰 근거: ['white truck parked with a yellow box in its bed', 'orange cones marking a designated area']
  - 추론: The truck is positioned here, and the cones suggest this is a designated loading/unloading area.
- 제안한 영역들: ['area_1: conveyor belt production zone (high)', 'area_2: truck unloading zone (high)', 'area_3: central storage area (medium)', 'area_4: material storage zone (medium)', 'area_5: charging station (high)', 'area_6: proposed unloading destination (high)']
- 관찰한 객체: ['robotic arm x4 (top-left and bottom-left corners)', 'truck x1 (top-right)', 'conveyor belt x1 (left side, inside a white-bordered zone)', 'stack of boxes x1 (center of the floor)', 'orange traffic cone x10 (scattered across the floor)', 'battery charging station x1 (bottom-right)', 'whiteboard x1 (bottom-center)', 'green pathway x1 (top-center)', 'barrier x2 (top-center and bottom-left)', 'pallet with barrels x3 (right side, near center)', 'robotic vehicle x1 (bottom-right)']
- 빨간 큐브 보고: 없음
- 확인 불가로 적은 것: ['The location of the red cube (it is not visible in the image)', 'The exact function of the whiteboard', 'The real size or scale of the model']
- 물리 검증이 필요하다고 적은 것: ['Whether the red cube is physically present and accessible', 'Whether the truck is capable of receiving the cube', 'Whether the path from the conveyor to the truck is clear']
- 목적지와 겹치는 다른 제안 영역: [{'id': 'area_6', 'function_name': 'proposed unloading destination', 'iou': 1.0, 'other_inside_destination': 1.0}]
- 트럭을 근거로 든 영역: ['area_2', 'area_6']

### E2 / B
- decision: `proposed`, 필요한 장소: unloading zone for a truck
- 이유: The instruction asks to place an object at the truck's unloading zone, which is clearly marked and designated in the image with the truck parked and cones around it.
- 목적지 `area_2` = unloading zone (high)
  - 관찰 근거: ['truck', 'cones', 'marked parking space']
  - 추론: The truck is parked in a designated area with cones, indicating this is a designated unloading or loading zone.
- 제안한 영역들: ['area_1: production zone (high)', 'area_2: unloading zone (high)', 'area_3: storage area (medium)', 'area_4: charging station (high)', 'area_5: central work area (medium)', 'area_6: control or monitoring station (medium)']
- 관찰한 객체: ['robotic arm x4 (top-left and bottom-left corners)', 'truck x1 (top-right)', 'stack of boxes x1 (center)', 'cones x10 (scattered across the floor)', 'barriers x2 (top-center and bottom-left)', 'battery charging station x1 (bottom-right)', 'whiteboard x1 (bottom-center)', 'pallets with containers x3 (right-center and bottom-center)', 'grate x2 (bottom-center and top-center)', 'red button x1 (bottom-left)']
- 빨간 큐브 보고: 없음
- 확인 불가로 적은 것: ['The location of the red cubes (the object to be moved)', 'Whether the robotic arms are currently moving or have produced the cubes', 'The exact size or shape of the red cubes']
- 물리 검증이 필요하다고 적은 것: ['Whether the red cubes are actually present and located on the conveyor belt', 'Whether the robotic arms are currently operating and have produced the cubes', 'Whether the truck is accessible and ready to receive the cubes']
- 목적지와 겹치는 다른 제안 영역: []
- 트럭을 근거로 든 영역: ['area_2']

### E2 / C1
- decision: `proposed`, 필요한 장소: loading/unloading zone for the truck
- 이유: The instruction requires placing a red cube at the truck's unloading location, which is clearly marked by the truck, cones, and dashed lines in area_2.
- 목적지 `area_2` = loading/unloading zone for truck (high)
  - 관찰 근거: ['truck', 'orange cones', 'white dashed lines']
  - 추론: The truck is parked in a marked area with cones, indicating a designated loading/unloading zone.
- 제안한 영역들: ['area_1: production cell (high)', 'area_2: loading/unloading zone for truck (high)', 'area_3: storage area (high)', 'area_4: central staging area (medium)', 'area_5: charging station (high)', 'area_6: pedestrian/guided path (medium)']
- 관찰한 객체: ['robotic arm x4 (top-left and bottom-left corners)', 'truck x1 (top-right)', 'conveyor belt x1 (left side, near robotic arms)', 'stack of boxes x1 (center of the floor)', 'orange traffic cone x12 (distributed across the floor)', 'storage pallet x3 (right side, near the truck)', 'whiteboard x1 (bottom center)', 'charging station x1 (bottom-right)', 'green pathway x1 (top-center)', 'gray grated floor x2 (bottom center and bottom-right)', 'green metal racks x2 (top-center and bottom-left)']
- 빨간 큐브 보고: 없음
- 확인 불가로 적은 것: ['The location of the red cube (it is not visible in the image)', 'The exact function of the whiteboard text', 'The real size or scale of the model']
- 물리 검증이 필요하다고 적은 것: ['Whether the red cube is present and accessible', 'Whether the truck is stationary and ready to receive the cube', 'Whether the area is clear of obstructions for unloading']
- 목적지와 겹치는 다른 제안 영역: []
- 트럭을 근거로 든 영역: ['area_2']
