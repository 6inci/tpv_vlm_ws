# 조건 비교 요약 — run `20261118_154227_dev_61bc` (prompt v3, split dev)

> No accuracy is computed. dest_iou compares the model's own boxes with each other (consistency).

| 이미지 | 조건 | 상태 | 결정 | 목적지 기능 이름 | 신뢰도 | 목적지 박스(px) | 이미지 대비 면적 | 트럭을 근거로 인용 | 영역/객체 수 | 토큰 | 시간(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | A | OK | proposed | unloading zone | high | [1401.6, 0.0, 1728.0, 302.4] | 0.0476 | True | 6/12 | 1454 | 43.667 |
| D1 | B | OK | proposed | proposed destination area for unloading | high | [1190.4, 0.0, 1824.0, 302.4] | 0.0924 | True | 6/15 | 1632 | 48.455 |
| D1 | C1 | OK | proposed | loading/unloading zone for truck | high | [1209.6, 0.0, 1766.4, 302.4] | 0.0812 | True | 6/12 | 1488 | 44.409 |

## 목적지 박스끼리의 겹침 (IoU, 일관성 지표이며 정확도가 아님)

| 종류 | 대상 | a | b | IoU |
|---|---|---|---|---|
| same_image_different_condition | D1 | A | B | 0.515 |
| same_image_different_condition | D1 | A | C1 | 0.586 |
| same_image_different_condition | D1 | B | C1 | 0.879 |

## 모델 응답 요약 (원문 그대로)

### D1 / A
- decision: `proposed`, 필요한 장소: unloading zone for the truck
- 이유: The instruction asks to place the red cubes at the truck's unloading area, which is clearly marked by the truck's position and surrounding cones.
- 목적지 `area_2` = unloading zone (high)
  - 관찰 근거: ['truck', 'cones', 'green pathway']
  - 추론: The truck is parked here, surrounded by cones, indicating this is a designated area for unloading goods.
- 제안한 영역들: ['area_1: production and loading zone (high)', 'area_2: unloading zone (high)', 'area_3: central storage and sorting area (medium)', 'area_4: material storage and staging area (medium)', 'area_5: loading/unloading access point (medium)', 'area_6: proposed unloading destination (high)']
- 관찰한 객체: ['robotic arm x4 (top-left and bottom-left corners)', 'truck x1 (top-right corner)', 'conveyor belt x1 (left side, near the robotic arms)', 'stack of boxes x1 (center of the floor)', 'cones x12 (distributed across the floor)', 'robotic vehicle x1 (bottom-right corner)', 'whiteboard x1 (bottom-center)', 'green pathway x1 (top-center)', 'storage racks x2 (top-center and bottom-left)', 'barrels x5 (right side, near the truck)', 'charging station x1 (bottom-right)', 'grate x1 (bottom-center)']
- 빨간 큐브 보고: 없음
- 확인 불가로 적은 것: ['The location of the red cubes on the conveyor belt', 'Whether the robotic arms are actively moving or not', 'The exact size or shape of the red cubes']
- 물리 검증이 필요하다고 적은 것: ['Whether the red cubes are actually present on the conveyor belt', 'Whether the robotic arms are capable of picking up the red cubes', 'Whether the truck is accessible for unloading']
- 목적지와 겹치는 다른 제안 영역: [{'id': 'area_6', 'function_name': 'proposed unloading destination', 'iou': 1.0, 'other_inside_destination': 1.0}]
- 트럭을 근거로 든 영역: ['area_2', 'area_6']

### D1 / B
- decision: `proposed`, 필요한 장소: unloading zone for a truck
- 이유: The instruction asks to unload a red cube from a conveyor belt to the truck's location, which is clearly marked as a designated unloading zone with traffic cones and a pathway.
- 목적지 `area_6` = proposed destination area for unloading (high)
  - 관찰 근거: ['truck', 'orange traffic cones', 'green pathway']
  - 추론: The truck is located here, and the area is marked with cones and a pathway, indicating it is the designated unloading zone.
- 제안한 영역들: ['area_1: production and sorting zone (high)', 'area_2: loading/unloading zone (high)', 'area_3: storage and material handling area (medium)', 'area_4: central work area (medium)', 'area_5: unloading and delivery zone (medium)', 'area_6: proposed destination area for unloading (high)']
- 관찰한 객체: ['robotic arm x4 (top-left and bottom-left corners)', 'truck x1 (top-right)', 'conveyor belt x1 (left side, near robotic arms)', 'stack of boxes x1 (center of the floor)', 'orange traffic cones x12 (distributed around the floor, especially near green pathways and truck area)', 'green pathway x1 (top-center)', 'black barrels x4 (bottom-right, near a black vehicle)', 'blue barrels x4 (bottom-right, near black barrels)', 'white pallet x1 (bottom-right, near black barrels)', 'black vehicle x1 (bottom-right)', 'whiteboard x1 (bottom-center)', 'gray metal grate x1 (bottom-center, near whiteboard)', 'green metal structure x2 (top-center and bottom-left)', 'white container x1 (bottom-right, near black barrels)', 'red object x1 (bottom-right, near black vehicle)']
- 빨간 큐브 보고: 없음
- 확인 불가로 적은 것: ['The location of the red cube on the conveyor belt', 'The exact position of the robotic arm handling the red cube', 'The current state of the conveyor belt (whether it is active or idle)']
- 물리 검증이 필요하다고 적은 것: ['Whether the red cube is physically present on the conveyor belt', 'Whether the robotic arm is actively moving or has completed its task', 'Whether the truck is ready to receive the red cube']
- 목적지와 겹치는 다른 제안 영역: [{'id': 'area_2', 'function_name': 'loading/unloading zone', 'iou': 1.0, 'other_inside_destination': 1.0}]
- 트럭을 근거로 든 영역: ['area_2', 'area_6']
- 일관성 문제: ['observed_objects has 15 items, more than the requested 12']

### D1 / C1
- decision: `proposed`, 필요한 장소: unloading zone for the truck
- 이유: The instruction asks to place a red cube at the truck's unloading zone, which is clearly marked and designated for vehicle loading/unloading.
- 목적지 `area_2` = loading/unloading zone for truck (high)
  - 관찰 근거: ['truck', 'orange traffic cones', 'white dashed lines']
  - 추론: The truck is parked in a marked area with cones, indicating a designated loading/unloading zone.
- 제안한 영역들: ['area_1: conveyor belt work cell (high)', 'area_2: loading/unloading zone for truck (high)', 'area_3: central staging area (medium)', 'area_4: storage area (medium)', 'area_5: emergency or power zone (medium)', 'area_6: unloading zone for truck (high)']
- 관찰한 객체: ['robotic arm x4 (top-left and bottom-left corners of the floor)', 'truck x1 (top-right corner of the floor)', 'conveyor belt x1 (left side of the floor, near the robotic arms)', 'stack of boxes x1 (center of the floor)', 'orange traffic cones x12 (distributed across the floor, especially near the green path and around the truck)', 'green path x1 (top-center of the floor)', 'black barrels x4 (bottom-right corner, on a pallet)', 'blue barrels x4 (bottom-right corner, on a pallet)', 'white pallet x1 (bottom-right corner, near the blue barrels)', 'black robot x1 (bottom-right corner, near the barrels)', 'whiteboard x1 (bottom-center of the floor)', 'green metal racks x2 (top-center and bottom-left)']
- 빨간 큐브 보고: 없음
- 확인 불가로 적은 것: ['The location of the red cube', "The exact position of the conveyor belt's output point", 'The current state of the conveyor belt (whether it is active or not)']
- 물리 검증이 필요하다고 적은 것: ['Whether the red cube is present and where it is located on the conveyor belt', 'Whether the conveyor belt is currently moving or stationary', 'Whether the truck is currently parked or moving']
- 목적지와 겹치는 다른 제안 영역: []
- 트럭을 근거로 든 영역: ['area_2', 'area_6']
