# NLA 비교 요약 — run `20261118_185651_smoke_8a5b` (prompt n0, EXPLORATORY_DIAGNOSTIC)

> Exploratory diagnostic. No accuracy is computed. dest_iou is None when either side withheld.

| 이미지 | 수준 | 상태 | 결정 | 목적지 기능 이름 | 신뢰도 | 목적지 박스(px) | 지시가 줬다고 주장 / 직접 일치하지 않는 주장 | 모델이 덧붙인 것 (image/지식/가정) | 토큰 |
|---|---|---|---|---|---|---|---|---|---|
| D1 | L1 | INVALID_RESPONSE | - | - | - | - | - | - | 1585 |
| D1 | L2 | OK | withheld | - | - | - | 1 / 0 | 2/2/0 | 1018 |
| D1 | L3 | OK | proposed | truck loading zone | high | [1267.2, 64.8, 1843.2, 280.8] | 1 / 0 | 1/2/0 | 1492 |

## 목적지 박스끼리의 겹침 (모델 자신의 박스끼리, 정확도 아님; 한쪽이라도 보류면 None)

| 종류 | 대상 | a | b | IoU |
|---|---|---|---|---|
| same_image_different_instruction | D1 | L1 | L2 | None |
| same_image_different_instruction | D1 | L1 | L3 | None |
| same_image_different_instruction | D1 | L2 | L3 | None |

## 모델 응답 (원문 그대로)

### D1 / L1 — 지시: 빨간 큐브를 트럭이 있는 하역장으로 옮겨줘.
- 상태 INVALID_RESPONSE

### D1 / L2 — 지시: 빨간 큐브를 출하할 수 있는 적절한 장소로 옮겨줘.
- 추론한 작업 목적: Move a red cube to an appropriate location for shipment.
- 결정: `withheld` / 필요한 장소: A designated shipping zone or vehicle loading area.
- 이유: The red cube is not visible in the image, and no specific area is marked as a shipping zone. The truck is a likely candidate, but no explicit instruction or visual cue confirms it as the designated destination.
- 지시문이 줬다고 한 것: ['The destination must be a place where the red cube can be shipped.']
  - 그중 지시문에 글자로 없는 용어를 포함한 주장(휴리스틱): 없음
- 모델이 덧붙인 것: [('image', 'The red cube is not visible in the image, so its location is unknown.'), ('general_knowledge', 'The truck is a likely destination for shipment.'), ('image', 'The green pathway may be a designated route for moving goods.'), ('general_knowledge', 'The area near the truck is a logical place for outgoing goods.')]
- 제안한 영역들: ['area_1: loading/unloading zone (high)']
- 확인 불가로 적은 것: ['The location of the red cube', 'The exact function of the whiteboard', 'The current status of the robotic vehicle']
- 물리 검증이 필요하다고 적은 것: ['Whether the red cube is present and its current location', 'Whether the truck is accessible for loading', 'Whether the green pathway is clear for movement']
- 빨간 큐브 보고: 없음 / count 0으로 적은 객체: 없음

### D1 / L3 — 지시: 생산된 빨간 큐브를 다음 물류 단계로 전달해 줘.
- 추론한 작업 목적: To move a red cube from the production area to the next logistics stage.
- 결정: `proposed` / 필요한 장소: A designated area for outgoing material transfer, such as the truck loading zone.
- 이유: The instruction requests delivery to the next logistics stage, and the truck loading zone is the most logical destination for outgoing materials in this model.
- 목적지 `area_5` = truck loading zone (high), 박스 [1267.2, 64.8, 1843.2, 280.8]
  - 관찰 근거: ['truck', 'cones']
  - 추론: Truck and cones suggest a designated area for loading or unloading
- 지시문이 줬다고 한 것: ['생산된 빨간 큐브를 다음 물류 단계로 전달해 줘.']
  - 그중 지시문에 글자로 없는 용어를 포함한 주장(휴리스틱): 없음
- 모델이 덧붙인 것: [('image', 'The red cube is not visible in the image, so its location is unknown.'), ('general_knowledge', 'The next logistics stage is likely the area where materials are loaded onto the truck or stored for transport.'), ('general_knowledge', 'The truck loading zone (area_5) is a logical destination for outgoing logistics.')]
- 제안한 영역들: ['area_1: production zone (high)', 'area_2: central staging area (medium)', 'area_3: loading/unloading zone (high)', 'area_4: transportation corridor (high)', 'area_5: truck loading zone (high)', 'area_6: work instruction zone (medium)']
- 확인 불가로 적은 것: ['The location of the red cube is not visible in the image.', 'The exact function of the whiteboard is not clear from the image alone.', 'The current state of the robotic vehicle is unknown.']
- 물리 검증이 필요하다고 적은 것: ['Whether the red cube is present and located in the production zone.', 'Whether the robotic vehicle is operational and can carry the cube.', 'Whether the truck is ready to receive the cube.']
- 빨간 큐브 보고: 없음 / count 0으로 적은 객체: 없음
