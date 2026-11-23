# V1 대 V2 검증 비교 (2026-11-18 21:42)

## 저장된 VLM 제안: V1 대 V2 (오프라인, 모델 호출 없음)

V1 재현(replay)이 저장된 V1 결과와 모든 검사에서 일치: **True** (False이면 이 비교는 신뢰할 수 없다)

| 사례 | VLM 제안 | V1 결정 (실패한 검사 / 미실행) | V2 의미 판정 | V2 최종 | 의미 수준 일치 | 계획 전달 수준 일치 |
|---|---|---|---|---|---|---|
| DEV_REF | loading/unloading zone | V1 primary: SEMANTIC_NOT_VERIFIED (V3 / V4) | EVIDENCE_RELATION_INCONSISTENT | EVIDENCE_RELATION_INCONSISTENT | AGREE_BLOCK | AGREE_BLOCK |
| E1 | loading dock | V1 primary: SEMANTIC_NOT_VERIFIED (V3b / V4) | SEMANTICALLY_SUPPORTED | SEMANTICALLY_SUPPORTED_PLACEMENT_UNVERIFIED | V1_BLOCKS__V2_ACCEPTS (V1 false-reject / conflation candidate) | AGREE_BLOCK |
| E1 | loading dock | V1 variant verification.max_unrefined_area_ratio=6.0: ADOPTED_SEMANTIC_ONLY_MAP_NOT_ALIGNED (- / -) | SEMANTICALLY_SUPPORTED | SEMANTICALLY_SUPPORTED_PLACEMENT_UNVERIFIED | AGREE_ACCEPT | AGREE_BLOCK |
| REF_P01 | exit/destination zone | V1 primary: SEMANTIC_NOT_VERIFIED (V2, V3 / V4) | EVIDENCE_NOT_ESTABLISHED | EVIDENCE_NOT_ESTABLISHED | AGREE_BLOCK | AGREE_BLOCK |

### DEV_REF
- 지시문의 목적: ['outbound_shipping'] → 필요한 능력 ['carrier_vehicle']
- VLM 제안: loading/unloading zone (VLM의 기능 설명은 근거로 쓰지 않음)
  - 근거 `pallets with containers`: registry ['pallet', 'container'] → **NOT_RUN**
  - 근거 `dump truck`: registry ['truck'] → **VERIFIED_PRESENT** / RELATION_FAILS
  - 근거 `orange cones`: registry ['traffic_cone'] → **NOT_RUN**
    - truck `truck_1`: 영역 안 비율 0.0, 간격 227.7 px (인접 허용 119.5 px) → RELATION_FAILS
- 의미 판정: **EVIDENCE_RELATION_INCONSISTENT** (근거와 공간 관계가 불일치하는 후보): verified 'truck' exists but none of its 1 instance(s) is inside or adjacent to the proposed region
- 배치: NOT_ESTABLISHED — no verified localization of a placeable spot inside the semantic region (evidence-grounded refinement or a human-confirmed region); the region conflicts with other equipment or with the robot's travel/no-go zones
- 계획 인터페이스: UNSUPPORTED_OBSTACLE_COVERAGE_INCOMPLETE (모델링되지 않은 물리 근거 [], 관찰되지 않은 장애물 후보 클래스 ['container', 'pallet', 'traffic_cone'])
- 계획 전제조건: 막힘 없음
- **최종: EVIDENCE_RELATION_INCONSISTENT** (근거와 공간 관계가 불일치하는 후보), 계획기로 전달 False
- V1 primary (`20261118_202916_fbe7`): SEMANTIC_NOT_VERIFIED; 실패/미실행 ['V3']; ['V3 FAILED: region is geometrically associated with the verified anchor (v0.2 association rules)', 'V4 NOT_RUN: v0.2 region_judge accepts the adopted region']

### E1
- 지시문의 목적: ['outbound_shipping'] → 필요한 능력 ['carrier_vehicle']
- VLM 제안: loading dock (VLM의 기능 설명은 근거로 쓰지 않음)
  - 근거 `truck`: registry ['truck'] → **VERIFIED_PRESENT** / RELATION_HOLDS
  - 근거 `cones`: registry ['traffic_cone'] → **VERIFIED_PRESENT** / RELATION_HOLDS
  - 근거 `barriers`: unmatched_object [] → **UNVERIFIABLE**
    - traffic_cone `traffic_cone_1`: 영역 안 비율 1.0, 간격 0.0 px (인접 허용 60 px) → RELATION_HOLDS
    - traffic_cone `traffic_cone_2`: 영역 안 비율 0.694, 간격 0.0 px (인접 허용 60 px) → RELATION_HOLDS
    - traffic_cone `traffic_cone_3`: 영역 안 비율 1.0, 간격 0.0 px (인접 허용 60 px) → RELATION_HOLDS
    - traffic_cone `traffic_cone_4`: 영역 안 비율 0.725, 간격 0.0 px (인접 허용 60 px) → RELATION_HOLDS
    - traffic_cone `traffic_cone_5`: 영역 안 비율 0.914, 간격 0.0 px (인접 허용 60 px) → RELATION_HOLDS
    - traffic_cone `traffic_cone_6`: 영역 안 비율 0.822, 간격 0.0 px (인접 허용 60 px) → RELATION_HOLDS
    - truck `truck_1`: 영역 안 비율 1.0, 간격 0.0 px (인접 허용 144.1 px) → RELATION_HOLDS
- 의미 판정: **SEMANTICALLY_SUPPORTED** (의미적으로 타당한 후보): verified 'truck' stands in a relation to the region and carries ['carrier_vehicle']
- 배치: NOT_ESTABLISHED — no verified localization of a placeable spot inside the semantic region (evidence-grounded refinement or a human-confirmed region)
- 계획 인터페이스: UNSUPPORTED_OBSTACLE_COVERAGE_INCOMPLETE (모델링되지 않은 물리 근거 ['barriers'], 관찰되지 않은 장애물 후보 클래스 [])
- 계획 전제조건: ['map alignment: MISALIGNED', 'red cube: TARGET_NOT_FOUND']
- **최종: SEMANTICALLY_SUPPORTED_PLACEMENT_UNVERIFIED** (의미적으로는 타당하지만 배치 가능성이 검증되지 않은 후보), 계획기로 전달 False
- V1 primary (`20261118_203058_46d8`): SEMANTIC_NOT_VERIFIED; 실패/미실행 ['V3b']; ['V3b FAILED: spatial precision of the adopted region', 'V4 NOT_RUN: v0.2 region_judge accepts the adopted region']
- V1 variant verification.max_unrefined_area_ratio=6.0 (`20261118_203426_260d`): ADOPTED_SEMANTIC_ONLY_MAP_NOT_ALIGNED; 실패/미실행 []; ['spatial precision: coarse_clipped_to_association_zone']

### REF_P01
- 지시문의 목적: ['outbound_shipping'] → 필요한 능력 ['carrier_vehicle']
- VLM 제안: exit/destination zone (VLM의 기능 설명은 근거로 쓰지 않음)
  - 근거 `lightning bolt symbol`: marking [] → **UNVERIFIABLE**
  - 근거 `red cube marker (assumed)`: registry ['red_cube'] → **NOT_EVIDENCE** / VLM 스스로 추정 표현
  - 근거 `whiteboard nearby`: unmatched_object [] → **UNVERIFIABLE**
- 의미 판정: **EVIDENCE_NOT_ESTABLISHED** (시각적 근거가 확인되지 않은 후보): no cited object could be verified: 0 not found, 0 not run, 2 claim(s) UNVERIFIABLE (no detector)
- 배치: NOT_ESTABLISHED — no verified localization of a placeable spot inside the semantic region (evidence-grounded refinement or a human-confirmed region); the region conflicts with other equipment or with the robot's travel/no-go zones
- 계획 인터페이스: UNSUPPORTED_OBSTACLE_COVERAGE_INCOMPLETE (모델링되지 않은 물리 근거 ['whiteboard nearby'], 관찰되지 않은 장애물 후보 클래스 [])
- 계획 전제조건: 막힘 없음
- **최종: EVIDENCE_NOT_ESTABLISHED** (시각적 근거가 확인되지 않은 후보), 계획기로 전달 False
- V1 primary (`20261118_203247_fd48`): SEMANTIC_NOT_VERIFIED; 실패/미실행 ['V2', 'V3']; ['V3 FAILED: region is geometrically associated with the verified anchor (v0.2 association rules)', 'V2 FAILED: the objects cited as evidence exist in the image (anchor object required)', 'V4 NOT_RUN: v0.2 region_judge accepts the adopted region']

## 합성 시나리오: V1 대 V2

| 시나리오 | V1 | V2 (VLM이 고른 후보) | 의미 수준 | 계획 전달 수준 | 기대와 일치 (V2/V1) |
|---|---|---|---|---|---|
| S1_consistent_and_localized | ADOPTED (-) | SEMANTICALLY_SUPPORTED_READY_FOR_PLANNING | AGREE_ACCEPT | AGREE_PASS | True/True |
| S2_evidence_object_absent | SEMANTIC_NOT_VERIFIED (V2, V3) | EVIDENCE_NOT_ESTABLISHED | AGREE_BLOCK | AGREE_BLOCK | True/True |
| S3_evidence_elsewhere | SEMANTIC_NOT_VERIFIED (V3) | EVIDENCE_RELATION_INCONSISTENT | AGREE_BLOCK | AGREE_BLOCK | True/True |
| S4_multiple_candidates | SEMANTIC_NOT_VERIFIED (V2, V3) | EVIDENCE_NOT_ESTABLISHED | AGREE_BLOCK | AGREE_BLOCK | True/True |
| S5_semantic_ok_placement_unverified | SEMANTIC_NOT_VERIFIED (V3) | SEMANTICALLY_SUPPORTED_PLACEMENT_UNVERIFIED | V1_BLOCKS__V2_ACCEPTS (V1 false-reject / conflation candidate) | AGREE_BLOCK | True/True |
| S6_non_truck_carrier | SEMANTIC_NOT_VERIFIED (V2, V3) | SEMANTICALLY_SUPPORTED_READY_FOR_PLANNING | V1_BLOCKS__V2_ACCEPTS (V1 false-reject / conflation candidate) | V1_BLOCKS__V2_PASSES | True/True |
| S7_purpose_mismatch | ADOPTED (-) | PURPOSE_FIT_UNDETERMINED | V1_ACCEPTS__V2_BLOCKS (V1 false-accept candidate) | V1_PASSES__V2_BLOCKS | True/True |
| S8_unmodeled_obstacle_claim | ADOPTED (-) | SEMANTICALLY_SUPPORTED_PLANNER_UNSUPPORTED | AGREE_ACCEPT | V1_PASSES__V2_BLOCKS | True/True |
| S9_purpose_unknown | ADOPTED (-) | PURPOSE_FIT_UNDETERMINED | V1_ACCEPTS__V2_BLOCKS (V1 false-accept candidate) | V1_PASSES__V2_BLOCKS | True/True |
| S10_other_function_verified | SEMANTIC_NOT_VERIFIED (V2, V3) | PURPOSE_FIT_UNDETERMINED | AGREE_BLOCK | AGREE_BLOCK | True/True |

- **S1_consistent_and_localized**: evidence exists, stands in the right relation, spot localized. V2 이유: verified 'truck' stands in a relation to the region and carries ['carrier_vehicle']
- **S2_evidence_object_absent**: the VLM cites a truck; the detector searched and found none. V2 이유: cited 'truck' (needed for the purpose) was searched for and not found
- **S3_evidence_elsewhere**: the cited truck exists but stands far from the proposed region. V2 이유: verified 'truck' exists but none of its 1 instance(s) is inside or adjacent to the proposed region
- **S4_multiple_candidates**: two destination candidates; the VLM chose the weaker one; the verifier does not swap. V2 이유: no cited object could be verified: 0 not found, 0 not run, 1 claim(s) UNVERIFIABLE (no detector) (후보별: {'area_1': 'SEMANTICALLY_SUPPORTED_READY_FOR_PLANNING', 'area_2': 'EVIDENCE_NOT_ESTABLISHED'} — NOT_MADE_BY_THE_VERIFIER: it never replaces the VLM's choice; a planner-side choice rule (v0.2: lowest goal score) is geometric, not a semantic criterion)
- **S5_semantic_ok_placement_unverified**: truck inside a wide region, no localization of a spot. V2 이유: verified 'truck' stands in a relation to the region and carries ['carrier_vehicle']
- **S6_non_truck_carrier**: a delivery van (registered as a carrier by CONFIG only) in a tight, localized region. V2 이유: verified 'delivery_van' stands in a relation to the region and carries ['carrier_vehicle']
- **S7_purpose_mismatch**: the instruction asks for a CHARGING place; the VLM proposes the truck bay. V2 이유: the purpose needs ['charging_equipment'], which no registered detector class can verify
- **S8_unmodeled_obstacle_claim**: the VLM also cites crates by the bay: a physical object no detector or obstacle model knows. V2 이유: verified 'truck' stands in a relation to the region and carries ['carrier_vehicle']
- **S9_purpose_unknown**: the instruction names no purpose the table knows. V2 이유: the instruction names no purpose the table knows, so no capability can be required
- **S10_other_function_verified**: pallets are verified inside the region, but they show storage, not shipping. V2 이유: verified, related evidence supports ['goods_staging'], not the capability the purpose needs (['carrier_vehicle'])
