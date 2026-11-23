# EXP120 E3a feasibility — oracle (diagnostic upper bound, not a system result)

- referents: `referents_draft.json` (DRAFT by Claude 2026-11-15 from the answer-key positions and...)
- 자동 생성 파일 (`tools/e3a_poc.py`). 숫자는 rows.json에서 다시 계산할 수 있다.

| 조건/묶음 | 항목 | 정답 있음 | correct_target | wrong_target | missed | correct_abstain | false_select | invalid |
|---|---|---|---|---|---|---|---|---|
| J-vlm/spatial | 12 | 5 | 4 | 0 | 1 | 5 | 2 | 0 |
| J-vlm/state | 12 | 12 | 6 | 2 | 4 | 0 | 0 | 0 |
| J-vlm/all | 24 | 17 | 10 | 2 | 5 | 5 | 2 | 0 |
| J-geo/spatial | 12 | 5 | 5 | 0 | 0 | 7 | 0 | 0 |
| J-geo/state | 12 | 12 | 12 | 0 | 0 | 0 | 0 | 0 |
| J-geo/all | 24 | 17 | 17 | 0 | 0 | 7 | 0 | 0 |

| 배치 | 질의 | 정답 | J-vlm 번호 | J-vlm 결과 | J-geo 번호 (이유) | J-geo 결과 |
|---|---|---|---|---|---|---|
| P01 | e01 | 1 | None | missed | 1 (ok) | correct_target |
| P01 | e02 | none | None | correct_abstain | None (none) | correct_abstain |
| P01 | e03 | 2 | None | missed | 2 (ok) | correct_target |
| P01 | e04 | 1 | 1 | correct_target | 1 (ok) | correct_target |
| P01 | e05 | 1 | 2 | wrong_target | 1 (ok) | correct_target |
| P01 | e06 | unknown | 2 | false_select | None (unknown) | correct_abstain |
| P02 | e01 | 1 | 1 | correct_target | 1 (ok) | correct_target |
| P02 | e02 | 2 | 2 | correct_target | 2 (ok) | correct_target |
| P02 | e03 | 1 | None | missed | 1 (ok) | correct_target |
| P02 | e04 | 2 | 2 | correct_target | 2 (ok) | correct_target |
| P02 | e05 | 2 | 2 | correct_target | 2 (ok) | correct_target |
| P02 | e06 | unknown | None | correct_abstain | None (unknown) | correct_abstain |
| P03 | e01 | 2 | 2 | correct_target | 2 (ok) | correct_target |
| P03 | e02 | none | None | correct_abstain | None (none) | correct_abstain |
| P03 | e03 | 1 | None | missed | 1 (ok) | correct_target |
| P03 | e04 | 2 | 1 | wrong_target | 2 (ok) | correct_target |
| P03 | e05 | 2 | 2 | correct_target | 2 (ok) | correct_target |
| P03 | e06 | unknown | 1 | false_select | None (unknown) | correct_abstain |
| P04 | e01 | 2 | 2 | correct_target | 2 (ok) | correct_target |
| P04 | e02 | none | None | correct_abstain | None (none) | correct_abstain |
| P04 | e03 | 1 | None | missed | 1 (ok) | correct_target |
| P04 | e04 | 2 | 2 | correct_target | 2 (ok) | correct_target |
| P04 | e05 | 2 | 2 | correct_target | 2 (ok) | correct_target |
| P04 | e06 | unknown | None | correct_abstain | None (unknown) | correct_abstain |
