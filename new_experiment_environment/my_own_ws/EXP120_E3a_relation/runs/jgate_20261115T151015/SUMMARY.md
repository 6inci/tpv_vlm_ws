# EXP120 J-gate 탐색 (dev, 모델 재호출 없음) — 자동 생성

- 정답: `referents_draft.json` (초안, 사용자 검토 대기)
- 규칙: `config.json` jgate. 입력 실행의 J-vlm, J-geo 결과는 바꾸지 않는다.

| 후보 | 조건 | 묶음 | 정답 | 잘못된 선택 | 올바른 선택 | 놓침 | 올바른 보류 | 정답 없는데 선택 | 막은 오선택 | 막힌 정상 선택 | 표기 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | J-vlm | spatial | 9/12 | 2 | 4 | 1 | 5 | 2 | - | - | diagnostic (answer-key information) - not system performance |
| oracle | J-vlm | state | 6/12 | 2 | 6 | 4 | 0 | 0 | - | - | diagnostic (answer-key information) - not system performance |
| oracle | J-vlm | all | 15/24 | 4 | 10 | 5 | 5 | 2 | - | - | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-exist(L) | spatial | 11/12 | 0 | 4 | 1 | 7 | 0 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-exist(L) | state | 6/12 | 2 | 6 | 4 | 0 | 0 | 0 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-exist(L) | all | 17/24 | 2 | 10 | 5 | 7 | 0 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-exist(M) | spatial | 11/12 | 0 | 4 | 1 | 7 | 0 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-exist(M) | state | 6/12 | 2 | 6 | 4 | 0 | 0 | 0 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-exist(M) | all | 17/24 | 2 | 10 | 5 | 7 | 0 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-exist(D) | spatial | 9/12 | 2 | 4 | 1 | 5 | 2 | 0 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-exist(D) | state | 6/12 | 2 | 6 | 4 | 0 | 0 | 0 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-exist(D) | all | 15/24 | 4 | 10 | 5 | 5 | 2 | 0 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-exist(G) | spatial | 11/12 | 0 | 4 | 1 | 7 | 0 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-exist(G) | state | 6/12 | 2 | 6 | 4 | 0 | 0 | 0 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-exist(G) | all | 17/24 | 2 | 10 | 5 | 7 | 0 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-agree(M) | spatial | 11/12 | 0 | 4 | 1 | 7 | 0 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-agree(M) | state | 6/12 | 0 | 6 | 6 | 0 | 0 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-agree(M) | all | 17/24 | 0 | 10 | 7 | 7 | 0 | 4 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-agree(D) | spatial | 10/12 | 0 | 3 | 2 | 7 | 0 | 2 | 1 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-agree(D) | state | 6/12 | 0 | 6 | 6 | 0 | 0 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-agree(D) | all | 16/24 | 0 | 9 | 8 | 7 | 0 | 4 | 1 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-agree(G) | spatial | 11/12 | 0 | 4 | 1 | 7 | 0 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-agree(G) | state | 6/12 | 0 | 6 | 6 | 0 | 0 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| oracle | J-gate-agree(G) | all | 17/24 | 0 | 10 | 7 | 7 | 0 | 4 | 0 | diagnostic (answer-key information) - not system performance |
| detected | J-vlm | spatial | 8/12 | 3 | 4 | 1 | 4 | 3 | - | - | dev exploration with system-available information (EXP111 detections + list or detector) |
| detected | J-vlm | state | 4/12 | 3 | 4 | 5 | 0 | 0 | - | - | dev exploration with system-available information (EXP111 detections + list or detector) |
| detected | J-vlm | all | 12/24 | 6 | 8 | 6 | 4 | 3 | - | - | dev exploration with system-available information (EXP111 detections + list or detector) |
| detected | J-gate-exist(L) | spatial | 10/12 | 1 | 4 | 1 | 6 | 1 | 2 | 0 | dev exploration with system-available information (EXP111 detections + list or detector) |
| detected | J-gate-exist(L) | state | 4/12 | 3 | 4 | 5 | 0 | 0 | 0 | 0 | dev exploration with system-available information (EXP111 detections + list or detector) |
| detected | J-gate-exist(L) | all | 14/24 | 4 | 8 | 6 | 6 | 1 | 2 | 0 | dev exploration with system-available information (EXP111 detections + list or detector) |
| detected | J-gate-exist(M) | spatial | 10/12 | 1 | 4 | 1 | 6 | 1 | 2 | 0 | map-type information, but the dev map is built from the answer key - not a system claim |
| detected | J-gate-exist(M) | state | 4/12 | 3 | 4 | 5 | 0 | 0 | 0 | 0 | map-type information, but the dev map is built from the answer key - not a system claim |
| detected | J-gate-exist(M) | all | 14/24 | 4 | 8 | 6 | 6 | 1 | 2 | 0 | map-type information, but the dev map is built from the answer key - not a system claim |
| detected | J-gate-exist(D) | spatial | 8/12 | 3 | 4 | 1 | 4 | 3 | 0 | 0 | dev exploration with system-available information (EXP111 detections + list or detector) |
| detected | J-gate-exist(D) | state | 4/12 | 3 | 4 | 5 | 0 | 0 | 0 | 0 | dev exploration with system-available information (EXP111 detections + list or detector) |
| detected | J-gate-exist(D) | all | 12/24 | 6 | 8 | 6 | 4 | 3 | 0 | 0 | dev exploration with system-available information (EXP111 detections + list or detector) |
| detected | J-gate-exist(G) | spatial | 10/12 | 1 | 4 | 1 | 6 | 1 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| detected | J-gate-exist(G) | state | 4/12 | 3 | 4 | 5 | 0 | 0 | 0 | 0 | diagnostic (answer-key information) - not system performance |
| detected | J-gate-exist(G) | all | 14/24 | 4 | 8 | 6 | 6 | 1 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| detected | J-gate-agree(M) | spatial | 10/12 | 1 | 4 | 1 | 6 | 1 | 2 | 0 | map-type information, but the dev map is built from the answer key - not a system claim |
| detected | J-gate-agree(M) | state | 3/12 | 0 | 3 | 9 | 0 | 0 | 3 | 1 | map-type information, but the dev map is built from the answer key - not a system claim |
| detected | J-gate-agree(M) | all | 13/24 | 1 | 7 | 10 | 6 | 1 | 5 | 1 | map-type information, but the dev map is built from the answer key - not a system claim |
| detected | J-gate-agree(D) | spatial | 10/12 | 0 | 3 | 2 | 7 | 0 | 3 | 1 | dev exploration with system-available information (EXP111 detections + list or detector) |
| detected | J-gate-agree(D) | state | 3/12 | 0 | 3 | 9 | 0 | 0 | 3 | 1 | dev exploration with system-available information (EXP111 detections + list or detector) |
| detected | J-gate-agree(D) | all | 13/24 | 0 | 6 | 11 | 7 | 0 | 6 | 2 | dev exploration with system-available information (EXP111 detections + list or detector) |
| detected | J-gate-agree(G) | spatial | 10/12 | 1 | 4 | 1 | 6 | 1 | 2 | 0 | diagnostic (answer-key information) - not system performance |
| detected | J-gate-agree(G) | state | 3/12 | 0 | 3 | 9 | 0 | 0 | 3 | 1 | diagnostic (answer-key information) - not system performance |
| detected | J-gate-agree(G) | all | 13/24 | 1 | 7 | 10 | 6 | 1 | 5 | 1 | diagnostic (answer-key information) - not system performance |
