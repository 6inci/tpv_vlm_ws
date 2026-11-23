# X6 비교 M1 (run02) vs M3 (x6_M3_dev) — dev, 탐색

- 자동 생성 (`explore/X6_model_size/x6_compare.py`). 채점 규칙은 EXP110 등록 규칙과 같다.
- 실패 분류 (실패한 질의만, 서로 겹치지 않음): invalid = 출력 형식 오류, miss_only = 정답 일부/전부 누락, wrong_only = 정답은 다 찾았지만 잘못된 객체 추가, miss_and_wrong = 둘 다.

## C1

| 지표 | M1 en | M3 en | M1 ko | M3 ko |
|---|---|---|---|---|
| n | 196 | 196 | 192 | 192 |
| success_rate | 0.199 | 0.1939 | 0.1458 | 0.1458 |
| success | 39 | 38 | 28 | 28 |
| invalid | 16 | 3 | 12 | 0 |
| miss_only | 55 | 94 | 76 | 127 |
| wrong_only | 21 | 6 | 22 | 0 |
| miss_and_wrong | 65 | 55 | 54 | 37 |
| instance_recall | 0.3528 | 0.4355 | 0.1723 | 0.145 |
| false_positive_boxes | 187 | 95 | 216 | 48 |
| multi_n | 88 | 93 | 79 | 88 |
| multi_success | 0 | 2 | 0 | 0 |
| multi_partial_miss | 28 | 41 | 19 | 32 |
| multi_total_miss | 46 | 44 | 52 | 56 |
| multi_instance_recall | 0.4191 | 0.5104 | 0.1543 | 0.1469 |
| absent_n | 16 | 16 | 16 | 16 |
| absent_hallucinated | 0 | 0 | 4 | 0 |

소형 팔레트·블록 질의 (task objects)

| 지표 | M1 en | M3 en | M1 ko | M3 ko |
|---|---|---|---|---|
| n | 24 | 24 | 24 | 24 |
| success | 0 | 0 | 0 | 0 |
| instance_recall | 0.0 | 0.0 | 0.0 | 0.0 |
| false_positive_boxes | 23 | 23 | 17 | 6 |
| invalid | 3 | 0 | 2 | 0 |
| miss_only | 0 | 7 | 13 | 18 |
| wrong_only | 0 | 0 | 0 | 0 |
| miss_and_wrong | 21 | 17 | 9 | 6 |

짝 비교 (M3 − M1, 질의 묶음 bootstrap 95% CI가 주 근거, McNemar는 참고)

- en: 차이 -0.005, CI [-0.0612, 0.0459], M3만 성공 6, M1만 성공 7, 짝 196개, McNemar p=1.0
- ko: 차이 +0.000, CI [-0.0521, 0.0521], M3만 성공 7, M1만 성공 7, 짝 192개, McNemar p=1.0

| 시간·메모리 | M1 | M3 |
|---|---|---|
| model_calls | 388 | 388 |
| latency_median_s | 1.99 | 1.98 |
| latency_p90_s | 6.79 | 4.59 |
| latency_max_s | 32.98 | 29.39 |
| latency_sum_s | 1605.4 | 1190.2 |
| generated_tokens_median | 31.0 | 6.0 |
| hit_max_new_tokens | 17 | 1 |
| peak_vram_gib_reported | 8.915 | 7.014 |

성공이 갈린 질의 (M1만 성공 / M3만 성공 배치 수)

| 질의 | 언어 | 클래스 | 유형 | M1만 | M3만 | 둘 다 | 둘 다 실패 |
|---|---|---|---|---|---|---|---|
| q004 | en | storage_cubby | Q3 | 3 | 0 | 0 | 1 |
| q058 | en | ibc_tank | Q1 | 3 | 0 | 0 | 1 |
| q069 | ko | drum | Q2 | 3 | 0 | 1 | 0 |
| q072 | en | drum | Q2 | 0 | 3 | 1 | 0 |
| q094 | ko | None | Q5 | 0 | 3 | 1 | 0 |
| q046 | ko | box_stack | Q1 | 1 | 1 | 2 | 0 |
| q082 | ko | fire_extinguisher | Q1 | 2 | 0 | 0 | 2 |
| q041 | en | storage_rack | Q3 | 0 | 1 | 0 | 3 |
| q042 | ko | truck | Q1 | 0 | 1 | 3 | 0 |
| q044 | ko | truck | Q3 | 0 | 1 | 3 | 0 |
| q047 | en | box_stack | Q1 | 0 | 1 | 3 | 0 |
| q071 | ko | drum | Q2 | 1 | 0 | 0 | 3 |
| q083 | en | fire_extinguisher | Q1 | 1 | 0 | 0 | 3 |
| q088 | en | traffic_cone | Q3 | 0 | 1 | 0 | 3 |
| q092 | ko | None | Q5 | 0 | 1 | 3 | 0 |

## C6-dino

| 지표 | M1 en | M3 en | M1 ko | M3 ko |
|---|---|---|---|---|
| n | 196 | 196 | 192 | 192 |
| success_rate | 0.2551 | 0.1786 | 0.125 | 0.0885 |
| success | 50 | 35 | 24 | 17 |
| invalid | 2 | 0 | 0 | 0 |
| miss_only | 37 | 42 | 28 | 80 |
| wrong_only | 3 | 14 | 8 | 11 |
| miss_and_wrong | 104 | 105 | 132 | 84 |
| instance_recall | 0.381 | 0.4113 | 0.1261 | 0.1387 |
| false_positive_boxes | 182 | 303 | 174 | 141 |
| multi_n | 95 | 96 | 88 | 88 |
| multi_success | 0 | 2 | 0 | 0 |
| multi_partial_miss | 44 | 47 | 25 | 32 |
| multi_total_miss | 50 | 45 | 62 | 55 |
| multi_instance_recall | 0.3825 | 0.4248 | 0.116 | 0.1418 |
| absent_n | 16 | 16 | 16 | 16 |
| absent_hallucinated | 1 | 8 | 5 | 8 |

소형 팔레트·블록 질의 (task objects)

| 지표 | M1 en | M3 en | M1 ko | M3 ko |
|---|---|---|---|---|
| n | 24 | 24 | 24 | 24 |
| success | 0 | 0 | 0 | 0 |
| instance_recall | 0.0 | 0.025 | 0.0 | 0.0 |
| false_positive_boxes | 28 | 59 | 25 | 10 |
| invalid | 0 | 0 | 0 | 0 |
| miss_only | 0 | 0 | 3 | 14 |
| wrong_only | 0 | 0 | 0 | 0 |
| miss_and_wrong | 24 | 24 | 21 | 10 |

짝 비교 (M3 − M1, 질의 묶음 bootstrap 95% CI가 주 근거, McNemar는 참고)

- en: 차이 -0.076, CI [-0.1531, -0.0102], M3만 성공 5, M1만 성공 20, 짝 196개, McNemar p=0.00408
- ko: 차이 -0.036, CI [-0.0885, 0.0], M3만 성공 0, M1만 성공 7, 짝 192개, McNemar p=0.01562

| 시간·메모리 | M1 | M3 |
|---|---|---|
| model_calls | 384 | 384 |
| latency_median_s | 1.25 | 1.98 |
| latency_p90_s | 1.5 | 2.28 |
| latency_max_s | 2.71 | 3.24 |
| latency_sum_s | 506.2 | 794.1 |
| generated_tokens_median | 8.0 | 7.5 |
| hit_max_new_tokens | 0 | 0 |
| peak_vram_gib_reported | 8.911 | 7.014 |

성공이 갈린 질의 (M1만 성공 / M3만 성공 배치 수)

| 질의 | 언어 | 클래스 | 유형 | M1만 | M3만 | 둘 다 | 둘 다 실패 |
|---|---|---|---|---|---|---|---|
| q020 | ko | conveyor | Q1 | 4 | 0 | 0 | 0 |
| q095 | en | None | Q5 | 4 | 0 | 0 | 0 |
| q097 | en | None | Q5 | 4 | 0 | 0 | 0 |
| q062 | en | ibc_tank | Q3 | 3 | 0 | 0 | 1 |
| q011 | en | workbench | Q1 | 2 | 0 | 0 | 2 |
| q021 | en | conveyor | Q1 | 1 | 1 | 1 | 1 |
| q037 | en | storage_rack | Q1 | 0 | 2 | 0 | 2 |
| q047 | en | box_stack | Q1 | 2 | 0 | 2 | 0 |
| q094 | ko | None | Q5 | 2 | 0 | 0 | 2 |
| q002 | en | storage_cubby | Q1 | 1 | 0 | 0 | 3 |
| q004 | en | storage_cubby | Q3 | 1 | 0 | 0 | 3 |
| q013 | en | workbench | Q3 | 1 | 0 | 0 | 3 |
| q023 | en | conveyor | Q3 | 0 | 1 | 0 | 3 |
| q061 | en | ibc_tank | Q3 | 1 | 0 | 0 | 3 |
| q090 | ko | None | Q5 | 1 | 0 | 0 | 3 |
| q093 | en | None | Q5 | 0 | 1 | 3 | 0 |

