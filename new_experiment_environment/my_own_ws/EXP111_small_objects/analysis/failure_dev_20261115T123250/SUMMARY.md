# EXP111 step 0: small-object failure causes (dev P01-P04, EXP110 run02 outputs)

Unit = (image, query, answer instance). "hit" = IoU>=0.5 or box center inside the instance.

## block

- answer size: width 16-24 px, height 14-25 px = 0.5-0.8 Qwen tokens wide (1 token = 32 px)

| lang | n | C6: not in DINO top-100 | filtered out | candidate, not chosen | chosen | invalid | C1 hit (IoU / center) | C1 first box landed on |
|---|---|---|---|---|---|---|---|---|
| en | 24 | 8 | 13 | 3 | 0 | 0 | 0 / 0 | box_stack 10, INVALID 4, drum(blue) 4, fire_extinguisher 4 |
| ko | 24 | 7 | 17 | 0 | 0 | 0 | 0 / 0 | NO BOX 20, drum(blue) 4 |

| query | n | DINO rank of the instance (IoU or center hit; "-" = not in top-100) | C6 chose (landed on) |
|---|---|---|---|
| q028 블록 | 8 | -, 42, 48, 48, 61, -, 62, - | pallet 6, storage_rack 4, drum(blue) 4 |
| q029 block | 8 | -, 30, 45, 45, 29, -, 28, - | box_stack 8 |
| q030 파란 블록 | 4 | -, 53, -, - | nothing 4 |
| q031 blue block | 4 | 84, 17, -, - | drum(blue) 4 |
| q032 빨간 블록 | 4 | 43, 42, 62, 66 | drum(dark_green) 1 |
| q033 red block | 4 | 41, 75, 44, 45 | fire_extinguisher 4 |
| q034 큐브 | 8 | -, 45, 80, 80, 57, 57, 59, 59 | traffic_cone 2, storage_rack 2, box_stack 2 |
| q035 cube | 8 | -, 27, 42, 42, 22, -, 23, - | box_stack 8 |

## small_pallet

- answer size: width 46-54 px, height 38-57 px = 1.4-1.7 Qwen tokens wide (1 token = 32 px)

| lang | n | C6: not in DINO top-100 | filtered out | candidate, not chosen | chosen | invalid | C1 hit (IoU / center) | C1 first box landed on |
|---|---|---|---|---|---|---|---|---|
| en | 16 | 0 | 12 | 4 | 0 | 0 | 0 / 0 | pallet 10, drum(dark_green) 4, INVALID 2 |
| ko | 16 | 6 | 10 | 0 | 0 | 0 | 0 / 0 | pallet 6, INVALID 4, drum(dark_green) 4, NO BOX 2 |

| query | n | DINO rank of the instance (IoU or center hit; "-" = not in top-100) | C6 chose (landed on) |
|---|---|---|---|
| q024 소형 팔레트 | 8 | 58, -, 69, 88, 83, -, 85, - | storage_rack 4, box_stack 2, nothing 2 |
| q025 small pallet | 8 | 48, 30, 81, 4, 24, 17, 28, 16 | drum(blue) 6, pallet 4, cart 4 |
| q026 작은 회색 팔레트 | 8 | 63, -, 79, 66, 83, -, 83, - | box_stack 6, nothing 2 |
| q027 small gray pallet | 8 | 76, 17, 89, 4, 37, 9, 39, 11 | cart 8 |

