# EXP110 scores (dev, run run02)

thresholds: none (all saved predictions)

| cond | lang | n | success | recall | precision | invalid |
|---|---|---|---|---|---|---|
| C1 | en | 196 | 0.199 | 0.3528 | 0.4834 | 16 |
| C1 | ko | 192 | 0.1458 | 0.1723 | 0.2752 | 12 |
| C2 | en | 196 | 0.0 | 0.8992 | 0.0229 | 0 |
| C2 | ko | 192 | 0.0 | 0.7815 | 0.0194 | 0 |
| C6-dino_s05k30 | en | 196 | 0.199 | 0.3569 | 0.3925 | 8 |
| C6-dino_s05k30 | ko | 192 | 0.2135 | 0.2227 | 0.3498 | 0 |
| C6-dino_s05k60 | en | 196 | 0.1276 | 0.2298 | 0.2522 | 25 |
| C6-dino_s05k60 | ko | 192 | 0.1667 | 0.1786 | 0.2038 | 5 |
| C6-dino_s10k30 | en | 196 | 0.2551 | 0.381 | 0.4489 | 6 |
| C6-dino_s10k30 | ko | 192 | 0.125 | 0.1408 | 0.2597 | 0 |
| C6-dino | en | 196 | 0.2551 | 0.381 | 0.5094 | 2 |
| C6-dino | ko | 192 | 0.125 | 0.1261 | 0.2564 | 0 |

## C6 failure decomposition (English)

| cond | answer boxes: ok / selector_miss / detector_miss | failed queries | with >=1 detector_miss | with >=1 selector_miss | FP only | invalid |
|---|---|---|---|---|---|---|
| C6-dino_s05k30 | 177 / 208 / 111 | 157 | 72 | 105 | 9 | 8 |
| C6-dino_s05k60 | 114 / 307 / 75 | 171 | 53 | 128 | 10 | 25 |
| C6-dino_s10k30 | 189 / 151 / 156 | 146 | 80 | 92 | 5 | 6 |
| C6-dino | 189 / 97 / 210 | 146 | 112 | 64 | 3 | 2 |

## C6 vs C1 (English, exploratory on dev)

- C6-dino - C1: diff 0.0561 CI95 [-0.0051, 0.1276], discordant 16/5, McNemar p=0.0266
- C6-dino_s10k30 - C1: diff 0.0561 CI95 [0.0, 0.1173], discordant 17/6, McNemar p=0.03469
- C6-dino_s05k30 - C1: diff 0.0 CI95 [-0.0714, 0.0714], discordant 12/12, McNemar p=1.0
- C6-dino_s05k60 - C1: diff -0.0714 CI95 [-0.1582, 0.0102], discordant 8/22, McNemar p=0.01612
