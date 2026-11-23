# EXP110 scores (dev, run run01)

thresholds: none (all saved predictions)

| cond | lang | n | success | recall | precision | invalid |
|---|---|---|---|---|---|---|
| C1 | en | 196 | 0.199 | 0.3065 | 0.4735 | 18 |
| C1 | ko | 192 | 0.1458 | 0.1723 | 0.2752 | 12 |
| C2 | en | 196 | 0.0 | 0.8992 | 0.0229 | 0 |
| C2 | ko | 192 | 0.0 | 0.7815 | 0.0194 | 0 |
| C3 | en | 196 | 0.0408 | 0.3931 | 0.0872 | 0 |
| C3 | ko | 192 | 0.0208 | 0.0336 | 0.0404 | 0 |
| C4 | en | 196 | 0.0663 | 0.0665 | 0.0535 | 0 |
| C4 | ko | 192 | 0.0833 | 0.0 | 0.0 | 0 |
| C5B | en | 196 | 0.0867 | 0.0907 | 0.1316 | 0 |
| C5B | ko | 192 | 0.0833 | 0.0 | None | 0 |
| C5L | en | 196 | 0.1684 | 0.129 | 0.2896 | 0 |
| C5L | ko | 192 | 0.0833 | 0.0 | None | 0 |
| C6-dino | en | 196 | 0.2551 | 0.381 | 0.5094 | 2 |
| C6-dino | ko | 192 | 0.125 | 0.1261 | 0.2564 | 0 |
| C6-florence | en | 196 | 0.2296 | 0.129 | 0.5517 | 2 |
| C6-florence | ko | 192 | 0.0833 | 0.0 | None | 0 |
| C6-pf | en | 196 | 0.1122 | 0.2601 | 0.2745 | 32 |
| C6-pf | ko | 192 | 0.1042 | 0.1029 | 0.1795 | 17 |
| C6-yoloe | en | 196 | 0.1224 | 0.3306 | 0.5077 | 0 |
| C6-yoloe | ko | 192 | 0.1094 | 0.0294 | 0.3043 | 2 |
| C6-yoloworld | en | 196 | 0.1173 | 0.0605 | 0.3333 | 1 |
| C6-yoloworld | ko | 192 | 0.0833 | 0.0 | 0.0 | 0 |

## C6 failure decomposition (English)

| cond | answer boxes: ok / selector_miss / detector_miss | failed queries | with >=1 detector_miss | with >=1 selector_miss | FP only | invalid |
|---|---|---|---|---|---|---|
| C6-dino | 189 / 97 / 210 | 146 | 112 | 64 | 3 | 2 |
| C6-florence | 64 / 0 / 432 | 151 | 147 | 0 | 4 | 2 |
| C6-pf | 129 / 249 / 118 | 174 | 81 | 115 | 14 | 32 |
| C6-yoloe | 164 / 25 / 307 | 172 | 145 | 22 | 11 | 0 |
| C6-yoloworld | 30 / 3 / 463 | 173 | 168 | 3 | 2 | 1 |

## C6 vs C1 (English, exploratory on dev)

- C6-pf - C1: diff -0.0867 CI95 [-0.1786, -0.0102], discordant 6/23, McNemar p=0.00232
- C6-dino - C1: diff 0.0561 CI95 [-0.0051, 0.1276], discordant 16/5, McNemar p=0.0266
- C6-yoloe - C1: diff -0.0765 CI95 [-0.1633, 0.0001], discordant 6/21, McNemar p=0.00592
- C6-yoloworld - C1: diff -0.0816 CI95 [-0.1531, -0.0204], discordant 1/17, McNemar p=0.00014
- C6-florence - C1: diff 0.0306 CI95 [-0.051, 0.1122], discordant 16/10, McNemar p=0.32694
