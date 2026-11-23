# EXP110 scores (test, run run03)

thresholds: {'C2': 0.5}

| cond | lang | n | success | recall | precision | invalid |
|---|---|---|---|---|---|---|
| C1 | en | 294 | 0.2075 | 0.414 | 0.4768 | 23 |
| C1 | ko | 288 | 0.1562 | 0.1541 | 0.3047 | 17 |
| C2 | en | 294 | 0.102 | 0.0565 | 0.2781 | 0 |
| C2 | ko | 288 | 0.0833 | 0.0 | None | 0 |
| C6-dino | en | 294 | 0.2721 | 0.4005 | 0.5458 | 0 |
| C6-dino | ko | 288 | 0.1319 | 0.1106 | 0.2446 | 0 |

## C6 failure decomposition (English)

| cond | answer boxes: ok / selector_miss / detector_miss | failed queries | with >=1 detector_miss | with >=1 selector_miss | FP only | invalid |
|---|---|---|---|---|---|---|
| C6-dino | 298 / 132 / 314 | 214 | 167 | 91 | 2 | 0 |

## C6 vs C1 (English, test; dev = exploratory, test = registered comparison DESIGN §8/§12)

- C6-dino - C1: diff 0.0646 CI95 [-0.0034, 0.1395], discordant 28/9, McNemar p=0.00256
