# EXP110 scores (dev, run x6_M3_dev)

thresholds: none (all saved predictions)

| cond | lang | n | success | recall | precision | invalid |
|---|---|---|---|---|---|---|
| C1 | en | 196 | 0.1939 | 0.4355 | 0.6945 | 3 |
| C1 | ko | 192 | 0.1458 | 0.145 | 0.5897 | 0 |
| C6-dino | en | 196 | 0.1786 | 0.4113 | 0.4024 | 0 |
| C6-dino | ko | 192 | 0.0885 | 0.1387 | 0.3188 | 0 |

## C6 failure decomposition (English)

| cond | answer boxes: ok / selector_miss / detector_miss | failed queries | with >=1 detector_miss | with >=1 selector_miss | FP only | invalid |
|---|---|---|---|---|---|---|
| C6-dino | 204 / 82 / 210 | 161 | 112 | 59 | 14 | 0 |

## C6 vs C1 (English, dev; dev = exploratory, test = registered comparison DESIGN §8/§12)

- C6-dino - C1: diff -0.0153 CI95 [-0.1122, 0.0816], discordant 13/16, McNemar p=0.71107
