# EXP110-X1 three metrics (exploratory, dev: P01, P02, P03, P04)

C2 rows here use ALL saved predictions (no threshold): compare C1 and C6 only.

strict = IoU>=0.5 all & only | cu = center 1:1 + count | tr = first box on a correct object (Q5: no box)

## en

| run | cond | group | n | strict | cu | tr |
|---|---|---|---|---|---|---|
| run02 | C1 | absent | 16 | 1.0 | 1.0 | 1.0 |
| run02 | C1 | all | 196 | 0.199 | 0.199 | 0.4286 |
| run02 | C1 | multi | 88 | 0.0568 | 0.0568 | 0.4545 |
| run02 | C1 | task | 24 | 0.0 | 0.0 | 0.0 |
| run02 | C1 | unique | 68 | 0.2647 | 0.2647 | 0.4118 |
| run02 | C2 | absent | 16 | 0.0 | 0.0 | 0.0 |
| run02 | C2 | all | 196 | 0.0 | 0.0 | 0.1939 |
| run02 | C2 | multi | 88 | 0.0 | 0.0 | 0.2273 |
| run02 | C2 | task | 24 | 0.0 | 0.0 | 0.0 |
| run02 | C2 | unique | 68 | 0.0 | 0.0 | 0.2647 |
| run02 | C6-dino | absent | 16 | 0.9375 | 0.9375 | 0.9375 |
| run02 | C6-dino | all | 196 | 0.2551 | 0.2755 | 0.5153 |
| run02 | C6-dino | multi | 88 | 0.0909 | 0.125 | 0.5568 |
| run02 | C6-dino | task | 24 | 0.0 | 0.0 | 0.0 |
| run02 | C6-dino | unique | 68 | 0.3971 | 0.4118 | 0.5441 |
| run02 | C6-dino_s05k30 | absent | 16 | 0.8125 | 0.8125 | 0.8125 |
| run02 | C6-dino_s05k30 | all | 196 | 0.199 | 0.2194 | 0.4541 |
| run02 | C6-dino_s05k30 | multi | 88 | 0.1136 | 0.1364 | 0.5341 |
| run02 | C6-dino_s05k30 | task | 24 | 0.0 | 0.0 | 0.0 |
| run02 | C6-dino_s05k30 | unique | 68 | 0.2353 | 0.2647 | 0.4265 |
| run02 | C6-dino_s05k60 | absent | 16 | 0.5625 | 0.5625 | 0.5625 |
| run02 | C6-dino_s05k60 | all | 196 | 0.1276 | 0.1684 | 0.3673 |
| run02 | C6-dino_s05k60 | multi | 88 | 0.0568 | 0.1136 | 0.3977 |
| run02 | C6-dino_s05k60 | task | 24 | 0.0417 | 0.0417 | 0.0417 |
| run02 | C6-dino_s05k60 | unique | 68 | 0.1471 | 0.1912 | 0.3971 |
| run02 | C6-dino_s10k30 | absent | 16 | 1.0 | 1.0 | 1.0 |
| run02 | C6-dino_s10k30 | all | 196 | 0.2551 | 0.2704 | 0.4898 |
| run02 | C6-dino_s10k30 | multi | 88 | 0.1136 | 0.1364 | 0.5341 |
| run02 | C6-dino_s10k30 | task | 24 | 0.0 | 0.0 | 0.0 |
| run02 | C6-dino_s10k30 | unique | 68 | 0.3529 | 0.3676 | 0.4853 |
| x6_M3_dev | C1 | absent | 16 | 1.0 | 1.0 | 1.0 |
| x6_M3_dev | C1 | all | 196 | 0.1939 | 0.1939 | 0.4082 |
| x6_M3_dev | C1 | multi | 88 | 0.1136 | 0.1136 | 0.5 |
| x6_M3_dev | C1 | task | 24 | 0.0 | 0.0 | 0.0 |
| x6_M3_dev | C1 | unique | 68 | 0.1765 | 0.1765 | 0.2941 |
| x6_M3_dev | C6-dino | absent | 16 | 0.5 | 0.5 | 0.5 |
| x6_M3_dev | C6-dino | all | 196 | 0.1786 | 0.1837 | 0.4592 |
| x6_M3_dev | C6-dino | multi | 88 | 0.1136 | 0.1136 | 0.5909 |
| x6_M3_dev | C6-dino | task | 24 | 0.0 | 0.0417 | 0.0833 |
| x6_M3_dev | C6-dino | unique | 68 | 0.25 | 0.25 | 0.4118 |

## ko

| run | cond | group | n | strict | cu | tr |
|---|---|---|---|---|---|---|
| run02 | C1 | absent | 16 | 0.75 | 0.75 | 0.75 |
| run02 | C1 | all | 192 | 0.1458 | 0.1458 | 0.276 |
| run02 | C1 | multi | 80 | 0.0625 | 0.0625 | 0.2625 |
| run02 | C1 | task | 24 | 0.0 | 0.0 | 0.0 |
| run02 | C1 | unique | 72 | 0.1528 | 0.1528 | 0.2778 |
| run02 | C2 | absent | 16 | 0.0 | 0.0 | 0.0 |
| run02 | C2 | all | 192 | 0.0 | 0.0 | 0.0885 |
| run02 | C2 | multi | 80 | 0.0 | 0.0 | 0.1125 |
| run02 | C2 | task | 24 | 0.0 | 0.0 | 0.0 |
| run02 | C2 | unique | 72 | 0.0 | 0.0 | 0.1111 |
| run02 | C6-dino | absent | 16 | 0.6875 | 0.6875 | 0.6875 |
| run02 | C6-dino | all | 192 | 0.125 | 0.151 | 0.3438 |
| run02 | C6-dino | multi | 80 | 0.0125 | 0.0125 | 0.35 |
| run02 | C6-dino | task | 24 | 0.0 | 0.0 | 0.0 |
| run02 | C6-dino | unique | 72 | 0.1667 | 0.2361 | 0.375 |
| run02 | C6-dino_s05k30 | absent | 16 | 0.5625 | 0.5625 | 0.5625 |
| run02 | C6-dino_s05k30 | all | 192 | 0.2135 | 0.2396 | 0.4323 |
| run02 | C6-dino_s05k30 | multi | 80 | 0.0625 | 0.0625 | 0.4 |
| run02 | C6-dino_s05k30 | task | 24 | 0.0 | 0.0 | 0.0417 |
| run02 | C6-dino_s05k30 | unique | 72 | 0.375 | 0.4444 | 0.5694 |
| run02 | C6-dino_s05k60 | absent | 16 | 0.4375 | 0.4375 | 0.4375 |
| run02 | C6-dino_s05k60 | all | 192 | 0.1667 | 0.1875 | 0.3438 |
| run02 | C6-dino_s05k60 | multi | 80 | 0.0375 | 0.0375 | 0.325 |
| run02 | C6-dino_s05k60 | task | 24 | 0.0 | 0.0417 | 0.0417 |
| run02 | C6-dino_s05k60 | unique | 72 | 0.3056 | 0.3472 | 0.4444 |
| run02 | C6-dino_s10k30 | absent | 16 | 0.625 | 0.625 | 0.625 |
| run02 | C6-dino_s10k30 | all | 192 | 0.125 | 0.151 | 0.3438 |
| run02 | C6-dino_s10k30 | multi | 80 | 0.0125 | 0.0125 | 0.35 |
| run02 | C6-dino_s10k30 | task | 24 | 0.0 | 0.0 | 0.0 |
| run02 | C6-dino_s10k30 | unique | 72 | 0.1806 | 0.25 | 0.3889 |
| x6_M3_dev | C1 | absent | 16 | 1.0 | 1.0 | 1.0 |
| x6_M3_dev | C1 | all | 192 | 0.1458 | 0.1458 | 0.2865 |
| x6_M3_dev | C1 | multi | 80 | 0.0125 | 0.0125 | 0.25 |
| x6_M3_dev | C1 | task | 24 | 0.0 | 0.0 | 0.0 |
| x6_M3_dev | C1 | unique | 72 | 0.1528 | 0.1528 | 0.2639 |
| x6_M3_dev | C6-dino | absent | 16 | 0.5 | 0.5 | 0.5 |
| x6_M3_dev | C6-dino | all | 192 | 0.0885 | 0.1146 | 0.3073 |
| x6_M3_dev | C6-dino | multi | 80 | 0.0125 | 0.0125 | 0.35 |
| x6_M3_dev | C6-dino | task | 24 | 0.0 | 0.0 | 0.0 |
| x6_M3_dev | C6-dino | unique | 72 | 0.1111 | 0.1806 | 0.3194 |
