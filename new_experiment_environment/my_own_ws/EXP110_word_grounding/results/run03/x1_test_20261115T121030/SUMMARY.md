# EXP110-X1 three metrics (exploratory, test: P05, P06, P07, P08, P09, P10)

C2 rows here use ALL saved predictions (no threshold): compare C1 and C6 only.

strict = IoU>=0.5 all & only | cu = center 1:1 + count | tr = first box on a correct object (Q5: no box)

## en

| run | cond | group | n | strict | cu | tr |
|---|---|---|---|---|---|---|
| run03 | C1 | absent | 24 | 1.0 | 1.0 | 1.0 |
| run03 | C1 | all | 294 | 0.2075 | 0.2075 | 0.4354 |
| run03 | C1 | multi | 132 | 0.0606 | 0.0606 | 0.447 |
| run03 | C1 | task | 36 | 0.0 | 0.0 | 0.0 |
| run03 | C1 | unique | 102 | 0.2843 | 0.2843 | 0.4412 |
| run03 | C2 | absent | 24 | 0.0 | 0.0 | 0.0 |
| run03 | C2 | all | 294 | 0.0 | 0.0 | 0.1837 |
| run03 | C2 | multi | 132 | 0.0 | 0.0 | 0.2273 |
| run03 | C2 | task | 36 | 0.0 | 0.0 | 0.0 |
| run03 | C2 | unique | 102 | 0.0 | 0.0 | 0.2353 |
| run03 | C6-dino | absent | 24 | 1.0 | 1.0 | 1.0 |
| run03 | C6-dino | all | 294 | 0.2721 | 0.2789 | 0.5102 |
| run03 | C6-dino | multi | 132 | 0.1212 | 0.1364 | 0.553 |
| run03 | C6-dino | task | 36 | 0.0 | 0.0 | 0.0278 |
| run03 | C6-dino | unique | 102 | 0.3922 | 0.3922 | 0.5098 |

## ko

| run | cond | group | n | strict | cu | tr |
|---|---|---|---|---|---|---|
| run03 | C1 | absent | 24 | 0.8333 | 0.8333 | 0.8333 |
| run03 | C1 | all | 288 | 0.1562 | 0.1562 | 0.2847 |
| run03 | C1 | multi | 120 | 0.05 | 0.05 | 0.2167 |
| run03 | C1 | task | 36 | 0.0 | 0.0 | 0.0 |
| run03 | C1 | unique | 108 | 0.1759 | 0.1759 | 0.3333 |
| run03 | C2 | absent | 24 | 0.0 | 0.0 | 0.0 |
| run03 | C2 | all | 288 | 0.0 | 0.0 | 0.0833 |
| run03 | C2 | multi | 120 | 0.0 | 0.0 | 0.1083 |
| run03 | C2 | task | 36 | 0.0 | 0.0 | 0.0 |
| run03 | C2 | unique | 108 | 0.0 | 0.0 | 0.1019 |
| run03 | C6-dino | absent | 24 | 0.7083 | 0.7083 | 0.7083 |
| run03 | C6-dino | all | 288 | 0.1319 | 0.1632 | 0.3264 |
| run03 | C6-dino | multi | 120 | 0.025 | 0.025 | 0.3 |
| run03 | C6-dino | task | 36 | 0.0 | 0.0 | 0.0 |
| run03 | C6-dino | unique | 108 | 0.1667 | 0.25 | 0.3796 |
