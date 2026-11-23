# X3 summary (run_20261115T092412, dev P01-P04, recomputed from rows.json)

| cond | lang | n | invalid | first box on referent (primary) | (lenient) | only referent boxes (lenient) |
|---|---|---|---|---|---|---|
| C1 | en | 48 | 5/48 | 10/48 | 13/48 | 13/48 |
| C1 | ko | 48 | 2/48 | 4/48 | 4/48 | 3/48 |
| C6-dino | en | 48 | 0/48 | 7/48 | 9/48 | 9/48 |
| C6-dino | ko | 48 | 0/48 | 7/48 | 10/48 | 10/48 |

Per query, first box on referent (lenient), images P01-P04 as 1/0, "x" = invalid:

| rid | text (en) | C1 en | C6 en | C1 ko | C6 ko |
|---|---|---|---|---|---|
| r01 | the shelf next to the truck | 1111 | 1000 | 0000 | 1111 |
| r02 | the left one of the two shelves at the top | 11x0 | 0000 | 1110 | 0000 |
| r03 | the shelf in the bottom-left corner | 1111 | 1001 | 0000 | 1100 |
| r04 | the cart next to the IBC tote | 0000 | 0000 | 0000 | 0000 |
| r05 | the left cart | 0xxx | 0001 | 00x0 | 0000 |
| r06 | the upper charging station | 0000 | 0000 | 0100 | 0000 |
| r07 | the robot arm next to the conveyor belt | 0000 | 0100 | 0000 | 1111 |
| r08 | the empty pallet | 1011 | 0000 | 0000 | 0000 |
| r09 | the drum next to the IBC tote | 0000 | 1111 | 0000 | 0000 |
| r10 | the small pallet with blocks on it | 0x00 | 0000 | x000 | 0000 |
| r11 | the empty small pallet | 0000 | 0000 | 0000 | 0000 |
| r12 | the shelf next to the forklift | 0000 | 0000 | 0000 | 0000 |

C6-dino failures with a referent (lenient), where they happen:

| lang | failures | referent not in DINO top-100 | in DINO but filtered out of candidates | in candidates, Qwen chose otherwise / invalid |
|---|---|---|---|---|
| en | 35 | 6 | 8 | 21 |
| ko | 34 | 7 | 25 | 2 |
