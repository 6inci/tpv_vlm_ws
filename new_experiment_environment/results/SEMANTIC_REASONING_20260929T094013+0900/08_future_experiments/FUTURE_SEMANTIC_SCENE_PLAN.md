# Future Semantic Scene Plan

| Scene | Objects | Change | Held constant | Shortcut tested | Instructions | Required annotation |
|---|---|---|---|---|---|---|
| A baseline | truck, racks, pallets, workbench, robot arms, conveyor, cart, drums, mobile robot | none | camera/lighting/config | baseline | current TYPE A–E set | reviewed objects, relations, regions, masks, targets |
| B rearranged | same as A | exchange Loading/Storage/Work/Material positions | object identities, camera | absolute-position shortcut | same TYPE C/D wording | new boxes, relations, region polygons, valid targets |
| C monochrome | same geometry | recolor or use white printed objects | layout/camera | color shortcut | same TYPE A–D wording | identity and region review |
| D partial cues | truck without pallet; manipulator without conveyor | remove one semantic cue | camera/remaining objects | evidence reduction | Loading/Work Area requests | presence/absence and expected abstention |
| E ambiguous | repeated pallets/racks/drums | add equally plausible instances | camera/classes | relation disambiguation | nearest/between instructions | instance IDs, relations, intended referent |
