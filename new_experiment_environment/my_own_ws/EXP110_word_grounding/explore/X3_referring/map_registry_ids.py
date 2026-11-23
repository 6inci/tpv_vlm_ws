#!/usr/bin/env python3
"""EXP110-X3 helper: map fixed registry ids (#1-#29) to one answer-key box per dev image.

Registry centers come from candidates_draft.json (eyeballed, screenshot pixels; camera frame = screenshot - (1, 38)).
Rule: among answer-key boxes with the registry class (and colour variant for drums), take the one whose center is
nearest to the registry center. Distance (raw px) and whether the registry center lies inside the box are reported,
so eyeballing errors stay visible. Movable ids (9, 13 small pallets) and unqueried ids (21 robot, 24 board) are skipped.
Usage: python explore/X3_referring/map_registry_ids.py > explore/X3_referring/registry_id_to_box_dev.json
"""
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import score as S  # noqa: E402

REG_DIR = Path(S.CFG['registry']).parent
cands = json.loads((REG_DIR / 'candidates_draft.json').read_text())
reg = json.loads(Path(S.CFG['registry']).read_text())
OFFSET = (1, 38)  # screenshot -> camera frame, from candidates_draft.json coordinate_system
id2cls, id2var = {}, {}
for c in reg['classes']:
    for i in c['ids']:
        if isinstance(i, int) and c.get('mobility') == 'fixed':
            id2cls[i] = c['key']
    for k, v in (c.get('attributes') or {}).items() if isinstance(c.get('attributes'), dict) else []:
        id2var[int(k)] = v

out = {'method': __doc__.split('\n')[3].strip(), 'offset_screenshot_to_camera': OFFSET, 'images': {}}
for layout in S.CFG['splits']['dev']:
    stem = sorted((ROOT / 'annotations' / 'pilot').glob(f'{layout}_*.json'))[0].stem
    gt = S.load_gt(stem)['objects']
    m = {}
    for c in cands['candidates']:
        i = c['id']
        if i not in id2cls:
            continue
        cx, cy = c['center_px'][0] - OFFSET[0], c['center_px'][1] - OFFSET[1]
        pool = [o for o in gt if o['label'] == id2cls[i] and (i not in id2var or o.get('variant') == id2var[i])]
        if not pool:
            m[i] = {'class': id2cls[i], 'box': None, 'reason': 'no answer box of this class'}
            continue
        best = min(pool, key=lambda o: math.dist((cx, cy), S.center(o['box_xyxy'])) if hasattr(S, 'center') else
                   math.dist((cx, cy), ((o['box_xyxy'][0] + o['box_xyxy'][2]) / 2, (o['box_xyxy'][1] + o['box_xyxy'][3]) / 2)))
        b = best['box_xyxy']
        d = math.dist((cx, cy), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        m[i] = {'class': id2cls[i], 'variant': best.get('variant'), 'box': [round(v, 1) for v in b],
                'registry_center_camera_px': [cx, cy], 'center_dist_px': round(d, 1), 'center_inside_box': S.center_in([cx, cy, cx, cy], b),
                'n_same_class_boxes': len(pool)}
    out['images'][layout] = m
print(json.dumps(out, indent=1, ensure_ascii=False))
