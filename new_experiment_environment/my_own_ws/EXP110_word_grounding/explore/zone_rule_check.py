#!/usr/bin/env python3
"""Check a proposed zone rule against the dev answer keys (exploratory; zones are a DRAFT, see NIGHT_REPORT §6.1).
Rule tested: no object CENTER inside Z02 minus (Z03 U Z04), and none inside Z01.
Zones: my_own_ws/2026-11-15_zone_draft/zones_draft.json (raw distorted pixels 1920x1080, drawn on P01; camera fixed).
Usage: python explore/zone_rule_check.py
"""
import glob, json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
z = json.loads((ROOT.parent / '2026-11-15_zone_draft' / 'zones_draft.json').read_text())
poly = {q['id']: np.array(q['polygon_xy'], np.float32) for q in z['zones']}


def inside(zid, p):
    return cv2.pointPolygonTest(poly[zid], p, False) >= 0


cfg = json.loads((ROOT / 'config.json').read_text())
for layout in cfg['splits']['dev']:
    f = sorted(glob.glob(str(ROOT / 'annotations' / 'pilot' / f'{layout}_*.json')))[0]
    objs = json.loads(Path(f).read_text())['objects']
    bad, n02 = [], 0
    for o in objs:
        b = o['box_xyxy']
        c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        n02 += inside('Z02', c)
        if (inside('Z02', c) and not inside('Z03', c) and not inside('Z04', c)) or inside('Z01', c):
            bad.append((o['label'], o.get('variant'), [round(v) for v in b]))
    print(layout, 'centers in Z02:', n02, '| violating centers:', bad)
