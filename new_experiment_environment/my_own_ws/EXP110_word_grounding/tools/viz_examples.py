#!/usr/bin/env python3
"""Render P01 dev examples (answer key vs C1 vs C6-dino) for six English queries.
Usage: python tools/viz_examples.py <out.png>"""
import json, sys, cv2, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools')); import score as S
Q = {q['id']: q for q in json.loads((ROOT/'queries.json').read_text())['queries']}
stem = 'P01_cheese_2026-11-14-181534'
img = cv2.imread(str(ROOT/'data/pilot'/f'{stem}.jpg'))
gt = S.load_gt(stem)['objects']
pred = {c: {r['qid']: r for r in json.loads((ROOT/'results/run01'/c/f'{stem}.json').read_text())['results']} for c in ['C1', 'C6-dino']}
qids = ['q043', 'q083', 'q064', 'q085', 'q025', 'q031']
rows = []
for qid in qids:
    q = Q[qid]; pos, _ = S.split_gt(gt, q); panels = []
    for c in ['C1', 'C6-dino']:
        r = pred[c][qid]; im = img.copy()
        if c == 'C6-dino':
            for cd in r['candidates']:
                b = [int(v) for v in cd['box']]; cv2.rectangle(im, b[:2], b[2:], (160, 160, 160), 2)
        for b in pos:
            b = [int(v) for v in b]; cv2.rectangle(im, b[:2], b[2:], (0, 220, 0), 5)
        P = [p['box'] for p in r['preds']] if r.get('valid', True) else []
        for b in P:
            ok = any(S.iou(b, g) >= 0.5 for g in pos)
            b = [int(v) for v in b]; cv2.rectangle(im, b[:2], b[2:], (255, 120, 0) if ok else (0, 0, 255), 4)
        s = S.score_one(P, r.get('valid', True), gt, q)
        txt = f'{c} | "{q["text"]}" | GT {s["n_gt"]}  found {s["tp"]}  wrong {s["fp"]}  -> {"SUCCESS" if s["success"] else "FAIL"}'
        im = cv2.resize(im, (960, 540))
        cv2.rectangle(im, (0, 0), (960, 34), (0, 0, 0), -1)
        cv2.putText(im, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        panels.append(im)
    rows.append(np.hstack(panels))
leg = np.zeros((40, 1920, 3), np.uint8)
cv2.putText(leg, 'green = answer key   blue = correct prediction   red = wrong prediction   gray = DINO candidates shown to Qwen (C6 only)', (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
out = np.vstack([leg] + rows)
cv2.imwrite(sys.argv[1], out); print(out.shape)
