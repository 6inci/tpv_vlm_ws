#!/usr/bin/env python3
"""Explain each panel of viz_examples.py (P01, six English queries) from saved run01 outputs (no model run).
Per query: C1 validity/parse error/raw text and what each predicted box overlaps in the answer key;
C6-dino candidate filter and chosen boxes; for every answer box, the best-ranked DINO (C2) box with IoU>=0.5.
Usage: python tools/explain_examples.py > <out>.txt"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools')); import score as S
Q = {q['id']: q for q in json.loads((ROOT/'queries.json').read_text())['queries']}
stem = 'P01_cheese_2026-11-14-181534'
gt = S.load_gt(stem)['objects']
raw = {c: json.loads((ROOT/'results/run01'/c/f'{stem}.json').read_text()) for c in ['C1', 'C2', 'C6-dino']}
res = {c: {r['qid']: r for r in d['results']} for c, d in raw.items()}
W, H = raw['C1']['image_size']
print('image', stem, '| boxes in raw (distorted) pixels, 1920x1080 | IoU threshold', S.IOU_T)
print('C6-dino params', raw['C6-dino'].get('params'))


def hit(b):  # which answer-key object this predicted box overlaps most
    m = max((S.iou(b, o['box_xyxy']), o['label'], o.get('variant')) for o in gt)
    return f"{m[1]}{':' + m[2] if m[2] else ''} (IoU {m[0]:.2f})"


for qid in ['q043', 'q083', 'q064', 'q085', 'q025', 'q031']:
    q = Q[qid]; pos, _ = S.split_gt(gt, q)
    print(f'\n== {qid} "{q["text"]}" | answer boxes {len(pos)}')
    r = res['C1'][qid]
    print(f"C1 valid={r['valid']} error={r['error']} hit_max_new_tokens={r.get('hit_max_new_tokens')}")
    print('   raw:', (r['raw'] or '')[:160].replace('\n', ' '))
    print('   parsed boxes ->', [hit(p['box']) for p in r['preds']] if r['valid'] else 'not scored (invalid)')
    if not r['valid']:  # what the raw text pointed at, for diagnosis only (still scored as failure)
        m = re.search(r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', r['raw'] or '')
        if m:  # 0-1000 normalised -> raw pixels (same conversion as run_models.parse_c1)
            n = [int(v) for v in m.groups()]
            b = [n[0] * W / 1000, n[1] * H / 1000, n[2] * W / 1000, n[3] * H / 1000]
            print('   first box in raw text (diagnosis only) ->', hit(b))
        else:
            print('   first box in raw text: none found')
    r6 = res['C6-dino'][qid]
    print(f"C6-dino candidates {len(r6['candidates'])}, raw {r6.get('raw')} -> chosen", [hit(p['box']) for p in r6['preds']])
    P = sorted(res['C2'][qid]['preds'], key=lambda p: -p['score'])
    cut = P[14]['score'] if len(P) > 14 else None
    print(f"DINO (C2) saved boxes {len(P)}, 15th score {cut}")
    for g in pos:
        h = [(i + 1, p['score']) for i, p in enumerate(P) if S.iou(p['box'], g) >= S.IOU_T]
        size = f'{g[2] - g[0]:.0f}x{g[3] - g[1]:.0f}px'
        print(f'   answer {[round(v) for v in g]} {size}:', f'best DINO rank {h[0][0]} score {h[0][1]:.3f}' if h else 'no DINO box with IoU>=0.5')
