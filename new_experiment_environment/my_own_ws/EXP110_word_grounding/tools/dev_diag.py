#!/usr/bin/env python3
"""EXP110 dev diagnostics from saved run01 predictions (no model run): C1 localisation/count/invalid causes,
C2 candidate coverage vs C6 filter (min_score, top_k), C6-dino per-class ok/selector_miss/detector_miss,
small-object instance recall, latency/VRAM. Usage: python tools/dev_diag.py > <out>.txt"""
import json, sys, glob, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import score as S
Q = {q['id']: q for q in json.loads((ROOT/'queries.json').read_text())['queries']}
DEV = S.CFG['splits']['dev']; W, H = 1920, 1080
def load(cond, lay):
    return json.loads(sorted((ROOT/'results/run01'/cond).glob(f'{lay}_*.json'))[0].read_text())
def gt(d): return S.load_gt(Path(d['image']).stem)['objects']

print('== 1. C1 (Qwen alone) localisation, en, per GT positive: best IoU of any pred ==')
ious, cen, inval, hitmax = [], 0, collections.Counter(), 0
under = collections.Counter()
for lay in DEV:
    d = load('C1', lay); g = gt(d)
    for r in d['results']:
        q = Q[r['qid']]
        if q['lang'] != 'en' or q['type'] == 'Q5': continue
        if not r['valid']:
            inval[(r['hit_max_new_tokens'], (r['error'] or '')[:40])] += 1; continue
        pos, _ = S.split_gt(g, q); P = [p['box'] for p in r['preds']]
        under[('n_pred<n_gt' if len(P) < len(pos) else 'n_pred>n_gt' if len(P) > len(pos) else 'equal')] += 1
        for b in pos:
            m = max([S.iou(p, b) for p in P] or [0]); ious.append(m)
            cen += any(S.center_in(p, b) for p in P)
ious = np.array(ious)
print(f'GT positives {len(ious)}; IoU>=0.5 {np.mean(ious>=.5):.3f}; IoU 0.3-0.5 {np.mean((ious>=.3)&(ious<.5)):.3f}; 0<IoU<0.3 {np.mean((ious>0)&(ious<.3)):.3f}; no overlap {np.mean(ious==0):.3f}; center-hit {cen/len(ious):.3f}')
print('count vs GT (valid en Q1-3 queries):', dict(under))
print('invalid (hit_max, error):', dict(inval))

print('\n== 2. C2 (DINO) candidate coverage of en positives under C6 filter settings ==')
def cands(preds, ms, k, area=.25, nms=.7):
    c = [p for p in preds if (p['score'] or 0) >= ms and (p['box'][2]-p['box'][0])*(p['box'][3]-p['box'][1]) <= area*W*H]
    c.sort(key=lambda p: -p['score']); keep = []
    for p in c:
        if all(S.iou(p['box'], x['box']) < nms for x in keep): keep.append(p)
        if len(keep) >= k: break
    return [p['box'] for p in keep]
data2 = [(load('C2', lay), lay) for lay in DEV]
for ms in (0.01, 0.05, 0.10):
    for k in (15, 30, 60):
        cov = n = 0
        for d, lay in data2:
            g = gt(d)
            for r in d['results']:
                q = Q[r['qid']]
                if q['lang'] != 'en' or q['type'] == 'Q5': continue
                pos, _ = S.split_gt(g, q); C = cands(r['preds'], ms, k)
                cov += sum(any(S.iou(c, b) >= .5 for c in C) for b in pos); n += len(pos)
        print(f'min_score {ms:.2f} top_k {k:2d}: coverage {cov/n:.3f} ({cov}/{n})')
# no filter at all (all saved boxes, area unrestricted)
cov = n = 0
for d, lay in data2:
    g = gt(d)
    for r in d['results']:
        q = Q[r['qid']]
        if q['lang'] != 'en' or q['type'] == 'Q5': continue
        pos, _ = S.split_gt(g, q)
        cov += sum(any(S.iou(p['box'], b) >= .5 for p in r['preds']) for b in pos); n += len(pos)
print(f'all saved C2 boxes (<=100, score>=0.01): coverage {cov/n:.3f}')

print('\n== 3. C6-dino per class (en, all query types): positives ok / selector_miss / detector_miss ==')
pc = collections.defaultdict(collections.Counter); sel = collections.Counter()
for lay in DEV:
    d = load('C6-dino', lay); g = gt(d)
    for r in d['results']:
        q = Q[r['qid']]
        if q['lang'] != 'en' or q['type'] == 'Q5': continue
        b = S.c6_breakdown(r, d.get('shared_candidates'), g, q)
        for k2, v in b.items(): pc[q['source_class']][k2] += v
        pos, _ = S.split_gt(g, q); ch = [p['box'] for p in r['preds']]
        wrong = sum(not any(S.iou(c, x) >= .5 for x in pos) for c in ch)
        sel['chosen_total'] += len(ch); sel['chosen_wrong'] += wrong
        sel['empty_choice'] += (len(ch) == 0 and len(pos) > 0)
for c in sorted(pc): print(f"{c:18s} ok {pc[c]['ok']:3d}  sel_miss {pc[c]['selector_miss']:3d}  det_miss {pc[c]['detector_miss']:3d}")
print('selector:', dict(sel))

print('\n== 4. small/task objects instance recall (en, Q1-Q3) ==')
for cond in ['C1', 'C2', 'C6-dino', 'C6-florence', 'C6-yoloe']:
    out = {}
    for cls in ['block', 'small_pallet', 'traffic_cone', 'fire_extinguisher']:
        tp = n = 0
        for lay in DEV:
            d = load(cond, lay); g = gt(d)
            for r in d['results']:
                q = Q[r['qid']]
                if q['lang'] != 'en' or q['source_class'] != cls or q['type'] == 'Q5': continue
                if not r.get('valid', True): n += len(S.split_gt(g, q)[0]); continue
                s = S.score_one([p['box'] for p in r['preds']], True, g, q); tp += s['tp']; n += s['n_gt']
        out[cls] = f'{tp}/{n}'
    print(cond, out)

print('\n== 5. latency / VRAM per condition (dev) ==')
for cond in ['C1', 'C2', 'C5L', 'C6-dino', 'C6-florence']:
    L = []; v = []
    for lay in DEV:
        d = load(cond, lay); v.append(d.get('peak_vram_gib')); L.append(d['wall_s'] / d['n_queries'])
    print(cond, f'{np.mean(L):.2f} s/query (wall incl. detector? see note)', 'peak VRAM', v)
