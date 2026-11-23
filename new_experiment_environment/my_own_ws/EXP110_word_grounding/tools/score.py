#!/usr/bin/env python3
"""Score EXP110 predictions against the human answer key (annotations/pilot/<image stem>.json).

Usage (qwen_vl env):
  python tools/score.py --run run01 --split dev [--thr C2=0.25,C3=0.05,C4=0.05] [--sweep]
Writes results/<run>/scores_<timestamp>/ (never overwrites): per_query.csv, summary.json, SUMMARY.md.
Rules (DESIGN §7): Hungarian match IoU>=0.5 on query positives; unmatched predictions overlapping an ignore box
(IoU>=0.5) are dropped; the rest are false positives. success = all positives matched and no false positive.
Invalid model output (schema violation) = failure. Only images whose annotation status is DONE are scored.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config.json').read_text())
IOU_T = 0.5
SCORED_CONDS = ['C1', 'C2', 'C3', 'C4', 'C5B', 'C5L'] + [f'C6-{v}' for v in CFG['c6']['variants']]
INCOMPLETE = {}  # cond -> reasons; filled by rows_for(), reported in summary.json


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_in(p, g):
    cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
    return g[0] <= cx <= g[2] and g[1] <= cy <= g[3]


def match(preds, gts):
    """Hungarian on IoU (pairs below IoU_T forbidden). Returns list of (pred_idx, gt_idx)."""
    from scipy.optimize import linear_sum_assignment
    if not preds or not gts:
        return []
    m = np.array([[iou(p, g) for g in gts] for p in preds])
    cost = np.where(m >= IOU_T, -m, 1e6)
    r, c = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(r, c) if m[i, j] >= IOU_T]


def split_gt(gt_objects, q):
    pos = [o['box_xyxy'] for o in gt_objects if not o['ignore'] and o['label'] in q['target_labels']
           and (q['target_variant'] is None or o.get('variant') == q['target_variant'])]
    ign = [o['box_xyxy'] for o in gt_objects if o['ignore'] or o['label'] in q['ignore_labels']]
    return pos, ign


def score_one(pred_boxes, valid, gt_objects, q):
    pos, ign = split_gt(gt_objects, q)
    if not valid:
        return {'n_gt': len(pos), 'tp': 0, 'fp': 0, 'fn': len(pos), 'n_pred': 0, 'success': False, 'invalid': True,
                'center_recall_hits': 0, 'count_match': False}
    pairs = match(pred_boxes, pos)
    matched_p = {i for i, _ in pairs}
    rest = [p for i, p in enumerate(pred_boxes) if i not in matched_p]
    fp = [p for p in rest if not any(iou(p, g) >= IOU_T for g in ign)]
    tp = len(pairs)
    n_eff = tp + len(fp)  # predictions that are not ignored
    return {'n_gt': len(pos), 'tp': tp, 'fp': len(fp), 'fn': len(pos) - tp, 'n_pred': n_eff,
            'success': tp == len(pos) and not fp, 'invalid': False,
            'center_recall_hits': sum(any(center_in(p, g) for p in pred_boxes) for g in pos), 'count_match': n_eff == len(pos)}


def c6_breakdown(result, shared, gt_objects, q):
    """Per positive: in candidates? chosen? -> detector_miss / selector_miss / ok."""
    cands = [c['box'] for c in (shared if shared is not None else result.get('candidates') or [])]
    chosen = [p['box'] for p in result['preds']]
    pos, _ = split_gt(gt_objects, q)
    out = defaultdict(int)
    for g in pos:
        if not any(iou(c, g) >= IOU_T for c in cands):
            out['detector_miss'] += 1
        elif not any(iou(c, g) >= IOU_T for c in chosen):
            out['selector_miss'] += 1
        else:
            out['ok'] += 1
    return dict(out)


def load_gt(stem):
    p = ROOT / CFG['paths']['annotations'] / 'pilot' / f'{stem}.json'
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    return rec if rec['status'] == 'DONE' else None


def parse_thr(s):
    return {k: float(v) for k, v in (x.split('=') for x in s.split(','))} if s else {}


def rows_for(run_dir, split, thr, queries):
    rows = []
    qmap = {q['id']: q for q in queries}
    want = {q['id'] for q in queries}
    for cond in SCORED_CONDS:
        cdir = run_dir / cond
        if not cdir.exists():
            continue
        # completeness: every split image present with every query id, else the condition is excluded (not scored partially)
        missing = []
        for layout in CFG['splits'][split]:
            fs = sorted(cdir.glob(f'{layout}_*.json'))
            if not fs:
                missing.append(f'{layout}: no file')
            elif {r['qid'] for r in json.loads(fs[0].read_text())['results']} != want:
                missing.append(f'{layout}: query set differs')
        if missing:
            INCOMPLETE[cond] = missing
            print(f'WARNING {cond} excluded (incomplete): {missing}', file=sys.stderr)
            continue
        for layout in CFG['splits'][split]:
            files = sorted(cdir.glob(f'{layout}_*.json'))
            if not files:
                continue
            d = json.loads(files[0].read_text())
            stem = Path(d['image']).stem
            gt = load_gt(stem)
            if gt is None:
                continue
            if gt['image_sha256'] != d['image_sha256']:
                raise SystemExit(f'{stem}: answer key and prediction refer to different images')
            t = thr.get(cond)
            for r in d['results']:
                q = qmap[r['qid']]
                boxes = [p['box'] for p in r['preds'] if t is None or p['score'] is None or p['score'] >= t]
                s = score_one(boxes, r.get('valid', True), gt['objects'], q)
                row = {'cond': cond, 'layout': layout, 'qid': q['id'], 'type': q['type'], 'lang': q['lang'], 'text': q['text'],
                       'class': q['source_class'], 'threshold': t, **s}
                if cond.startswith('C6'):
                    row.update(c6_breakdown(r, d.get('shared_candidates'), gt['objects'], q))
                rows.append(row)
    return rows


def summarize(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r['cond'], r['lang'])].append(r)
        g[(r['cond'], r['lang'], r['type'])].append(r)
        g[(r['cond'], r['lang'], 'class:' + str(r['class']))].append(r)
    out = {}
    for k, rs in g.items():
        tp, fp, n = sum(r['tp'] for r in rs), sum(r['fp'] for r in rs), sum(r['n_gt'] for r in rs)
        out['|'.join(k)] = {'n': len(rs), 'success_rate': round(np.mean([r['success'] for r in rs]), 4),
                            'recall': round(tp / n, 4) if n else None, 'precision': round(tp / (tp + fp), 4) if tp + fp else None,
                            'invalid': sum(r['invalid'] for r in rs)}
        if k[0].startswith('C6'):  # failure decomposition, two explicit units
            fail = [r for r in rs if not r['success']]
            out['|'.join(k)]['c6_instances'] = {m: sum(r.get(m, 0) for r in rs) for m in ('ok', 'selector_miss', 'detector_miss')}
            out['|'.join(k)]['c6_failed_queries'] = {
                'n': len(fail), 'invalid': sum(r['invalid'] for r in fail),
                'any_detector_miss': sum(r.get('detector_miss', 0) > 0 for r in fail),
                'any_selector_miss': sum(r.get('selector_miss', 0) > 0 for r in fail),
                'fp_only': sum(not r['invalid'] and r.get('detector_miss', 0) == 0 and r.get('selector_miss', 0) == 0 for r in fail)}
    return out


def paired(rows, a, b, lang='en', n_boot=2000, seed=0):
    """McNemar exact on (query, layout) pairs + query-cluster bootstrap CI of success-rate difference a-b."""
    from scipy.stats import binomtest
    ka = {(r['qid'], r['layout']): r['success'] for r in rows if r['cond'] == a and r['lang'] == lang}
    kb = {(r['qid'], r['layout']): r['success'] for r in rows if r['cond'] == b and r['lang'] == lang}
    keys = sorted(set(ka) & set(kb))
    if not keys:
        return None
    n10 = sum(ka[k] and not kb[k] for k in keys)
    n01 = sum(kb[k] and not ka[k] for k in keys)
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    byq = defaultdict(list)
    for k in keys:
        byq[k[0]].append(ka[k] - kb[k])
    qids = sorted(byq)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(qids), len(qids), replace=True)
        vals = [v for i in pick for v in byq[qids[i]]]
        diffs.append(np.mean(vals))
    return {'a': a, 'b': b, 'lang': lang, 'pairs': len(keys), 'a_only': n10, 'b_only': n01, 'mcnemar_p': round(p, 5),
            'diff': round(float(np.mean([ka[k] - kb[k] for k in keys])), 4),
            'ci95_query_bootstrap': [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--split', required=True, choices=['dev', 'test'])
    ap.add_argument('--thr', default='', help='score thresholds per condition, e.g. C2=0.25,C3=0.05')
    ap.add_argument('--sweep', action='store_true', help='success rate vs threshold for C2-C4 (dev only)')
    a = ap.parse_args()
    if a.sweep and a.split != 'dev':  # refuse before any test answer key or prediction is read
        raise SystemExit('threshold sweep is dev-only')
    run_dir = ROOT / CFG['paths']['results'] / a.run
    queries = json.loads((ROOT / 'queries.json').read_text())['queries']
    thr = parse_thr(a.thr)
    rows = rows_for(run_dir, a.split, thr, queries)
    if not rows:
        raise SystemExit('nothing scored: no DONE answer keys for this split yet (annotations/pilot/<stem>.json)')
    out = run_dir / f"scores_{a.split}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    out.mkdir()
    with open(out / 'per_query.csv', 'w', newline='') as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (list(rows[0]).index(k) if k in rows[0] else 99, k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summ = {'thresholds': thr, 'iou': IOU_T, 'excluded_incomplete': INCOMPLETE, 'groups': summarize(rows),
            'paired_en': [p for c in SCORED_CONDS if c.startswith('C6') for p in [paired(rows, c, 'C1')] if p]}
    if a.sweep:
        if a.split != 'dev':
            raise SystemExit('threshold sweep is dev-only')
        summ['sweep'] = {}
        for c in ('C2', 'C3', 'C4'):
            for t in (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                rs = [r for r in rows_for(run_dir, a.split, {**thr, c: t}, queries) if r['cond'] == c and r['lang'] == 'en']
                if rs:
                    summ['sweep'][f'{c}@{t}'] = round(np.mean([r['success'] for r in rs]), 4)
    (out / 'summary.json').write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    lines = [f'# EXP110 scores ({a.split}, run {a.run})', '', f'thresholds: {thr or "none (all saved predictions)"}', '',
             '| cond | lang | n | success | recall | precision | invalid |', '|---|---|---|---|---|---|---|']
    for k, v in sorted(summ['groups'].items()):
        if k.count('|') == 1:
            c, l = k.split('|')
            lines.append(f"| {c} | {l} | {v['n']} | {v['success_rate']} | {v['recall']} | {v['precision']} | {v['invalid']} |")
    lines += ['', '## C6 failure decomposition (English)', '',
              '| cond | answer boxes: ok / selector_miss / detector_miss | failed queries | with >=1 detector_miss | with >=1 selector_miss | FP only | invalid |',
              '|---|---|---|---|---|---|---|']
    for k, v in sorted(summ['groups'].items()):
        if k.endswith('|en') and k.count('|') == 1 and 'c6_instances' in v:
            i, f = v['c6_instances'], v['c6_failed_queries']
            lines.append(f"| {k.split('|')[0]} | {i['ok']} / {i['selector_miss']} / {i['detector_miss']} | {f['n']} | "
                         f"{f['any_detector_miss']} | {f['any_selector_miss']} | {f['fp_only']} | {f['invalid']} |")
    lines += ['', f"## C6 vs C1 (English, {a.split}; dev = exploratory, test = registered comparison DESIGN §8/§12)", '']
    lines += [f"- {p['a']} - C1: diff {p['diff']} CI95 {p['ci95_query_bootstrap']}, discordant {p['a_only']}/{p['b_only']}, "
              f"McNemar p={p['mcnemar_p']}" for p in summ['paired_en']]
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('written', out)


def selftest():
    q = {'target_labels': ['drum'], 'target_variant': 'blue', 'ignore_labels': []}
    gt = [{'label': 'drum', 'variant': 'blue', 'box_xyxy': [0, 0, 10, 10], 'ignore': False},
          {'label': 'drum', 'variant': 'black', 'box_xyxy': [20, 0, 30, 10], 'ignore': False},
          {'label': 'ignore', 'variant': None, 'box_xyxy': [50, 50, 60, 60], 'ignore': True}]
    s = score_one([[0, 0, 10, 10]], True, gt, q)
    assert s['success'] and s['tp'] == 1 and s['fp'] == 0
    s = score_one([[0, 0, 10, 10], [20, 0, 30, 10]], True, gt, q)  # black drum for 'blue drum' -> FP
    assert not s['success'] and s['fp'] == 1
    s = score_one([[0, 0, 10, 10], [50, 50, 60, 60]], True, gt, q)  # box on ignore region is dropped
    assert s['success'] and s['n_pred'] == 1
    s = score_one([[0, 0, 10, 10]], False, gt, q)  # invalid output = failure
    assert not s['success'] and s['invalid']
    neg = {'target_labels': [], 'target_variant': None, 'ignore_labels': []}
    assert score_one([], True, gt, neg)['success'] and not score_one([[0, 0, 10, 10]], True, gt, neg)['success']
    assert match([[0, 0, 10, 10], [0, 0, 9, 10]], [[0, 0, 10, 10]]) == [(0, 0)]  # one GT matched once
    br = c6_breakdown({'preds': [{'box': [0, 0, 10, 10]}], 'candidates': None}, [{'box': [0, 0, 10, 10]}],
                      gt, {'target_labels': ['drum'], 'target_variant': None, 'ignore_labels': []})
    assert br == {'ok': 1, 'detector_miss': 1}, br
    rows = [{'cond': 'C6-pf', 'lang': 'en', 'qid': f'q{i}', 'layout': 'P01', 'success': i < 6} for i in range(10)] + \
           [{'cond': 'C1', 'lang': 'en', 'qid': f'q{i}', 'layout': 'P01', 'success': i < 3} for i in range(10)]
    p = paired(rows, 'C6-pf', 'C1', n_boot=200)
    assert p['a_only'] == 3 and p['b_only'] == 0 and abs(p['diff'] - 0.3) < 1e-9
    print('selftest PASS')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    else:
        main()
