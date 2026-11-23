#!/usr/bin/env python3
"""EXP110-X1 (exploratory): score saved predictions under three candidate metrics (dev; test after DESIGN freeze, --split test).

  strict : existing primary (Hungarian IoU>=0.5, all positives found, no FP)          -> tools/score.py
  cu     : context-unit. One-to-one match where the predicted box CENTER lies inside the answer box
           (Hungarian, IoU as tie-break). Success = every positive matched and no unmatched prediction
           outside ignore boxes (= right count + rough location).
  tr     : target-reaching (word level). Success = the FIRST prediction's center lies inside any positive
           (Q1-Q3); for absent objects (Q5) success = no prediction. First = output order (C1), Qwen's id
           order as written in the raw answer (C6; stored chosen_ids are sorted), highest score (detectors).
           Note: cu drops unmatched boxes whose CENTER lies in an ignore box (strict uses IoU>=0.5); dev has 0 ignores.
All boxes in raw (distorted) pixels, 1920x1080. Nothing here is a registered metric; see DESIGN §12.
Usage: python explore/X1_metrics/x1_metrics.py --runs run01,run02 --out <dir>
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import score as S  # noqa: E402
import run_models as R  # noqa: E402  (strip_fence only; no model is loaded)

GROUPS = {'task': {'block', 'small_pallet'},
          'unique': {'truck', 'conveyor', 'workbench', 'locker', 'storage_cubby', 'ibc_tank', 'box_stack', 'fire_extinguisher'},
          'multi': {'traffic_cone', 'pallet', 'robot_arm', 'storage_rack', 'drum', 'cart', 'charger'}}


def group_of(q):
    if q['type'] == 'Q5':
        return 'absent'
    return next(g for g, cs in GROUPS.items() if q['source_class'] in cs)


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def cu_success(preds, pos, ign):
    from scipy.optimize import linear_sum_assignment
    matched = set()
    if preds and pos:
        m = np.array([[S.center_in(p, g) for g in pos] for p in preds], dtype=bool)
        cost = np.where(m, -1.0 - np.array([[S.iou(p, g) for g in pos] for p in preds]), 1e6)
        r, c = linear_sum_assignment(cost)
        matched = {i for i, j in zip(r, c) if m[i, j]}
        n_match = len(matched)
    else:
        n_match = 0
    fp = [p for i, p in enumerate(preds) if i not in matched and not any(S.center_in(p, g) for g in ign)]
    return n_match == len(pos) and not fp


def tr_success(first, pos, qtype):
    if qtype == 'Q5':
        return first is None
    return first is not None and any(S.center_in(first, g) for g in pos)


def ordered_boxes(cond, r, shared):
    """Prediction boxes in the order the system would act on them."""
    if not r.get('valid', True):
        return None  # invalid output: nothing usable
    if cond.startswith('C6'):
        cands = {c['id']: c['box'] for c in (shared if shared is not None else r.get('candidates') or [])}
        # stored chosen_ids are sorted by run_models.parse_c6; take Qwen's own order from the raw answer
        order = json.loads(R.strip_fence(r['raw']))['ids'] if r.get('raw') else []
        seen, ids = set(), []
        for i in order:
            if i in r.get('chosen_ids', []) and i not in seen:
                seen.add(i); ids.append(i)
        return [cands[i] for i in ids if i in cands]
    if cond == 'C1':
        return [p['box'] for p in r['preds']]
    return [p['box'] for p in sorted(r['preds'], key=lambda p: -(p['score'] or 0))]


def rows_for_run(run, queries, split='dev'):
    run_dir = ROOT / 'results' / run
    qmap = {q['id']: q for q in queries}
    rows = []
    for cond in S.SCORED_CONDS:
        if not (run_dir / cond).exists():
            continue
        for layout in S.CFG['splits'][split]:
            files = sorted((run_dir / cond).glob(f'{layout}_*.json'))
            if not files:
                continue
            d = json.loads(files[0].read_text())
            gt = S.load_gt(Path(d['image']).stem)
            assert gt['image_sha256'] == d['image_sha256']
            for r in d['results']:
                q = qmap[r['qid']]
                pos, ign = S.split_gt(gt['objects'], q)
                boxes = ordered_boxes(cond, r, d.get('shared_candidates'))
                valid = boxes is not None
                boxes = boxes or []
                rows.append({'run': run, 'cond': cond, 'layout': layout, 'qid': q['id'], 'lang': q['lang'], 'type': q['type'],
                             'group': group_of(q), 'class': q['source_class'],
                             'strict': S.score_one([p['box'] for p in r['preds']] if valid else [], valid, gt['objects'], q)['success'],
                             'cu': valid and cu_success(boxes, pos, ign),
                             'tr': valid and tr_success(boxes[0] if boxes else None, pos, q['type'])})
    return rows


def selftest():
    g = [[0, 0, 10, 10], [20, 0, 30, 10]]
    assert cu_success([[1, 1, 9, 9], [19, -5, 33, 12]], g, [])           # loose boxes, right count
    assert not cu_success([[1, 1, 9, 9]], g, [])                           # missing one
    assert not cu_success([[1, 1, 9, 9], [2, 2, 8, 8]], g[:1], [])         # duplicate on one object = FP
    assert cu_success([[1, 1, 9, 9], [50, 50, 60, 60]], g[:1], [[45, 45, 65, 65]])  # extra inside ignore dropped
    assert tr_success([1, 1, 9, 9], g, 'Q1') and not tr_success(None, g, 'Q1') and tr_success(None, [], 'Q5')
    print('selftest PASS')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='run01')
    ap.add_argument('--out', required=False)
    ap.add_argument('--split', default='dev', choices=['dev', 'test'], help='test only after DESIGN freeze (§12)')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    queries = json.loads((ROOT / 'queries.json').read_text())['queries']
    rows = [r for run in a.runs.split(',') for r in rows_for_run(run, queries, a.split)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'rows.json').write_text(json.dumps(rows, ensure_ascii=False))
    agg = defaultdict(list)
    for r in rows:
        for grp in ('all', r['group']):
            agg[(r['run'], r['cond'], r['lang'], grp)].append(r)
    table = {'|'.join(k): {m: round(float(np.mean([r[m] for r in v])), 4) for m in ('strict', 'cu', 'tr')} | {'n': len(v)}
             for k, v in sorted(agg.items())}
    (out / 'summary.json').write_text(json.dumps(table, indent=1))
    lines = [f"# EXP110-X1 three metrics (exploratory, {a.split}: {', '.join(S.CFG['splits'][a.split])})", '',
             'C2 rows here use ALL saved predictions (no threshold): compare C1 and C6 only.', '',
             'strict = IoU>=0.5 all & only | cu = center 1:1 + count | tr = first box on a correct object (Q5: no box)', '']
    for lang in ('en', 'ko'):
        lines += [f'## {lang}', '', '| run | cond | group | n | strict | cu | tr |', '|---|---|---|---|---|---|---|']
        for k, v in table.items():
            run, cond, l, grp = k.split('|')
            if l == lang:
                lines.append(f"| {run} | {cond} | {grp} | {v['n']} | {v['strict']} | {v['cu']} | {v['tr']} |")
        lines.append('')
    (out / 'SUMMARY.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:40]))
    print('written', out)


if __name__ == '__main__':
    sys.exit(main())
