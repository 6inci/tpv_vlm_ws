#!/usr/bin/env python3
"""Summarise EXP110-X4 runs against the full-image baselines on the same queries (dev, saved outputs only).
Per mode and baseline: instance recall (tp/n_gt), precision, strict success, by query type and language,
and what each false-positive box overlaps in the answer key (center inside which labelled object, else 'floor/none').
Usage: python explore/X4_block_zoom/x4_summary.py <oracle_run_dir> <tiles_run_dir> > <out>.md
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import score as S  # noqa: E402

Q = {q['id']: q for q in json.loads((ROOT / 'queries.json').read_text())['queries']}


def what_is(box, gt):
    hits = [o for o in gt if S.center_in(box, o['box_xyxy'])]
    if not hits:
        return 'floor/none'
    o = min(hits, key=lambda o: (o['box_xyxy'][2] - o['box_xyxy'][0]) * (o['box_xyxy'][3] - o['box_xyxy'][1]))
    return o['label'] + (':' + o['variant'] if o.get('variant') else '')


def rows_from_run(run, cond, qids):
    out = []
    for layout in S.CFG['splits']['dev']:
        d = json.loads(sorted((ROOT / 'results' / run / cond).glob(f'{layout}_*.json'))[0].read_text())
        gt = S.load_gt(Path(d['image']).stem)['objects']
        for r in d['results']:
            if r['qid'] in qids:
                boxes = [p['box'] for p in r['preds']]
                s = S.score_one(boxes, r.get('valid', True), gt, Q[r['qid']])
                out.append({'layout': layout, 'qid': r['qid'], 'boxes': boxes if r.get('valid', True) else [], 'strict': s, 'lenient': s})
    return out


def table(name, rows):
    agg = defaultdict(lambda: Counter())
    fps = Counter()
    for r in rows:
        q = Q[r['qid']]
        gt = S.load_gt(sorted((ROOT / 'annotations' / 'pilot').glob(f"{r['layout']}_*.json"))[0].stem)['objects']
        pos, _ = S.split_gt(gt, q)
        for key in (f"{q['lang']}|all", f"{q['lang']}|{q['type']}|{q['source_class']}"):
            a = agg[key]
            a['n'] += 1; a['succ'] += r['strict']['success']; a['tp'] += r['lenient']['tp']
            a['fp'] += r['lenient']['fp']; a['gt'] += r['lenient']['n_gt']; a['invalid'] += r['strict']['invalid']
        for b in r['boxes'] if 'boxes' in r else r.get('boxes_raw_px', []):
            if not any(S.iou(b, g) >= S.IOU_T for g in pos):
                fps[(q['source_class'], what_is(b, gt))] += 1
    lines = [f'### {name}', '', '| group | n | strict success | instance recall | precision | invalid |', '|---|---|---|---|---|---|']
    for k in sorted(agg):
        a = agg[k]
        prec = a['tp'] / (a['tp'] + a['fp']) if a['tp'] + a['fp'] else float('nan')
        lines.append(f"| {k} | {a['n']} | {a['succ']}/{a['n']} | {a['tp']}/{a['gt']} | {prec:.2f} | {a['invalid']} |")
    lines += ['', 'unmatched boxes -> what they hit (query class, object under box center): ' +
              ', '.join(f'{c}->{w}: {n}' for (c, w), n in fps.most_common(12)), '']
    return lines


def main():
    oracle, tiles = Path(sys.argv[1]), Path(sys.argv[2])
    lines = ['# EXP110-X4 summary (exploratory, dev P01-P04)', '',
             'strict = all positives at IoU>=0.5 and no FP (invalid crop answer -> failure); recall/precision count boxes from valid crops.', '']
    for name, d in (('X4 oracle crops (pallet location given)', oracle), ('X4 tiles (no prior)', tiles)):
        rows = json.loads((d / 'rows.json').read_text())
        for r in rows:
            r['boxes'] = r['boxes_raw_px']
        lines += table(name, rows)
        qids = {r['qid'] for r in rows}
        if d == tiles:
            for run, cond in (('run02', 'C1'), ('run02', 'C6-dino')):
                lines += table(f'baseline {cond} ({run}, full image, same queries as tiles)', rows_from_run(run, cond, qids))
    print('\n'.join(lines))


if __name__ == '__main__':
    sys.exit(main())
