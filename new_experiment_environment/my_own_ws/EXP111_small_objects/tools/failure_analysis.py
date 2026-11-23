#!/usr/bin/env python3
"""EXP111 step 0 (dev only): WHERE do blocks and small pallets get lost? Read-only over saved EXP110 outputs.

Per (dev image, small-object query, answer-key instance) it records:
  size_px            answer box width x height (raw distorted pixels, 1920x1080)
  qwen_tokens        size in Qwen3-VL visual tokens (one merged token = 32x32 px; Qwen sees the image at full
                     resolution: image_grid_thw = 68x120 patches of 16 px for 1920x1080, CONFIRMED 2026-11-15)
  dino_rank/score    first Grounding DINO box (run02 C2, query string only, top-100 saved) that hits the instance
  c6_stage           not_in_dino_top100 | filtered_out (C6-dino filter 0.10/top-15/area/NMS) | candidate_not_chosen | chosen | c6_invalid
  c1_hit             a C1 (Qwen alone) box hits the instance; c1_first_on = what the first C1 box landed on
A box "hits" an instance if IoU >= 0.5 (EXP110 rule) OR its center lies inside the instance box ("center"
is reported separately because 16-20 px blocks make IoU harsh). Answer keys are evaluation data: they are used
here only to diagnose, never as model input.
Usage: <qwen_vl python> tools/failure_analysis.py --out <new dir>
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXP110 = HERE.parent / 'EXP110_word_grounding'
sys.path.insert(0, str(EXP110 / 'tools'))
import score as S  # noqa: E402  (EXP110 code, read-only use)
import run_models as R  # noqa: E402

RUN = 'run02'
CLASSES = ('block', 'small_pallet')
TOKEN_PX = 32


def label_at(gt, box):
    c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    hit = [o for o in gt if o['box_xyxy'][0] <= c[0] <= o['box_xyxy'][2] and o['box_xyxy'][1] <= c[1] <= o['box_xyxy'][3]]
    if not hit:
        return 'nothing'
    o = min(hit, key=lambda o: (o['box_xyxy'][2] - o['box_xyxy'][0]) * (o['box_xyxy'][3] - o['box_xyxy'][1]))
    return o['label'] + (f"({o['variant']})" if o.get('variant') else '')


def hit(box, g):
    iou = S.iou(box, g)
    return iou >= 0.5, S.center_in(box, g), round(iou, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    qs = [q for q in R.QUERIES if q['source_class'] in CLASSES]
    vcfg = R.CFG['c6']['variants']['dino']
    rows = []
    for img_path in R.images_for('dev'):
        stem = img_path.stem
        gt = S.load_gt(stem)['objects']
        load = lambda c: {r['qid']: r for r in json.loads((EXP110 / 'results' / RUN / c / f'{stem}.json').read_text())['results']}
        c1, c2, c6 = load('C1'), load('C2'), load('C6-dino')
        for q in qs:
            pos, _ = S.split_gt(gt, q)
            r1, r2, r6 = c1[q['id']], c2[q['id']], c6[q['id']]
            cands = R.candidates(r2['preds'], vcfg, 1920, 1080)
            assert [c['box'] for c in cands] == [c['box'] for c in (r6['candidates'] or [])], 'C6 candidates not reproducible'
            c1_boxes = [p['box'] for p in r1['preds']] if r1['valid'] else []
            for g in pos:
                w, h = g[2] - g[0], g[3] - g[1]
                d_iou = [i for i, p in enumerate(r2['preds']) if hit(p['box'], g)[0]]
                d_ctr = [i for i, p in enumerate(r2['preds']) if hit(p['box'], g)[1]]
                cand_hit = [c['id'] for c in cands if hit(c['box'], g)[0] or hit(c['box'], g)[1]]
                chosen = set(r6['chosen_ids'] or [])
                if not r6['valid']:
                    stage = 'c6_invalid'
                elif not d_iou and not d_ctr:
                    stage = 'not_in_dino_top100'
                elif not cand_hit:
                    stage = 'filtered_out'
                elif chosen & set(cand_hit):
                    stage = 'chosen'
                else:
                    stage = 'candidate_not_chosen'
                first = (d_iou or d_ctr)[0] if (d_iou or d_ctr) else None
                rows.append({
                    'layout': stem[:3], 'qid': q['id'], 'lang': q['lang'], 'text': q['text'], 'class': q['source_class'],
                    'gt_box_raw_px': g, 'size_px': [round(w), round(h)], 'qwen_tokens': [round(w / TOKEN_PX, 2), round(h / TOKEN_PX, 2)],
                    'dino_rank_iou': d_iou[0] + 1 if d_iou else None, 'dino_rank_center': d_ctr[0] + 1 if d_ctr else None,
                    'dino_score': r2['preds'][first]['score'] if first is not None else None,
                    'c6_stage': stage, 'c6_chosen_on': [label_at(gt, c['box']) for c in cands if c['id'] in chosen],
                    'c1_valid': r1['valid'], 'c1_hit_iou': any(hit(b, g)[0] for b in c1_boxes),
                    'c1_hit_center': any(hit(b, g)[1] for b in c1_boxes),
                    'c1_first_on': label_at(gt, c1_boxes[0]) if c1_boxes else ('INVALID' if not r1['valid'] else 'NO BOX')})
    (out / 'rows.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    L = [f'# EXP111 step 0: small-object failure causes (dev P01-P04, EXP110 {RUN} outputs)', '',
         'Unit = (image, query, answer instance). "hit" = IoU>=0.5 or box center inside the instance.', '']
    for cls in CLASSES:
        rs = [r for r in rows if r['class'] == cls]
        ws = sorted(r['size_px'][0] for r in rs); hs = sorted(r['size_px'][1] for r in rs)
        L += [f'## {cls}', '', f"- answer size: width {ws[0]}-{ws[-1]} px, height {hs[0]}-{hs[-1]} px "
              f"= {ws[0] / TOKEN_PX:.1f}-{ws[-1] / TOKEN_PX:.1f} Qwen tokens wide (1 token = {TOKEN_PX} px)", '',
              '| lang | n | C6: not in DINO top-100 | filtered out | candidate, not chosen | chosen | invalid | C1 hit (IoU / center) | C1 first box landed on |',
              '|---|---|---|---|---|---|---|---|---|']
        for lang in ('en', 'ko'):
            rl = [r for r in rs if r['lang'] == lang]
            st = Counter(r['c6_stage'] for r in rl)
            fo = Counter(r['c1_first_on'] for r in rl).most_common(4)
            L.append(f"| {lang} | {len(rl)} | {st['not_in_dino_top100']} | {st['filtered_out']} | {st['candidate_not_chosen']} | "
                     f"{st['chosen']} | {st['c6_invalid']} | {sum(r['c1_hit_iou'] for r in rl)} / {sum(r['c1_hit_center'] for r in rl)} | "
                     f"{', '.join(f'{k} {v}' for k, v in fo)} |")
        L += ['', '| query | n | DINO rank of the instance (IoU or center hit; "-" = not in top-100) | C6 chose (landed on) |', '|---|---|---|---|']
        byq = defaultdict(list)
        for r in rs:
            byq[(r['qid'], r['text'])].append(r)
        for (qid, text), rl in sorted(byq.items()):
            ranks = [str(r['dino_rank_iou'] or r['dino_rank_center'] or '-') for r in rl]
            ch = Counter(x for r in rl for x in r['c6_chosen_on']).most_common(3)
            L.append(f"| {qid} {text} | {len(rl)} | {', '.join(ranks)} | {', '.join(f'{k} {v}' for k, v in ch) or 'none'} |")
        L.append('')
    (out / 'SUMMARY.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('written', out)


if __name__ == '__main__':
    sys.exit(main())
