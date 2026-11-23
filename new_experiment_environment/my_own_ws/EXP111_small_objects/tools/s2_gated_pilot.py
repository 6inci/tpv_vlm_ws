#!/usr/bin/env python3
"""EXP111 pilot (dev only): can Qwen pick the small pallets once DINO candidates are size-gated?

Same as EXP110 C6-dino except the candidate filter: saved run02 DINO boxes for the query (top-100, query string
only) -> keep boxes whose both sides are within config s2_gated_pilot.size_gate_px -> greedy NMS -> top_k ->
numbered overlay -> Qwen3-VL-4B with the unchanged EXP110 C6 prompt. Score threshold is not applied
(EXP110 step 0: the pallets sit at DINO ranks 4-89 with low scores).
Score per answer instance: candidate coverage, and whether Qwen chose a candidate that hits it
(IoU >= 0.5 or center inside). Boxes are raw distorted pixels. Answer keys are used only for scoring.
Usage: <qwen_vl python> tools/s2_gated_pilot.py --out <new dir>
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXP110 = HERE.parent / 'EXP110_word_grounding'
sys.path.insert(0, str(EXP110 / 'tools'))
import score as S  # noqa: E402
import run_models as R  # noqa: E402

CFG = json.loads((HERE / 'config.json').read_text())
PC = CFG['s2_gated_pilot']


def gate(preds):
    g = PC['size_gate_px']
    ps = [p for p in preds if g['min_side'] <= p['box'][2] - p['box'][0] <= g['max_side']
          and g['min_side'] <= p['box'][3] - p['box'][1] <= g['max_side']]
    ps.sort(key=lambda p: -(p['score'] or 0))
    kept = []
    for p in ps:
        if all(S.iou(p['box'], k['box']) < PC['nms_iou'] for k in kept):
            kept.append(p)
        if len(kept) == PC['top_k']:
            break
    return [{'id': i + 1, 'box': p['box'], 'score': p['score']} for i, p in enumerate(kept)]


def hits(box, g):
    return S.iou(box, g) >= 0.5 or S.center_in(box, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--describe', action='store_true', help='plan A: prepend config vocabulary_descriptions.small_pallet')
    a = ap.parse_args()
    from PIL import Image
    desc = CFG['vocabulary_descriptions']['small_pallet'] + '\n' if a.describe else ''
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    qmap = {q['id']: q for q in R.QUERIES}
    qwen = R.get_qwen()
    rows = []
    for img_path in R.images_for('dev'):
        stem = img_path.stem
        img = Image.open(img_path).convert('RGB')
        gt = S.load_gt(stem)['objects']
        c2 = {r['qid']: r for r in json.loads((EXP110 / 'results' / PC['source_run'] / 'C2' / f'{stem}.json').read_text())['results']}
        for qid in PC['queries']:
            q = qmap[qid]
            pos, _ = S.split_gt(gt, q)
            cands = gate(c2[qid]['preds'])
            raw, ntok, nin, dt = qwen.ask(R.overlay(img, cands), desc + R.CFG['qwen']['c6_prompt'].format(q=q['text']), R.CFG['qwen']['c6_max_new_tokens'])
            ids, err = R.parse_c6(raw, {c['id'] for c in cands})
            chosen = [c for c in cands if ids and c['id'] in ids]
            per = [{'gt': g, 'in_candidates': any(hits(c['box'], g) for c in cands), 'chosen': any(hits(c['box'], g) for c in chosen)} for g in pos]
            fp = [c['box'] for c in chosen if not any(hits(c['box'], g) for g in pos)]
            rows.append({'layout': stem[:3], 'qid': qid, 'lang': q['lang'], 'text': q['text'], 'n_candidates': len(cands),
                         'candidates': cands, 'raw': raw, 'error': err, 'chosen_ids': ids or [], 'instances': per,
                         'n_fp': len(fp), 'latency_s': round(dt, 2)})
            print(stem[:3], q['text'], '| cands', len(cands), '| in cands', sum(p['in_candidates'] for p in per), '/', len(per),
                  '| chosen', sum(p['chosen'] for p in per), '| FP', len(fp), '| raw', raw.strip()[:60], flush=True)
    man = {'id': 'EXP111-s2-pilot', 'describe': a.describe, 'description': desc or None, 'started': R.now(), 'env': R.env_info(), 'config': PC, 'qwen_model': R.CFG['models']['qwen'],
           'c6_prompt': R.CFG['qwen']['c6_prompt'], 'sha256': {p: R.sha256(HERE / p) for p in ['tools/s2_gated_pilot.py', 'config.json']}}
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1, ensure_ascii=False))
    (out / 'rows.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    for lang in ('en', 'ko'):
        rl = [r for r in rows if r['lang'] == lang]
        n = sum(len(r['instances']) for r in rl)
        print(lang, 'instances', n, '| in candidates', sum(p['in_candidates'] for r in rl for p in r['instances']),
              '| chosen', sum(p['chosen'] for r in rl for p in r['instances']), '| FP boxes', sum(r['n_fp'] for r in rl),
              '| queries all-found-no-FP', sum(all(p['chosen'] for p in r['instances']) and r['n_fp'] == 0 for r in rl), '/', len(rl))
    print('written', out)


if __name__ == '__main__':
    sys.exit(main())
