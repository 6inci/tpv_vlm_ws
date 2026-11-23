#!/usr/bin/env python3
"""EXP111 pilot (dev only): verify each size-gated DINO candidate on an enlarged crop.

Candidates: s2 gate (config s2_gated_pilot) on saved run02 DINO boxes for query q025 "small pallet".
For each candidate: square crop centred on it (EXP110 X4 oracle crop rule, config s3_zoom_verify_pilot.crop),
enlarged, then Qwen3-VL-4B answers the fixed yes/no prompt. Kept = candidates answered true.
Crop boxes are raw distorted pixels; nothing is converted back except the candidate box itself.
Score per answer instance (IoU >= 0.5 or center inside). Raw answers are saved.
Usage: <qwen_vl python> tools/s3_zoom_verify_pilot.py --out <new dir>
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXP110 = HERE.parent / 'EXP110_word_grounding'
sys.path.insert(0, str(EXP110 / 'tools'))
sys.path.insert(0, str(EXP110 / 'explore' / 'X4_block_zoom'))
sys.path.insert(0, str(HERE / 'tools'))
import score as S  # noqa: E402
import run_models as R  # noqa: E402
from x4_block_zoom import oracle_crops  # noqa: E402  (crop geometry only; input boxes here are DINO candidates)
from s2_gated_pilot import gate, hits, PC as S2  # noqa: E402

CFG = json.loads((HERE / 'config.json').read_text())
VC = CFG['s3_zoom_verify_pilot']


def parse_yes(raw):
    try:
        d = json.loads(R.strip_fence(raw))
        v = d.get('small_pallet') if isinstance(d, dict) else None
        return (v, None) if isinstance(v, bool) else (None, 'schema')
    except json.JSONDecodeError as e:
        return None, f'JSONDecodeError: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--describe', action='store_true', help='plan A: prepend config vocabulary_descriptions.small_pallet')
    a = ap.parse_args()
    from PIL import Image
    desc = CFG['vocabulary_descriptions']['small_pallet'] + '\n' if a.describe else ''
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    qwen = R.get_qwen()
    rows = []
    for img_path in R.images_for('dev'):
        stem = img_path.stem
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        gt = S.load_gt(stem)['objects']
        pos = [o['box_xyxy'] for o in gt if o['label'] == 'small_pallet']
        c2 = {r['qid']: r for r in json.loads((EXP110 / 'results' / S2['source_run'] / 'C2' / f'{stem}.json').read_text())['results']}
        cands = gate(c2[VC['query_for_dino']]['preds'])
        crops = oracle_crops([c['box'] for c in cands], W, H, VC['crop'])
        for c, (crop, scale) in zip(cands, crops):
            enl = img.crop(tuple(crop)).resize((round((crop[2] - crop[0]) * scale), round((crop[3] - crop[1]) * scale)), Image.BICUBIC)
            raw, ntok, nin, dt = qwen.ask(enl, desc + VC['prompt'], VC['max_new_tokens'])
            yes, err = parse_yes(raw)
            rows.append({'layout': stem[:3], 'cand_id': c['id'], 'box_raw_px': c['box'], 'dino_score': c['score'], 'crop_raw_px': crop,
                         'scale': round(scale, 3), 'raw': raw, 'error': err, 'yes': yes, 'latency_s': round(dt, 2),
                         'is_small_pallet': any(hits(c['box'], g) for g in pos)})
        kept = [r for r in rows if r['layout'] == stem[:3] and r['yes']]
        found = sum(any(hits(r['box_raw_px'], g) for r in kept) for g in pos)
        print(stem[:3], '| cands', len(cands), '| kept', len(kept), '| pallets found', found, '/', len(pos),
              '| FP', sum(not r['is_small_pallet'] for r in kept), flush=True)
    man = {'id': 'EXP111-s3-pilot', 'describe': a.describe, 'description': desc or None, 'started': R.now(), 'env': R.env_info(), 'config': {'s2': S2, 's3': VC},
           'sha256': {p: R.sha256(HERE / p) for p in ['tools/s3_zoom_verify_pilot.py', 'tools/s2_gated_pilot.py', 'config.json']}}
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1, ensure_ascii=False))
    (out / 'rows.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    tp = sum(r['yes'] and r['is_small_pallet'] for r in rows); fp = sum(bool(r['yes']) and not r['is_small_pallet'] for r in rows)
    fn = sum((not r['yes']) and r['is_small_pallet'] for r in rows); inv = sum(r['error'] is not None for r in rows)
    print('candidate level: TP', tp, 'FP', fp, 'FN', fn, 'TN', len(rows) - tp - fp - fn, '| invalid', inv, '| n', len(rows))
    print('written', out)


if __name__ == '__main__':
    sys.exit(main())
