#!/usr/bin/env python3
"""EXP111 pilot (dev only): find blocks by zooming around PREDICTED small pallets (no answer-key locations).

Pallets: candidates that s3A (size gate + zoom verification + description) answered true, false positives kept.
For each block query: one enlarged crop per predicted pallet (EXP110 X4 crop rule), EXP110 C1 prompt unchanged,
boxes mapped back to raw distorted pixels (raw = crop_origin + enlarged_px / scale), merged with greedy NMS.
Scored per (image, query) with the EXP110 strict rule (score.py score_one) and center hits.
Pallet state (loaded / empty) is derived in Python from the predicted English block boxes (config state_rule).
Usage: <qwen_vl python> tools/s4_blocks_on_predicted.py --out <new dir>
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXP110 = HERE.parent / 'EXP110_word_grounding'
sys.path.insert(0, str(EXP110 / 'tools'))
sys.path.insert(0, str(EXP110 / 'explore' / 'X4_block_zoom'))
import score as S  # noqa: E402
import run_models as R  # noqa: E402
from x4_block_zoom import oracle_crops, to_raw, nms  # noqa: E402

CFG = json.loads((HERE / 'config.json').read_text())
BC = CFG['s4_blocks_on_predicted']
CROP = CFG['s3_zoom_verify_pilot']['crop']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from PIL import Image
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    prows = json.loads((HERE / BC['pallet_rows']).read_text())
    qmap = {q['id']: q for q in R.QUERIES}
    qwen = R.get_qwen()
    prompt, mnt = R.CFG['qwen']['c1_prompt'], R.CFG['qwen']['c1_max_new_tokens']
    calls, rows, states = [], [], []
    for img_path in R.images_for('dev'):
        stem, layout = img_path.stem, img_path.stem[:3]
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        gt = S.load_gt(stem)['objects']
        pallets = [r['box_raw_px'] for r in prows if r['layout'] == layout and r['yes']]
        crops = oracle_crops(pallets, W, H, CROP)
        en_blocks = []
        for qid in BC['queries']:
            q = qmap[qid]
            merged, n_inv = [], 0
            for crop, scale in crops:
                enl = img.crop(tuple(crop)).resize((round((crop[2] - crop[0]) * scale), round((crop[3] - crop[1]) * scale)), Image.BICUBIC)
                raw, ntok, nin, dt = qwen.ask(enl, prompt.format(q=q['text']), mnt)
                preds, err = R.parse_c1(raw, *enl.size)
                boxes = [to_raw(p['box'], crop, scale) for p in (preds or [])]
                n_inv += err is not None
                merged += boxes
                calls.append({'layout': layout, 'qid': qid, 'crop_raw_px': crop, 'scale': scale, 'raw': raw, 'error': err,
                              'boxes_raw_px': boxes, 'latency_s': round(dt, 2)})
            boxes = nms(merged, BC['merge_nms_iou'], S.iou)
            if q['lang'] == 'en':
                en_blocks += boxes
            pos, _ = S.split_gt(gt, q)
            st = S.score_one(boxes, n_inv == 0, gt, q)
            ctr = sum(any(S.center_in(b, g) for b in boxes) for g in pos)
            rows.append({'layout': layout, 'qid': qid, 'lang': q['lang'], 'text': q['text'], 'n_pallet_crops': len(crops),
                         'n_invalid_crops': n_inv, 'boxes_raw_px': boxes, 'strict': st, 'center_hits': ctr})
            print(layout, q['text'], '| crops', len(crops), '| n_gt', st['n_gt'], 'tp', st['tp'], 'fp', st['fp'], 'center', ctr,
                  '| success', st['success'], flush=True)
        blocks_gt = [o['box_xyxy'] for o in gt if o['label'] == 'block']
        for g in [o['box_xyxy'] for o in gt if o['label'] == 'small_pallet']:
            true_state = 'loaded' if any(S.center_in(b, g) for b in blocks_gt) else 'empty'
            match = [p for p in pallets if S.iou(p, g) >= 0.5 or S.center_in(p, g)]
            pred_state = None if not match else ('loaded' if any(S.center_in(b, match[0]) for b in en_blocks) else 'empty')
            states.append({'layout': layout, 'gt_pallet': g, 'true_state': true_state, 'found': bool(match), 'pred_state': pred_state})
    man = {'id': 'EXP111-s4-pilot', 'started': R.now(), 'env': R.env_info(), 'config': BC, 'crop': CROP, 'c1_prompt': prompt,
           'sha256': {p: R.sha256(HERE / p) for p in ['tools/s4_blocks_on_predicted.py', 'config.json', BC['pallet_rows']]}}
    for name, obj in (('MANIFEST.json', man), ('calls.json', calls), ('rows.json', rows), ('states.json', states)):
        (out / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    for lang in ('en', 'ko'):
        rl = [r for r in rows if r['lang'] == lang]
        print(lang, '| queries success', sum(r['strict']['success'] for r in rl), '/', len(rl), '| block instances IoU-hit',
              sum(r['strict']['tp'] for r in rl), '| center-hit', sum(r['center_hits'] for r in rl), '/', sum(r['strict']['n_gt'] for r in rl),
              '| FP', sum(r['strict']['fp'] for r in rl))
    print('pallet state correct', sum(s['pred_state'] == s['true_state'] for s in states), '/', len(states), '| written', out)


if __name__ == '__main__':
    sys.exit(main())
