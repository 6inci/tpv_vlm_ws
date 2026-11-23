#!/usr/bin/env python3
"""EXP110-X4 (exploratory, dev only): ask Qwen (C1 prompt) on enlarged crops for blocks / small pallets.

Modes (explore/X4_block_zoom/config.json):
  oracle : square crop around each answer-key small pallet (upper bound, location given)
  tiles  : overlapping tiles over the whole image (no prior)
Coordinates: crop box [x0,y0,x1,y1] in raw distorted pixels -> crop enlarged by `scale` -> Qwen 0-1000 box parsed
against the enlarged crop size -> raw = crop_origin + enlarged_px / scale. Merged across crops with greedy NMS.
Only dev images are loaded. Raw Qwen text of every call is saved.
Usage: python explore/X4_block_zoom/x4_block_zoom.py --mode oracle|tiles --out <dir>   |   --selftest
"""
import argparse, json, platform, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
XCFG = json.loads((HERE / 'config.json').read_text())


def to_raw(box_enl, crop, scale):
    x0, y0 = crop[0], crop[1]
    return [x0 + box_enl[0] / scale, y0 + box_enl[1] / scale, x0 + box_enl[2] / scale, y0 + box_enl[3] / scale]


def oracle_crops(pallets, W, H, c):
    out = []
    for g in pallets:
        cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
        side = max(c['min_side_px'], c['context_scale'] * max(g[2] - g[0], g[3] - g[1]))
        x0, y0 = max(0, cx - side / 2), max(0, cy - side / 2)
        x1, y1 = min(W, x0 + side), min(H, y0 + side)
        x0, y0 = max(0, x1 - side), max(0, y1 - side)
        crop = [round(x0), round(y0), round(x1), round(y1)]
        out.append((crop, c['enlarged_long_side_px'] / max(crop[2] - crop[0], crop[3] - crop[1])))
    return out


def tile_crops(W, H, c):
    tw = W / (c['cols'] - (c['cols'] - 1) * c['overlap_frac'])
    th = H / (c['rows'] - (c['rows'] - 1) * c['overlap_frac'])
    out = []
    for r in range(c['rows']):
        for k in range(c['cols']):
            x0, y0 = k * tw * (1 - c['overlap_frac']), r * th * (1 - c['overlap_frac'])
            out.append(([round(x0), round(y0), round(min(W, x0 + tw)), round(min(H, y0 + th))], c['scale']))
    return out


def nms(boxes, iou_t, iou):
    keep = []
    for b in boxes:
        if all(iou(b, k) < iou_t for k in keep):
            keep.append(b)
    return keep


def selftest():
    assert to_raw([0, 0, 200, 100], [100, 50, 300, 150], 2.0) == [100.0, 50.0, 200.0, 100.0]
    t = tile_crops(1920, 1080, XCFG['modes']['tiles'])
    assert len(t) == 6 and t[0][0][:2] == [0, 0] and t[-1][0][2:] == [1920, 1080]
    o = oracle_crops([[582, 273, 628, 314]], 1920, 1080, XCFG['modes']['oracle'])[0]
    assert o[0][0] <= 582 and o[0][2] >= 628 and abs(o[1] * (o[0][2] - o[0][0]) - 768) < 1e-6
    o2 = oracle_crops([[0, 0, 40, 40]], 1920, 1080, XCFG['modes']['oracle'])[0][0]
    assert o2[0] == 0 and o2[1] == 0 and o2[2] - o2[0] == 160  # clipped at image border, size kept
    print('selftest PASS')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['oracle', 'tiles'])
    ap.add_argument('--out')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import run_models as R
    import score as S
    from PIL import Image
    mc = XCFG['modes'][a.mode]
    qmap = {q['id']: q for q in R.QUERIES}
    qs = [qmap[i] for i in mc['queries']]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    prompt, mnt = R.CFG['qwen']['c1_prompt'], R.CFG['qwen']['c1_max_new_tokens']
    manifest = {'id': XCFG['id'], 'mode': a.mode, 'started': R.now(), 'argv': sys.argv, 'env': R.env_info(),
                'x4_config': XCFG, 'c1_prompt': prompt, 'max_new_tokens': mnt, 'model': R.CFG['models']['qwen'],
                'sha256': {p: R.sha256(ROOT / p) for p in ['explore/X4_block_zoom/x4_block_zoom.py', 'explore/X4_block_zoom/config.json',
                                                            'tools/run_models.py', 'tools/score.py', 'config.json', 'queries.json']},
                'python': platform.python_version()}
    qwen = R.get_qwen()
    calls, rows = [], []
    for img_path in R.images_for('dev'):  # dev only
        stem = img_path.stem
        gt = S.load_gt(stem)
        img = Image.open(img_path).convert('RGB')
        assert gt['image_sha256'] == R.sha256(img_path)
        W, H = img.size
        if a.mode == 'oracle':
            crops = oracle_crops([o['box_xyxy'] for o in gt['objects'] if o['label'] == 'small_pallet'], W, H, mc)
        else:
            crops = tile_crops(W, H, mc)
        for q in qs:
            merged, n_invalid = [], 0
            for ci, (crop, scale) in enumerate(crops):
                enl = img.crop(tuple(crop)).resize((round((crop[2] - crop[0]) * scale), round((crop[3] - crop[1]) * scale)), Image.BICUBIC)
                raw, ntok, nin, dt = qwen.ask(enl, prompt.format(q=q['text']), mnt)
                preds, err = R.parse_c1(raw, *enl.size)
                raw_boxes = [to_raw(p['box'], crop, scale) for p in (preds or [])]
                n_invalid += err is not None
                merged += raw_boxes
                calls.append({'layout': stem[:3], 'qid': q['id'], 'text': q['text'], 'crop_raw_px': crop, 'scale': scale,
                              'enlarged_size': list(enl.size), 'raw': raw, 'error': err, 'boxes_raw_px': raw_boxes,
                              'generated_tokens': ntok, 'input_tokens': nin, 'latency_s': round(dt, 3)})
            boxes = nms(merged, XCFG['merge_nms_iou'], S.iou)
            strict = S.score_one(boxes, n_invalid == 0, gt['objects'], q)
            lenient = S.score_one(boxes, True, gt['objects'], q)
            rows.append({'layout': stem[:3], 'qid': q['id'], 'lang': q['lang'], 'text': q['text'], 'class': q['source_class'],
                         'n_crops': len(crops), 'n_invalid_crops': n_invalid, 'boxes_raw_px': boxes,
                         'strict': strict, 'lenient': lenient})
            print(stem[:3], q['text'], 'invalid', n_invalid, '| n_gt', strict['n_gt'], 'tp', lenient['tp'], 'fp', lenient['fp'],
                  'success', strict['success'], flush=True)
    manifest['finished'] = R.now()
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    (out / 'calls.json').write_text(json.dumps(calls, indent=1, ensure_ascii=False))
    (out / 'rows.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print('written', out)


if __name__ == '__main__':
    sys.exit(main())
