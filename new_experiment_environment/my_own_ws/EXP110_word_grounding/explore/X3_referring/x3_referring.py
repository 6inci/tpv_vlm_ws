#!/usr/bin/env python3
"""EXP110-X3 (exploratory, dev only): can the system go to ONE specific object from a referring phrase?

Queries/referents: explore/X3_referring/x3_queries_draft.json (Claude draft; referents pending user visual check).
Conditions (new calls, same prompts/settings as EXP110 config.json):
  C1      : Qwen alone, c1_prompt with the phrase -> boxes; first box = first in Qwen's output order
  C6-dino : Grounding DINO gets the phrase -> candidates (c6.variants.dino filter) -> Qwen picks ids;
            first box = first id in Qwen's raw answer order
Score (target-reaching): success = valid answer and the first box's CENTER lies inside a referent box.
  absent referent: success = valid answer with no box.  primary = main referent only; lenient = + also_accept_*.
  'only' = valid, >=1 box, and every returned box's center lies inside a (lenient) referent box.
Coordinates: raw distorted pixels 1920x1080 (answer keys and model boxes alike). Only dev images are loaded.
Usage: <qwen_vl python> explore/X3_referring/x3_referring.py --out <new dir>   |   --selftest
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import score as S  # noqa: E402

XQ = json.loads((HERE / 'x3_queries_draft.json').read_text())
REG = json.loads((HERE / 'registry_id_to_box_dev.json').read_text())


def movable(rule, gt):
    sp = [o['box_xyxy'] for o in gt if o['label'] == 'small_pallet']
    bl = [o['box_xyxy'] for o in gt if o['label'] == 'block']
    n = [sum(S.center_in(b, p) for b in bl) for p in sp]
    if rule == 'small_pallet containing >=1 block center':
        return [p for p, k in zip(sp, n) if k >= 1]
    if rule == 'small_pallet containing no block center':
        return [p for p, k in zip(sp, n) if k == 0]
    raise ValueError(rule)


def referents(ref, layout, gt):
    """-> (primary boxes, lenient boxes); ([], []) means 'absent'."""
    if ref.get('absent'):
        return [], []
    if 'fixed_id' in ref:
        prim = [REG['images'][layout][str(ref['fixed_id'])]['box']]
    else:
        prim = movable(ref['movable_rule'], gt)
    extra = [REG['images'][layout][str(i)]['box'] for i in ref.get('also_accept_fixed_ids', [])]
    if 'also_accept_movable_rule' in ref:
        extra += movable(ref['also_accept_movable_rule'], gt)
    assert prim, (ref, layout)
    return prim, prim + extra


def judge(boxes, valid, refs):
    """boxes in action order (None if invalid)."""
    if not valid:
        return {'first_hit': False, 'only': False}
    if not refs:  # absent
        return {'first_hit': not boxes, 'only': not boxes}
    hit = lambda b: any(S.center_in(b, r) for r in refs)
    return {'first_hit': bool(boxes) and hit(boxes[0]), 'only': bool(boxes) and all(hit(b) for b in boxes)}


def raw_order(raw, chosen):
    import run_models as R
    seen, out = set(), []
    for i in json.loads(R.strip_fence(raw))['ids']:
        if i in chosen and i not in seen:
            seen.add(i); out.append(i)
    return out


def draw_referents(img, layout, gt, path):
    from PIL import ImageDraw, ImageFont
    im = img.copy(); d = ImageDraw.Draw(im); f = ImageFont.load_default(size=20)
    for q in XQ['queries']:
        prim, len_ = referents(q['referent'], layout, gt)
        for b, col in [(b, (0, 255, 0)) for b in prim] + [(b, (255, 200, 0)) for b in len_[len(prim):]]:
            d.rectangle(b, outline=col, width=3)
            d.text((b[0] + 3, b[1] + 2), q['rid'], fill=col, font=f, stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((10, 10), f'{layout} X3 referents: green = primary, yellow = also accepted. r12 = absent (no box)', fill=(255, 255, 255),
           font=f, stroke_width=2, stroke_fill=(0, 0, 0))
    im.save(path, quality=90)


def selftest():
    r = [[0, 0, 10, 10]]
    assert judge([[2, 2, 4, 4], [50, 50, 60, 60]], True, r) == {'first_hit': True, 'only': False}
    assert judge([[50, 50, 60, 60], [2, 2, 4, 4]], True, r)['first_hit'] is False
    assert judge([], True, []) == {'first_hit': True, 'only': True} and judge([], False, [])['first_hit'] is False
    assert judge([[2, 2, 4, 4]], False, r)['first_hit'] is False
    assert raw_order('```json\n{"ids": [5, 2, 5]}\n```', [2, 5]) == [5, 2]
    gt = [{'label': 'small_pallet', 'box_xyxy': [0, 0, 50, 50]}, {'label': 'small_pallet', 'box_xyxy': [100, 0, 150, 50]},
          {'label': 'block', 'box_xyxy': [10, 10, 20, 20]}]
    assert movable('small_pallet containing >=1 block center', gt) == [[0, 0, 50, 50]]
    assert movable('small_pallet containing no block center', gt) == [[100, 0, 150, 50]]
    print('selftest PASS')


def summary(run_dir):
    """Markdown tables recomputed from <run_dir>/rows.json only."""
    rows = json.loads((Path(run_dir) / 'rows.json').read_text())
    L = [f'# X3 summary ({Path(run_dir).name}, dev P01-P04, recomputed from rows.json)', '',
         '| cond | lang | n | invalid | first box on referent (primary) | (lenient) | only referent boxes (lenient) |', '|---|---|---|---|---|---|---|']
    for cond in ('C1', 'C6-dino'):
        for lang in ('en', 'ko'):
            rs = [r for r in rows if r['cond'] == cond and r['lang'] == lang]
            k = lambda f: f"{sum(f(r) for r in rs)}/{len(rs)}"
            L.append(f"| {cond} | {lang} | {len(rs)} | {k(lambda r: not r['valid'])} | {k(lambda r: r['primary']['first_hit'])} | "
                     f"{k(lambda r: r['lenient']['first_hit'])} | {k(lambda r: r['lenient']['only'])} |")
    L += ['', 'Per query, first box on referent (lenient), images P01-P04 as 1/0, "x" = invalid:', '',
          '| rid | text (en) | C1 en | C6 en | C1 ko | C6 ko |', '|---|---|---|---|---|---|']
    for rid in sorted({r['rid'] for r in rows}):
        cell = lambda c, l: ''.join('x' if not r['valid'] else str(int(r['lenient']['first_hit']))
                                    for r in sorted([r for r in rows if r['rid'] == rid and r['cond'] == c and r['lang'] == l], key=lambda r: r['layout']))
        t = next(r['text'] for r in rows if r['rid'] == rid and r['lang'] == 'en')
        L.append(f"| {rid} | {t} | {cell('C1', 'en')} | {cell('C6-dino', 'en')} | {cell('C1', 'ko')} | {cell('C6-dino', 'ko')} |")
    L += ['', 'C6-dino failures with a referent (lenient), where they happen:', '',
          '| lang | failures | referent not in DINO top-100 | in DINO but filtered out of candidates | in candidates, Qwen chose otherwise / invalid |', '|---|---|---|---|---|']
    for lang in ('en', 'ko'):
        f = [r for r in rows if r['cond'] == 'C6-dino' and r['lang'] == lang and r['referent_lenient'] and not r['lenient']['first_hit']]
        L.append(f"| {lang} | {len(f)} | {sum(not r['dino_raw_has_referent'] for r in f)} | "
                 f"{sum(r['dino_raw_has_referent'] and not r['candidates_have_referent'] for r in f)} | {sum(bool(r['candidates_have_referent']) for r in f)} |")
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    ap.add_argument('--summary', help='existing run dir: write SUMMARY.md from rows.json')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.summary:
        md = summary(a.summary)
        (Path(a.summary) / 'SUMMARY.md').write_text(md)
        return print(md)
    import run_models as R
    from PIL import Image
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    vcfg = R.CFG['c6']['variants']['dino']
    qs = [{'id': f"{q['rid']}_{lang}", 'rid': q['rid'], 'lang': lang, 'text': q[lang]} for q in XQ['queries'] for lang in ('en', 'ko')]
    files = ['explore/X3_referring/x3_referring.py', 'explore/X3_referring/x3_queries_draft.json',
             'explore/X3_referring/registry_id_to_box_dev.json', 'tools/run_models.py', 'tools/score.py', 'config.json']
    man = {'id': XQ['id'], 'started': R.now(), 'argv': sys.argv, 'env': R.env_info(), 'c6_variant': {'dino': vcfg},
           'nms_iou': R.CFG['c6']['nms_iou'], 'prompts': {'C1': R.CFG['qwen']['c1_prompt'], 'C6': R.CFG['qwen']['c6_prompt']},
           'max_new_tokens': {'C1': R.CFG['qwen']['c1_max_new_tokens'], 'C6': R.CFG['qwen']['c6_max_new_tokens']},
           'sha256': {p: R.sha256(ROOT / p) for p in files}, 'image_space': R.CFG['image_space']}
    calls, rows = [], []
    dino_all = {}
    for img_path in R.images_for('dev'):  # pass 1: DINO only (Qwen 8.9 GiB + DINO 2.0 GiB may not fit in 12 GB together)
        o2 = {}
        R.run_c2(Image.open(img_path).convert('RGB'), qs, o2)
        dino_all[img_path.stem] = {r['qid']: r for r in o2['results']}
    del R.MODELS['dino']
    import torch
    torch.cuda.empty_cache()
    for img_path in R.images_for('dev'):  # pass 2: Qwen (C1, C6-dino)
        stem, layout = img_path.stem, img_path.stem[:3]
        gt = S.load_gt(stem)
        assert gt['image_sha256'] == R.sha256(img_path)
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        draw_referents(img, layout, gt['objects'], out / f'referents_check_{layout}.jpg')
        o1 = {}
        R.run_c1(img, qs, o1)
        c1 = {r['qid']: r for r in o1['results']}
        dino = dino_all[stem]
        qwen = R.get_qwen()
        for q in qs:
            ref = next(x for x in XQ['queries'] if x['rid'] == q['rid'])['referent']
            prim, len_ = referents(ref, layout, gt['objects'])
            r1 = c1[q['id']]
            b1 = [p['box'] for p in r1['preds']] if r1['valid'] else None
            cands = R.candidates(dino[q['id']]['preds'], vcfg, W, H)
            if cands:
                raw, ntok, nin, dt = qwen.ask(R.overlay(img, cands), R.CFG['qwen']['c6_prompt'].format(q=q['text']),
                                              R.CFG['qwen']['c6_max_new_tokens'])
                ids, err = R.parse_c6(raw, {c['id'] for c in cands})
                order = raw_order(raw, ids) if err is None else []
            else:
                raw, ntok, nin, dt, ids, err, order = None, 0, 0, 0.0, [], None, []
            cmap = {c['id']: c['box'] for c in cands}
            b6 = [cmap[i] for i in order] if err is None else None
            dino_any = any(S.center_in(p['box'], r) for p in dino[q['id']]['preds'] for r in len_) if len_ else None
            cand_any = any(S.center_in(c['box'], r) for c in cands for r in len_) if len_ else None
            calls.append({'layout': layout, 'qid': q['id'], 'text': q['text'], 'C1': r1, 'DINO_preds': dino[q['id']]['preds'],
                          'C6': {'candidates': cands, 'raw': raw, 'error': err, 'chosen_ids': ids or [], 'raw_order': order,
                                 'generated_tokens': ntok, 'input_tokens': nin, 'latency_s': round(dt, 3)}})
            for cond, boxes, valid in (('C1', b1, r1['valid']), ('C6-dino', b6, err is None)):
                rows.append({'layout': layout, 'rid': q['rid'], 'lang': q['lang'], 'text': q['text'], 'cond': cond,
                             'valid': valid, 'n_boxes': None if boxes is None else len(boxes),
                             'first_box': boxes[0] if boxes else None, 'referent_primary': prim, 'referent_lenient': len_,
                             'primary': judge(boxes, valid, prim), 'lenient': judge(boxes, valid, len_),
                             'dino_raw_has_referent': dino_any, 'candidates_have_referent': cand_any if cond == 'C6-dino' else None})
                rr = rows[-1]
                print(layout, q['id'], cond, 'valid', valid, 'n', rr['n_boxes'], 'first_hit', rr['primary']['first_hit'],
                      '/', rr['lenient']['first_hit'], flush=True)
    man['finished'] = R.now()
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1, ensure_ascii=False))
    (out / 'calls.json').write_text(json.dumps(calls, indent=1, ensure_ascii=False))
    (out / 'rows.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print('written', out)


if __name__ == '__main__':
    sys.exit(main())
