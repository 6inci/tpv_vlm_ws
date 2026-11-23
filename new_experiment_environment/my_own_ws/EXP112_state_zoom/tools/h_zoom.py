#!/usr/bin/env python3
"""EXP112 H-zoom runner: ask Qwen3-VL-4B about the state of each candidate small pallet under Z-full / Z-crop / Z-both,
plus the B-rule baseline (EXP111 s4: block boxes on the enlarged crop + Python centre rule). Settings: config.json.

Modes
  --smoke : EXP110 dev P01-P04, EXP111 predicted pallets as candidates. Format and GPU memory only:
            no answer key is read, no accuracy is computed, answers are written to files but not printed.
  --dev   : new EXP111 dev D01-D08, frames 0-2, oracle candidates (answer-key pallets of frame 0). One run per condition.
Every call is written to calls.jsonl (raw answer, parsed answer, error, latency, tokens). Scoring: tools/h_zoom_score.py.
Usage: <qwen_vl python> tools/h_zoom.py --smoke --out smoke/<ts>   |   --dev --out runs/dev_<ts>   |   --selftest
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
WS = HERE.parent
sys.path.insert(0, str(WS / 'EXP110_word_grounding' / 'tools'))
sys.path.insert(0, str(WS / 'EXP110_word_grounding' / 'explore' / 'X4_block_zoom'))
sys.path.insert(0, str(WS / 'EXP120_E3a_relation' / 'tools'))
import run_models as R  # noqa: E402
import score as S  # noqa: E402
from x4_block_zoom import oracle_crops, to_raw  # noqa: E402
from e3a_poc import neutral_overlay  # noqa: E402

CFG = json.loads((HERE / 'config.json').read_text())
CONDS = ('Z-full', 'Z-crop', 'Z-both', 'B-rule')
QTEXT = {q['id']: q['text'] for q in R.QUERIES}


def parse_state(raw):
    """-> (sorted list of lower-cased strings, None) or (None, error)."""
    try:
        d = json.loads(R.strip_fence(raw))
    except json.JSONDecodeError as e:
        return None, f'JSONDecodeError: {e}'
    v = d.get('blocks_on_pallet') if isinstance(d, dict) else None
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        return None, 'schema: expected {"blocks_on_pallet": [strings]}'
    return sorted({x.strip().lower() for x in v}), None


def ask_images(qwen, imgs, text, max_new_tokens):
    content = [{'type': 'image', 'image': im} for im in imgs] + [{'type': 'text', 'text': text}]
    inp = qwen.proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True, add_generation_prompt=True,
                                        return_dict=True, return_tensors='pt').to('cuda')
    t = time.perf_counter()
    with qwen.torch.inference_mode():
        out = qwen.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    dt = time.perf_counter() - t
    gen = out[0][inp['input_ids'].shape[-1]:]
    return qwen.proc.decode(gen, skip_special_tokens=True), int(gen.shape[-1]), int(inp['input_ids'].shape[-1]), dt


def enlarged(img, box):
    from PIL import Image
    (crop, scale), = oracle_crops([box], img.width, img.height, CFG['crop'])
    size = (round((crop[2] - crop[0]) * scale), round((crop[3] - crop[1]) * scale))
    return img.crop(tuple(crop)).resize(size, Image.BICUBIC), crop, scale


def b_rule(qwen, enl, crop, scale, box):
    """EXP111 s4 rule on one pallet crop -> (colours, loaded, per-query records)."""
    recs, colors, loaded = [], set(), False
    for qid in CFG['b_rule']['queries']:
        raw, ntok, nin, dt = qwen.ask(enl, R.CFG['qwen']['c1_prompt'].format(q=QTEXT[qid]), R.CFG['qwen']['c1_max_new_tokens'])
        preds, err = R.parse_c1(raw, enl.width, enl.height)
        hits = [to_raw(p['box'], crop, scale) for p in (preds or [])]
        inside = any(S.center_in(h, box) for h in hits)
        loaded |= inside
        for c, cq in CFG['b_rule']['color_queries'].items():
            if qid == cq and inside:
                colors.add(c)
        recs.append({'qid': qid, 'raw': raw, 'error': err, 'n_boxes': len(hits), 'inside': inside, 'latency_s': round(dt, 3),
                     'n_in': nin, 'n_gen': ntok})
    return sorted(colors), loaded, recs


def run(cases, out, qwen):
    """cases: [{'layout', 'frame', 'img', 'cands': [{'id', 'box'}]}]. Writes calls.jsonl; returns per-condition stats."""
    import torch
    stats = {}
    with open(out / 'calls.jsonl', 'w') as f:
        for cond in CONDS:
            torch.cuda.reset_peak_memory_stats()
            n = inval = 0
            lat, nin_all = [], []
            t0 = time.perf_counter()
            for cs in cases:
                full = neutral_overlay(cs['img'], cs['cands'])
                for c in cs['cands']:
                    enl, crop, scale = enlarged(cs['img'], c['box'])
                    rec = {'cond': cond, 'layout': cs['layout'], 'frame': cs['frame'], 'pallet_id': c['id'], 'box': c['box'], 'crop': crop}
                    if cond == 'B-rule':
                        colors, loaded, sub = b_rule(qwen, enl, crop, scale, c['box'])
                        # EXP111 s4: a malformed sub-query only contributes no boxes; the pallet answer stays valid
                        rec.update({'answer': colors if loaded else [], 'loaded': loaded, 'error': None, 'calls': sub,
                                    'sub_query_errors': [f"{s['qid']}: {s['error']}" for s in sub if s['error']],
                                    'latency_s': round(sum(s['latency_s'] for s in sub), 3)})
                        lat.append(rec['latency_s'])
                        nin_all += [s['n_in'] for s in sub]
                    else:
                        where = CFG['where'][cond].format(k=c['id'])
                        imgs = {'Z-full': [full], 'Z-crop': [enl], 'Z-both': [full, enl]}[cond]
                        raw, ntok, nin, dt = ask_images(qwen, imgs, CFG['state_prompt'].format(where=where), CFG['state_max_new_tokens'])
                        ans, err = parse_state(raw)
                        rec.update({'raw': raw, 'answer': ans, 'loaded': bool(ans) if ans is not None else None, 'error': err,
                                    'latency_s': round(dt, 3), 'n_in': nin, 'n_gen': ntok})
                        lat.append(dt)
                        nin_all.append(nin)
                    n += 1
                    inval += rec['error'] is not None or bool(rec.get('sub_query_errors'))
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            stats[cond] = {'pallet_answers': n, 'invalid': inval, 'latency_median_s': round(sorted(lat)[len(lat) // 2], 3) if lat else None,
                           'latency_max_s': round(max(lat), 3) if lat else None, 'input_tokens_max': max(nin_all) if nin_all else None,
                           'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 2 ** 30, 3), 'wall_s': round(time.perf_counter() - t0, 1)}
            print(cond, stats[cond], flush=True)
    return stats


def smoke_cases():
    from PIL import Image
    so = json.loads((HERE / CFG['smoke']['candidates']).read_text())
    cases = []
    for p in R.images_for('dev'):
        lay = p.stem[:3]
        assert so[lay]['image_stem'] == p.stem
        ps = sorted(so[lay]['pallets'], key=lambda q: q['box_raw_px'][0])
        cases.append({'layout': lay, 'frame': 0, 'img': Image.open(p).convert('RGB'),
                      'cands': [{'id': i, 'box': q['box_raw_px']} for i, q in enumerate(ps, 1)]})
    return cases


def dev_cases():
    from PIL import Image
    D = CFG['data']
    cases = []
    for lay in D['layouts']:
        gt = json.loads((HERE / D['boxes'].format(layout=lay)).read_text())
        if gt.get('status') != 'DONE':
            raise SystemExit(f'{lay}: answer key not DONE')
        pal = sorted([o['box_xyxy'] for o in gt['objects'] if o['label'] == 'small_pallet'], key=lambda b: b[0])
        for fr in D['frames']:
            img = Image.open(HERE / D['images'].format(layout=lay, frame=fr)).convert('RGB')
            cases.append({'layout': lay, 'frame': fr, 'img': img, 'cands': [{'id': i, 'box': b} for i, b in enumerate(pal, 1)]})
    return cases


def selftest():
    assert parse_state('{"blocks_on_pallet": ["Red ", "blue"]}') == (['blue', 'red'], None)
    assert parse_state('{"blocks_on_pallet": []}') == ([], None)
    assert parse_state('{"blocks": []}')[0] is None and parse_state('[1]')[0] is None and parse_state('nope')[0] is None
    assert parse_state('```json\n{"blocks_on_pallet": ["red block"]}\n```') == (['red block'], None)  # wrong colour, not invalid
    from PIL import Image
    enl, crop, scale = enlarged(Image.new('RGB', (1920, 1080)), [100, 100, 150, 150])
    assert crop == [45, 45, 205, 205] and enl.size == (768, 768), (crop, enl.size)
    print('selftest PASS')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--smoke', action='store_true')
    g.add_argument('--dev', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=False)
    cases = smoke_cases() if a.smoke else dev_cases()
    man = {'id': 'EXP112-' + ('smoke' if a.smoke else 'dev'), 'started': R.now(), 'env': R.env_info(), 'config': CFG,
           'n_cases': len(cases), 'n_pallet_frames': sum(len(c['cands']) for c in cases),
           'sha256': {p: R.sha256(HERE / p) for p in ['tools/h_zoom.py', 'config.json']}}
    if a.smoke:
        man['note'] = CFG['smoke']['purpose']
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1, ensure_ascii=False, default=str))
    stats = run(cases, out, R.get_qwen())
    man.update({'finished': R.now(), 'stats': stats})
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1, ensure_ascii=False, default=str))
    print('written', out)


if __name__ == '__main__':
    sys.exit(main())
