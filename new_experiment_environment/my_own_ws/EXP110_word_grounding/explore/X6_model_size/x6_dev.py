#!/usr/bin/env python3
"""EXP110-X6 (exploratory, dev only): run C1 and C6-dino with another Qwen model (M2 / M3) on the full EXP110 dev set.

Output files use the run_models.py schema, so `tools/score.py --run <name> --split dev` and X1 score them unchanged:
  results/<name>/C1/<stem>.json, results/<name>/C6-dino/<stem>.json
Everything except the model is the EXP110 dev setting: same images (P01-P04), queries.json (97), prompts,
max_new_tokens, and C6 candidates rebuilt from the saved run02 Grounding DINO outputs with config c6.variants.dino.
Not part of the registered EXP110 comparison (dev only, exploratory). Never runs on test.
Usage: <env python> explore/X6_model_size/x6_dev.py --model M3 --name x6_M3_dev
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(HERE))
from x6_smoke import load, ask, XCFG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(XCFG['models']))
    ap.add_argument('--name', required=True)
    a = ap.parse_args()
    import torch
    import run_models as R
    from PIL import Image
    run_dir = ROOT / 'results' / a.name
    run_dir.mkdir(parents=True, exist_ok=False)
    vcfg = R.CFG['c6']['variants']['dino']
    proc, model = load(a.model)
    man = {'id': 'EXP110-X6-dev', 'model_key': a.model, 'model_cfg': XCFG['models'][a.model], 'split': 'dev', 'started': R.now(),
           'env': R.env_info(), 'candidates_from': 'results/run02/C2 + config c6.variants.dino',
           'sha256': {p: R.sha256(ROOT / p) for p in ['explore/X6_model_size/x6_dev.py', 'explore/X6_model_size/x6_smoke.py',
                                                       'explore/X6_model_size/config.json', 'config.json', 'queries.json', 'tools/run_models.py']}}
    (run_dir / 'MANIFEST.json').write_text(json.dumps(man, indent=1, ensure_ascii=False, default=str))
    torch.cuda.reset_peak_memory_stats()
    for img_path in R.images_for('dev'):
        stem = img_path.stem
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        c2 = {r['qid']: r['preds'] for r in json.loads((ROOT / 'results' / 'run02' / 'C2' / f'{stem}.json').read_text())['results']}
        for cond in ('C1', 'C6-dino'):
            (run_dir / cond).mkdir(exist_ok=True)
            res, t0 = [], time.perf_counter()
            for q in R.QUERIES:
                if cond == 'C1':
                    raw, ntok, dt = ask(proc, model, img, R.CFG['qwen']['c1_prompt'].format(q=q['text']), R.CFG['qwen']['c1_max_new_tokens'])
                    preds, err = R.parse_c1(raw, w, h)
                    res.append({'qid': q['id'], 'lang': q['lang'], 'text': q['text'], 'valid': err is None, 'error': err,
                                'preds': preds or [], 'raw': raw, 'generated_tokens': ntok,
                                'hit_max_new_tokens': ntok >= R.CFG['qwen']['c1_max_new_tokens'], 'latency_s': round(dt, 3)})
                else:
                    cands = R.candidates(c2[q['id']], vcfg, w, h)
                    if not cands:
                        res.append({'qid': q['id'], 'lang': q['lang'], 'text': q['text'], 'candidates': [], 'valid': True, 'error': None,
                                    'chosen_ids': [], 'preds': [], 'raw': None, 'latency_s': 0.0})
                        continue
                    raw, ntok, dt = ask(proc, model, R.overlay(img, cands), R.CFG['qwen']['c6_prompt'].format(q=q['text']), R.CFG['qwen']['c6_max_new_tokens'])
                    ids, err = R.parse_c6(raw, {c['id'] for c in cands})
                    chosen = [c for c in cands if ids and c['id'] in ids]
                    res.append({'qid': q['id'], 'lang': q['lang'], 'text': q['text'], 'candidates': cands, 'valid': err is None,
                                'error': err, 'chosen_ids': ids or [], 'preds': [{'box': c['box'], 'score': c['score']} for c in chosen],
                                'raw': raw, 'generated_tokens': ntok, 'latency_s': round(dt, 3)})
            out = {'experiment_id': 'EXP110-X6', 'condition': cond, 'split': 'dev', 'image': str(img_path), 'image_sha256': R.sha256(img_path),
                   'image_size': [w, h], 'image_space': R.CFG['image_space'], 'model': XCFG['models'][a.model], 'n_queries': len(R.QUERIES),
                   'results': res, 'wall_s': round(time.perf_counter() - t0, 2), 'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3)}
            (run_dir / cond / f'{stem}.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
            print(a.model, cond, stem[:3], f"{out['wall_s']}s", 'invalid', sum(not r['valid'] for r in res), flush=True)
    man['finished'] = R.now()
    (run_dir / 'MANIFEST.json').write_text(json.dumps(man, indent=1, ensure_ascii=False, default=str))
    print('written', run_dir)


if __name__ == '__main__':
    sys.exit(main())
