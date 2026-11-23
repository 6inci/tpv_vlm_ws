#!/usr/bin/env python3
"""EXP110-X6 (exploratory, dev only): smoke test for the model-size comparison M1 / M2 / M3 (plan v3 §8.4).

  M1  Qwen3-VL-4B  bf16, GPU                 (env qwen_vl)
  M2  Qwen3-VL-8B  bf16, GPU + CPU offload   (env qwen_vl, accelerate device_map)
  M3  Qwen3-VL-8B  4-bit NF4 (bitsandbytes)  (env qwen_vl_q4 only)
Purpose: does the model load and answer, how fast, how much memory. NOT a performance comparison
(few queries, one image). Same prompts / max_new_tokens / candidate filter as EXP110 config.json; C6 candidates
come from the saved run02 Grounding DINO outputs (C2), so every model sees the identical overlay.
Boxes are raw distorted pixels 1920x1080. Raw model text of every call is saved.
Usage: <python> explore/X6_model_size/x6_smoke.py --model M1|M2|M3 --out <new dir>
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
XCFG = json.loads((HERE / 'config.json').read_text())


def load(model_key):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    m = XCFG['models'][model_key]
    kw = dict(local_files_only=True, revision=m.get('revision'))
    proc = AutoProcessor.from_pretrained(m['id'], **kw)
    if m['mode'] == 'bf16_gpu':
        model = Qwen3VLForConditionalGeneration.from_pretrained(m['id'], dtype=torch.bfloat16, **kw).to('cuda')
    elif m['mode'] == 'bf16_offload':
        model = Qwen3VLForConditionalGeneration.from_pretrained(m['id'], dtype=torch.bfloat16, device_map='auto',
                                                                max_memory={int(k) if k.isdigit() else k: v for k, v in m['max_memory'].items()}, **kw)
    elif m['mode'] == 'nf4':
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=m.get('double_quant', False))
        model = Qwen3VLForConditionalGeneration.from_pretrained(m['id'], quantization_config=q, device_map={'': 0}, **kw)
    else:
        raise ValueError(m['mode'])
    return proc, model.eval()


def ask(proc, model, img, text, max_new_tokens):
    import torch
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': text}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors='pt').to('cuda')
    t = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    dt = time.perf_counter() - t
    gen = out[0][inp['input_ids'].shape[-1]:]
    return proc.decode(gen, skip_special_tokens=True), int(gen.shape[-1]), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(XCFG['models']))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch, psutil
    import run_models as R
    import score as S
    from PIL import Image
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    layout = XCFG['layout']
    assert layout in R.CFG['splits']['dev'], 'dev images only'
    img_path = next(p for p in R.images_for('dev') if p.stem.startswith(layout))
    img = Image.open(img_path).convert('RGB')
    W, H = img.size
    gt = S.load_gt(img_path.stem)
    qmap = {q['id']: q for q in R.QUERIES}
    c2 = {r['qid']: r for r in json.loads((ROOT / 'results' / XCFG['candidates_run'] / 'C2' / f'{img_path.stem}.json').read_text())['results']}
    vcfg = R.CFG['c6']['variants']['dino']
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    proc, model = load(a.model)
    load_s = time.perf_counter() - t0
    dev_map = getattr(model, 'hf_device_map', None)
    rows = []
    for qid in XCFG['queries']:
        q = qmap[qid]
        raw1, n1, dt1 = ask(proc, model, img, R.CFG['qwen']['c1_prompt'].format(q=q['text']), R.CFG['qwen']['c1_max_new_tokens'])
        p1, e1 = R.parse_c1(raw1, W, H)
        cands = R.candidates(c2[qid]['preds'], vcfg, W, H)
        raw6, n6, dt6, ids, e6 = None, 0, 0.0, [], None
        if cands:
            raw6, n6, dt6 = ask(proc, model, R.overlay(img, cands), R.CFG['qwen']['c6_prompt'].format(q=q['text']), R.CFG['qwen']['c6_max_new_tokens'])
            ids, e6 = R.parse_c6(raw6, {c['id'] for c in cands})
        b6 = [c['box'] for c in cands if ids and c['id'] in ids]
        s1 = S.score_one([p['box'] for p in p1 or []], e1 is None, gt['objects'], q)
        s6 = S.score_one(b6, e6 is None, gt['objects'], q)
        rows.append({'qid': qid, 'text': q['text'], 'C1': {'raw': raw1, 'error': e1, 'gen_tokens': n1, 'latency_s': round(dt1, 2),
                     'success': s1['success'], 'tp': s1['tp'], 'fp': s1['fp'], 'n_gt': s1['n_gt']},
                     'C6': {'raw': raw6, 'error': e6, 'chosen_ids': ids or [], 'gen_tokens': n6, 'latency_s': round(dt6, 2),
                            'success': s6['success'], 'tp': s6['tp'], 'fp': s6['fp'], 'n_gt': s6['n_gt']}})
        print(a.model, qid, q['text'], '| C1', s1['success'], f'{dt1:.1f}s', '| C6', s6['success'], f'{dt6:.1f}s', flush=True)
    man = {'id': XCFG['id'], 'model': a.model, 'model_cfg': XCFG['models'][a.model], 'python': sys.executable,
           'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'load_s': round(load_s, 1),
           'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3),
           'process_rss_gib': round(psutil.Process().memory_info().rss / 2**30, 2),
           'device_map_summary': None if not dev_map else {d: sum(1 for v in dev_map.values() if v == d) for d in set(dev_map.values())},
           'image': str(img_path), 'image_sha256': R.sha256(img_path), 'started': R.now(),
           'sha256': {p: R.sha256(ROOT / p) for p in ['explore/X6_model_size/x6_smoke.py', 'explore/X6_model_size/config.json',
                                                       'tools/run_models.py', 'config.json']}}
    try:
        import bitsandbytes
        man['bitsandbytes'] = bitsandbytes.__version__
    except ImportError:
        pass
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1, ensure_ascii=False, default=str))
    (out / 'rows.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print('load', man['load_s'], 's | peak VRAM', man['peak_vram_gib'], 'GiB | RSS', man['process_rss_gib'], 'GiB | written', out)


if __name__ == '__main__':
    sys.exit(main())
