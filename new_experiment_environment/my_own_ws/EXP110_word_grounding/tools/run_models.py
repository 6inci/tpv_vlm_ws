#!/usr/bin/env python3
"""EXP110 model runner: one condition per call -> results/<run>/<cond>/<image stem>.json (never overwrites; resumes).

qwen_vl env   : C1 (Qwen alone) | C2 (Grounding DINO tiny) | C5B / C5L (Florence-2 base / large OVD)
                C6 --variants pf,dino,yoloe,yoloworld,florence (Qwen picks numbered detector candidates)
yolo_world env: C3 (YOLOE-26s text) | C4 (YOLO-World s) | PF (YOLOE-26s prompt-free proposals, used by C6-pf)
Usage: <env python> tools/run_models.py --run run01 --split dev --cond C2 [--limit-queries N] [--out-root DIR]
Boxes are [x1, y1, x2, y2] in raw image pixels (config.json image_space). Detectors only ever get the query string.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config.json').read_text())
QUERIES = json.loads((ROOT / 'queries.json').read_text())['queries']
DET = CFG['detector']
SNAP_FILES = ['config.json', 'queries.json', 'tools/run_models.py', 'tools/make_queries.py']


# ---------------- bookkeeping ----------------
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).astimezone().isoformat()


def images_for(split):
    files = []
    for layout in CFG['splits'][split]:
        m = sorted((ROOT / 'data' / 'pilot').glob(f'{layout}_*.jpg'))
        if len(m) != 1:
            raise SystemExit(f'expected exactly one image for {layout}, found {m}')
        files.append(m[0])
    return files


def snapshot(run_dir):
    """Copy code/config/queries/registry once per run; refuse to continue a run whose inputs changed (no git)."""
    srcs = {f: ROOT / f for f in SNAP_FILES}
    srcs['registry.json'] = Path(CFG['registry'])
    shas = {k: sha256(v) for k, v in srcs.items()}
    snap = run_dir / '_snapshot'
    if snap.exists():
        old = json.loads((snap / 'SHA256.json').read_text())
        if old != shas:
            changed = [k for k in shas if old.get(k) != shas[k]]
            raise SystemExit(f'{changed} changed since this run started -> start a new --run id')
        return
    for k, v in srcs.items():
        (snap / k).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(v, snap / k)
    (snap / 'SHA256.json').write_text(json.dumps(shas, indent=1))
    (snap / 'DESIGN.sha256').write_text(sha256(ROOT / 'DESIGN.md') + '  DESIGN.md (informational)\n')


def env_info():
    import torch
    info = {'python': sys.executable, 'python_version': platform.python_version(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}
    for mod in ('transformers', 'ultralytics'):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            pass
    info['nvidia_driver'] = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                           capture_output=True, text=True).stdout.strip()
    return info


def peak_gib():
    import torch
    return round(torch.cuda.max_memory_allocated() / 2**30, 3)


def keep(preds):
    preds = [p for p in preds if p['score'] is None or p['score'] >= DET['save_min_score']]
    preds.sort(key=lambda p: -(p['score'] or 0))
    return preds[:DET['save_max_boxes']]


def rbox(b):
    return [round(float(v), 1) for v in b]


# ---------------- Qwen ----------------
class Qwen:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch, mid = torch, CFG['models']['qwen']
        self.proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(mid, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()

    def ask(self, img, text, max_new_tokens):
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': text}]}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors='pt').to('cuda')
        t = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        dt = time.perf_counter() - t
        gen = out[0][inp['input_ids'].shape[-1]:]
        return self.proc.decode(gen, skip_special_tokens=True), int(gen.shape[-1]), int(inp['input_ids'].shape[-1]), dt


def strip_fence(raw):
    t = raw.strip()
    if t.startswith('```'):
        t = t.split('\n', 1)[1].rsplit('```', 1)[0]
    return t


def parse_c1(raw, w, h):
    """Strict: any malformed item makes the whole answer invalid (counted as failure, raw kept)."""
    try:
        data = json.loads(strip_fence(raw))
    except json.JSONDecodeError as e:
        return None, f'JSONDecodeError: {e}'
    if not isinstance(data, dict) or not isinstance(data.get('objects'), list):
        return None, 'schema: expected {"objects": [...]}'
    preds = []
    for i, o in enumerate(data['objects']):
        b = o.get('bbox_2d') if isinstance(o, dict) else None
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in b)):
            return None, f'schema: objects[{i}].bbox_2d is not 4 numbers'
        if not all(0 <= v <= 1000 for v in b) or not (b[0] < b[2] and b[1] < b[3]):
            return None, f'schema: objects[{i}].bbox_2d invalid {b}'
        preds.append({'box': rbox([b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h]), 'score': None,
                      'bbox_norm1000': b})
    return preds, None


def parse_c6(raw, valid_ids):
    try:
        data = json.loads(strip_fence(raw))
    except json.JSONDecodeError as e:
        return None, f'JSONDecodeError: {e}'
    ids = data.get('ids') if isinstance(data, dict) else None
    if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        return None, 'schema: expected {"ids": [int, ...]}'
    bad = [i for i in ids if i not in valid_ids]
    if bad:
        return None, f'ids not among candidates: {bad}'
    return sorted(set(ids)), None


# ---------------- C6 candidates ----------------
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def candidates(preds, vcfg, w, h):
    ps = [p for p in preds if (vcfg['min_score'] is None or (p['score'] is not None and p['score'] >= vcfg['min_score']))
          and (p['box'][2] - p['box'][0]) * (p['box'][3] - p['box'][1]) <= vcfg['max_area_frac'] * w * h]
    ps.sort(key=lambda p: -(p['score'] or 0))
    kept = []
    for p in ps:  # greedy NMS
        if all(iou(p['box'], k['box']) < CFG['c6']['nms_iou'] for k in kept):
            kept.append(p)
        if len(kept) == vcfg['top_k']:
            break
    return [{'id': i + 1, 'box': p['box'], 'score': p['score']} for i, p in enumerate(kept)]


def overlay(img, cands):
    from PIL import ImageDraw, ImageFont
    pal = [(255, 64, 64), (64, 200, 255), (255, 210, 0), (0, 230, 120), (255, 120, 255), (255, 150, 40), (160, 120, 255)]
    im = img.copy()
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=22)
    for c in cands:
        col = pal[(c['id'] - 1) % len(pal)]
        x1, y1, x2, y2 = c['box']
        d.rectangle([x1, y1, x2, y2], outline=col, width=3)
        t = str(c['id'])
        tb = d.textbbox((0, 0), t, font=font)
        tx, ty = x1, max(0, y1 - (tb[3] - tb[1]) - 6)
        d.rectangle([tx, ty, tx + tb[2] - tb[0] + 6, ty + tb[3] - tb[1] + 6], fill=(0, 0, 0))
        d.text((tx + 3, ty + 3 - tb[1]), t, fill=col, font=font)
    return im


# ---------------- conditions ----------------
def get_qwen():
    if 'qwen' not in MODELS:
        MODELS['qwen'] = Qwen()
    return MODELS['qwen']


def run_c1(img, qs, out):
    q_ = get_qwen()
    w, h = img.size
    res = []
    for q in qs:
        raw, ntok, nin, dt = q_.ask(img, CFG['qwen']['c1_prompt'].format(q=q['text']), CFG['qwen']['c1_max_new_tokens'])
        preds, err = parse_c1(raw, w, h)
        res.append({'qid': q['id'], 'lang': q['lang'], 'text': q['text'], 'valid': err is None, 'error': err,
                    'preds': preds or [], 'raw': raw, 'generated_tokens': ntok, 'input_tokens': nin,
                    'hit_max_new_tokens': ntok >= CFG['qwen']['c1_max_new_tokens'], 'latency_s': round(dt, 3)})
    out['model'], out['results'] = CFG['models']['qwen'], res


def run_c2(img, qs, out):
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    if 'dino' not in MODELS:
        mid = CFG['models']['dino']
        MODELS['dino'] = (AutoProcessor.from_pretrained(mid, local_files_only=True),
                          AutoModelForZeroShotObjectDetection.from_pretrained(mid, local_files_only=True).to('cuda').eval())
    proc, model = MODELS['dino']
    w, h = img.size
    res = []
    for q in qs:
        text = q['text'].lower().strip().rstrip('.') + '.'
        inputs = proc(images=img, text=text, return_tensors='pt').to('cuda')
        t = time.perf_counter()
        with torch.inference_mode():
            o = model(**inputs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        r = proc.post_process_grounded_object_detection(o, inputs.input_ids, threshold=DET['dino_box_threshold'],
                                                        text_threshold=DET['dino_text_threshold'], target_sizes=[(h, w)])[0]
        labels = r.get('text_labels', r.get('labels'))
        preds = keep([{'box': rbox(b), 'score': round(float(s), 4), 'phrase': l} for b, s, l in zip(r['boxes'], r['scores'], labels)])
        res.append({'qid': q['id'], 'lang': q['lang'], 'text': q['text'], 'prompt': text, 'preds': preds, 'latency_s': round(dt, 4)})
    out['model'], out['results'] = CFG['models']['dino'], res
    out['params'] = {k: DET[k] for k in ('dino_box_threshold', 'dino_text_threshold', 'save_min_score', 'save_max_boxes')}


def run_c5(img, qs, out, key):
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration
    if key not in MODELS:
        mid = CFG['models'][key]
        MODELS[key] = (AutoProcessor.from_pretrained(mid, local_files_only=True),
                       Florence2ForConditionalGeneration.from_pretrained(mid, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval())
    proc, model = MODELS[key]
    task = '<OPEN_VOCABULARY_DETECTION>'
    res = []
    for q in qs:
        inp = proc(text=task + q['text'], images=img, return_tensors='pt').to('cuda', torch.bfloat16)
        t = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**inp, max_new_tokens=DET['florence_max_new_tokens'], num_beams=DET['florence_num_beams'], do_sample=False)
        dt = time.perf_counter() - t
        raw = proc.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = proc.post_process_generation(raw, task=task, image_size=img.size)[task]
        preds = [{'box': rbox(b), 'score': None} for b in parsed.get('bboxes', [])]
        res.append({'qid': q['id'], 'lang': q['lang'], 'text': q['text'], 'preds': preds, 'raw': raw, 'latency_s': round(dt, 4)})
    out['model'], out['results'] = CFG['models'][key], res
    out['params'] = {'task': task, 'num_beams': DET['florence_num_beams'], 'max_new_tokens': DET['florence_max_new_tokens']}


def yolo_kwargs(max_det):
    return dict(device=0, conf=DET['save_min_score'], iou=DET['yolo_iou'], imgsz=DET['yolo_imgsz'], max_det=max_det, verbose=False)


def yolo_preds(r):
    return keep([{'box': rbox(b), 'score': round(float(s), 4)} for b, s in zip(r.boxes.xyxy, r.boxes.conf)])


def run_c3(img, qs, out):
    if 'yoloe' not in MODELS:
        os.chdir(CFG['models']['yoloe_dir'])  # ultralytics resolves mobileclip2_b.ts relative to cwd
        from ultralytics import YOLOE
        m = YOLOE(CFG['models']['yoloe_text'])
        texts = [q['text'] for q in QUERIES]
        pe = m.get_text_pe(texts)  # encode every query once, (1, n, d)
        assert pe.shape[1] == len(texts), pe.shape
        MODELS['yoloe'] = (m, {t: pe[:, i:i + 1] for i, t in enumerate(texts)})
    m, pes = MODELS['yoloe']
    res = []
    for q in qs:
        m.set_classes([q['text']], pes[q['text']])
        t = time.perf_counter()
        r = m.predict(img, **yolo_kwargs(DET['save_max_boxes']))[0]
        res.append({'qid': q['id'], 'lang': q['lang'], 'text': q['text'], 'preds': yolo_preds(r), 'latency_s': round(time.perf_counter() - t, 4)})
    out['model'], out['results'] = CFG['models']['yoloe_text'], res
    out['params'] = yolo_kwargs(DET['save_max_boxes'])


def run_c4(img, qs, out):
    if 'yw' not in MODELS:
        import ultralytics.nn.text_model as tm
        from ultralytics import YOLOWorld
        orig = tm.clip.load
        tm.clip.load = lambda name, *a, **k: orig(name, *a, **dict(k, download_root=CFG['models']['yolo_world_clip_dir']))
        MODELS['yw'] = YOLOWorld(CFG['models']['yolo_world'])
    m = MODELS['yw']
    res = []
    for q in qs:
        m.set_classes([q['text']])
        t = time.perf_counter()
        r = m.predict(img, **yolo_kwargs(DET['save_max_boxes']))[0]
        res.append({'qid': q['id'], 'lang': q['lang'], 'text': q['text'], 'preds': yolo_preds(r), 'latency_s': round(time.perf_counter() - t, 4)})
    out['model'], out['results'] = CFG['models']['yolo_world'], res
    out['params'] = yolo_kwargs(DET['save_max_boxes'])


def run_pf(img, qs, out):
    if 'pf' not in MODELS:
        from ultralytics import YOLOE
        MODELS['pf'] = YOLOE(CFG['models']['yoloe_pf'])
    m = MODELS['pf']
    t = time.perf_counter()
    r = m.predict(img, **yolo_kwargs(300))[0]
    props = [{'box': rbox(b), 'score': round(float(s), 4), 'pf_label': r.names[int(c)]}
             for b, s, c in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)]
    out['model'], out['params'] = CFG['models']['yoloe_pf'], yolo_kwargs(300)
    out['proposals'], out['latency_s'] = sorted(props, key=lambda p: -p['score']), round(time.perf_counter() - t, 4)


def run_c6(img, qs, out, variant, run_dir, stem):
    q_ = get_qwen()
    vcfg = CFG['c6']['variants'][variant]
    src = json.loads((run_dir / vcfg['source'] / f'{stem}.json').read_text())
    w, h = img.size
    shared = None
    if vcfg['source'] == 'PF':  # query-agnostic candidates: same overlay for every query
        shared = candidates(src['proposals'], vcfg, w, h)
        overlay(img, shared).save(run_dir / f'C6-{variant}' / f'{stem}_overlay.jpg', quality=92)
    by_q = {r['qid']: r['preds'] for r in src.get('results', [])}
    res = []
    for q in qs:
        cands = shared if shared is not None else candidates(by_q[q['id']], vcfg, w, h)
        if not cands:  # nothing to choose from -> empty prediction, no Qwen call
            res.append({'qid': q['id'], 'lang': q['lang'], 'text': q['text'], 'candidates': [], 'valid': True, 'error': None,
                        'chosen_ids': [], 'preds': [], 'raw': None, 'latency_s': 0.0})
            continue
        raw, ntok, nin, dt = q_.ask(overlay(img, cands), CFG['qwen']['c6_prompt'].format(q=q['text']), CFG['qwen']['c6_max_new_tokens'])
        ids, err = parse_c6(raw, {c['id'] for c in cands})
        chosen = [c for c in cands if ids and c['id'] in ids]
        res.append({'qid': q['id'], 'lang': q['lang'], 'text': q['text'], 'candidates': None if shared is not None else cands,
                    'valid': err is None, 'error': err, 'chosen_ids': ids or [],
                    'preds': [{'box': c['box'], 'score': c['score']} for c in chosen], 'raw': raw,
                    'generated_tokens': ntok, 'input_tokens': nin, 'latency_s': round(dt, 3)})
    out['model'] = {'selector': CFG['models']['qwen'], 'candidates_from': vcfg['source']}
    out['params'] = {'variant': variant, **vcfg, 'nms_iou': CFG['c6']['nms_iou']}
    if shared is not None:
        out['shared_candidates'] = shared
    out['results'] = res


MODELS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--split', required=True, choices=['dev', 'test'])
    ap.add_argument('--cond', required=True, choices=['C1', 'C2', 'C3', 'C4', 'C5B', 'C5L', 'PF', 'C6'])
    ap.add_argument('--variants', default='pf,dino,yoloe,yoloworld,florence')
    ap.add_argument('--limit-queries', type=int, default=None, help='smoke tests only')
    ap.add_argument('--out-root', default=str(ROOT / CFG['paths']['results']))
    ap.add_argument('--settings-frozen', action='store_true', help='required for --split test (DESIGN §10)')
    a = ap.parse_args()
    if a.split == 'test' and not a.settings_frozen:
        raise SystemExit('test split is sealed until DESIGN is frozen (pass --settings-frozen only then)')
    import torch
    from PIL import Image
    run_dir = Path(a.out_root) / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot(run_dir)
    qs = QUERIES[:a.limit_queries] if a.limit_queries else QUERIES
    conds = [f'C6-{v}' for v in a.variants.split(',')] if a.cond == 'C6' else [a.cond]
    for cond in conds:
        cdir = run_dir / cond
        cdir.mkdir(exist_ok=True)
        man = cdir / 'MANIFEST.json'
        log = json.loads(man.read_text()) if man.exists() else {'condition': cond, 'split': a.split, 'sessions': []}
        log['sessions'].append({'started': now(), 'env': env_info(), 'limit_queries': a.limit_queries, 'argv': sys.argv})
        man.write_text(json.dumps(log, indent=1))
        torch.cuda.reset_peak_memory_stats()
        for img_path in images_for(a.split):
            target = cdir / f'{img_path.stem}.json'
            if target.exists():
                print(cond, img_path.stem, 'exists, skip')
                continue
            img = Image.open(img_path).convert('RGB')
            out = {'experiment_id': CFG['experiment_id'], 'condition': cond, 'split': a.split, 'image': str(img_path),
                   'image_sha256': sha256(img_path), 'image_size': list(img.size), 'image_space': CFG['image_space'],
                   'started': now(), 'n_queries': len(qs)}
            t = time.perf_counter()
            if cond == 'C1':
                run_c1(img, qs, out)
            elif cond == 'C2':
                run_c2(img, qs, out)
            elif cond in ('C5B', 'C5L'):
                run_c5(img, qs, out, 'florence_base' if cond == 'C5B' else 'florence_large')
            elif cond == 'C3':
                run_c3(img, qs, out)
            elif cond == 'C4':
                run_c4(img, qs, out)
            elif cond == 'PF':
                run_pf(img, qs, out)
            else:
                run_c6(img, qs, out, cond.split('-', 1)[1], run_dir, img_path.stem)
            out['finished'], out['wall_s'], out['peak_vram_gib'] = now(), round(time.perf_counter() - t, 2), peak_gib()
            target.write_text(json.dumps(out, indent=1, ensure_ascii=False))
            nv = sum(1 for r in out.get('results', []) if r.get('valid') is False)
            print(cond, img_path.stem, f"{out['wall_s']}s", f'invalid={nv}' if 'results' in out else f"proposals={len(out.get('proposals', []))}", flush=True)
        log['sessions'][-1]['finished'] = now()
        man.write_text(json.dumps(log, indent=1))


def selftest():
    assert parse_c1('```json\n{"objects": [{"bbox_2d": [100, 200, 500, 800]}]}\n```', 2000, 1000)[0][0]['box'] == [200.0, 200.0, 1000.0, 800.0]
    assert parse_c1('{"objects": [{"bbox_2d": [5, 5, 1, 1]}]}', 10, 10)[0] is None
    assert parse_c1('{"objects": []}', 10, 10) == ([], None)
    assert parse_c6('{"ids": [3, 1, 3]}', {1, 2, 3}) == ([1, 3], None) and parse_c6('{"ids": [9]}', {1})[0] is None
    preds = [{'box': [0, 0, 10, 10], 'score': 0.9}, {'box': [1, 1, 10, 10], 'score': 0.8}, {'box': [50, 50, 60, 60], 'score': 0.05},
             {'box': [0, 0, 100, 100], 'score': 0.95}]
    c = candidates(preds, {'min_score': 0.1, 'max_area_frac': 0.25, 'top_k': 5}, 100, 100)
    assert [x['box'] for x in c] == [[0, 0, 10, 10]], c  # NMS drops the overlap, score + area filters drop the rest
    print('selftest PASS')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    else:
        main()
