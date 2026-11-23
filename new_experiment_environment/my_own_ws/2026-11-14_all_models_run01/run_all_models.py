"""Exploratory run (2026-11-14): every installed detection/VLM model on one image.

Usage: <env python> run_all_models.py <stage>
  stages (qwen_vl env):    selftest qwen dino sam2 florence render
  stages (yolo_world env): yolo_world yoloe_pf
All boxes are [x1, y1, x2, y2] in input-screenshot pixels (see config.json). Not for geometry.
"""
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUN = Path(__file__).resolve().parent
CFG = json.loads((RUN / 'config.json').read_text())
RAW = RUN / 'raw'
RAW.mkdir(exist_ok=True)


def load_rgb():
    return Image.open(CFG['image']).convert('RGB')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target(stage):
    path = RAW / f'{stage}.json'
    if path.exists():  # never overwrite results of this run
        print(f'{stage}: {path.name} exists, skip')
        sys.exit(0)
    return path


def meta(stage, model_id, latency_s, peak_vram_gib, extra=None):
    import torch
    m = {'stage': stage, 'model': model_id, 'python': sys.executable, 'torch': torch.__version__,
         'timestamp': datetime.now(timezone.utc).astimezone().isoformat(),
         'input_image': CFG['image'], 'input_sha256': sha256(CFG['image']),
         'coordinate_system': CFG['coordinate_system'],
         'latency_s': round(latency_s, 4), 'peak_vram_gib': round(peak_vram_gib, 3)}
    m.update(extra or {})
    return m


def peak_gib():
    import torch
    return torch.cuda.max_memory_allocated() / 2**30


# ---------------- Qwen ----------------
def parse_qwen(raw, width, height):
    """Raw Qwen text -> (accepted detections in px, rejected items, parse_error). No silent fixes."""
    text = raw.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return [], [], f'JSONDecodeError: {e}'
    items = data.get('objects') if isinstance(data, dict) else data
    if not isinstance(items, list):
        return [], [], 'top-level is not {"objects": [...]} or a list'
    dets, rejected = [], []
    for i, it in enumerate(items):
        b = it.get('bbox_2d') if isinstance(it, dict) else None
        label = it.get('label') if isinstance(it, dict) else None
        reason = None
        if not isinstance(label, str) or not label:
            reason = 'missing label'
        elif not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
            reason = 'bbox_2d not 4 numbers'
        elif not all(0 <= v <= 1000 for v in b):
            reason = 'bbox_2d outside 0-1000'
        elif not (b[0] < b[2] and b[1] < b[3]):
            reason = 'bbox_2d not x1<x2, y1<y2'
        if reason:
            rejected.append({'index': i, 'item': it, 'reason': reason})
            continue
        px = [b[0] / 1000 * width, b[1] / 1000 * height, b[2] / 1000 * width, b[3] / 1000 * height]
        dets.append({'label': label, 'score': None, 'bbox_norm1000': b, 'bbox_px': [round(v, 1) for v in px]})
    return dets, rejected, None


def run_qwen(stage='qwen', prompt_key='prompt', repetition_penalty=1.0):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    out_path = target(stage)
    mid, q = CFG['models']['qwen'], dict(CFG['qwen'])
    q['prompt'] = q[prompt_key]
    img = load_rgb()
    torch.cuda.reset_peak_memory_stats()
    proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(mid, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': q['prompt']}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        model.generate(**inp, max_new_tokens=1)  # warmup
        t = time.perf_counter()
        out = model.generate(**inp, max_new_tokens=q['max_new_tokens'], do_sample=False, repetition_penalty=repetition_penalty)
        dt = time.perf_counter() - t
    gen = out[0][inp['input_ids'].shape[-1]:]
    raw = proc.decode(gen, skip_special_tokens=True)
    (RAW / f'{stage}_raw.txt').write_text(raw)
    dets, rejected, err = parse_qwen(raw, *img.size)
    res = meta(stage, mid, dt, peak_gib(), {'prompt': q['prompt'], 'repetition_penalty': repetition_penalty,'input_tokens': int(inp['input_ids'].shape[-1]),
               'generated_tokens': int(gen.shape[-1]), 'hit_max_new_tokens': int(gen.shape[-1]) >= q['max_new_tokens'],
               'raw_response_file': f'{stage}_raw.txt', 'parse_error': err, 'rejected': rejected})
    res['panels'] = {stage: dets}
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(stage, len(dets), 'accepted', len(rejected), 'rejected', 'parse_error', err, f'{dt:.1f}s')


# ---------------- Grounding DINO ----------------
def run_dino():
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    out_path = target('dino')
    mid, th = CFG['models']['dino'], CFG['thresholds']
    img = load_rgb()
    text = '. '.join(CFG['vocabulary']) + '.'
    torch.cuda.reset_peak_memory_stats()
    proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid, local_files_only=True).to('cuda').eval()
    inputs = proc(images=img, text=text, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        model(**inputs)
        torch.cuda.synchronize(); t = time.perf_counter()
        out = model(**inputs)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
    r = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=th['dino_box'], text_threshold=th['dino_text'],
                                                    target_sizes=[img.size[::-1]])[0]
    labels = r.get('text_labels', r.get('labels'))
    dets = [{'label': l, 'score': round(float(s), 4), 'bbox_px': [round(float(v), 1) for v in b]}
            for b, s, l in zip(r['boxes'], r['scores'], labels)]
    res = meta('dino', mid, dt, peak_gib(), {'text_prompt': text, 'thresholds': {k: th[k] for k in ('dino_box', 'dino_text')}})
    res['panels'] = {'dino': dets}
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print('dino', len(dets), f'{dt*1000:.0f}ms')


# ---------------- SAM2 (prompted by DINO boxes) ----------------
def run_sam2():
    import torch
    from transformers import Sam2Processor, Sam2Model
    src = json.loads((RAW / 'dino.json').read_text())['panels']['dino']
    out_path = target('sam2')
    mid = CFG['models']['sam2']
    img = load_rgb()
    boxes = [d['bbox_px'] for d in src]
    torch.cuda.reset_peak_memory_stats()
    proc = Sam2Processor.from_pretrained(mid, local_files_only=True)
    model = Sam2Model.from_pretrained(mid, local_files_only=True).to('cuda').eval()
    inputs = proc(images=img, input_boxes=[boxes], return_tensors='pt').to('cuda')
    with torch.inference_mode():
        model(pixel_values=inputs['pixel_values'], input_boxes=inputs['input_boxes'], multimask_output=False)
        torch.cuda.synchronize(); t = time.perf_counter()
        out = model(pixel_values=inputs['pixel_values'], input_boxes=inputs['input_boxes'], multimask_output=False)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
    masks = proc.post_process_masks(out.pred_masks.cpu(), inputs['original_sizes'].cpu())[0][:, 0].numpy().astype(np.uint8)
    np.savez_compressed(RAW / 'sam2_masks.npz', masks=masks)
    ious = out.iou_scores.flatten().float().cpu().tolist()
    dets = [dict(d, sam_iou_score=round(ious[i], 4), mask_index=i, mask_area_px=int(masks[i].sum())) for i, d in enumerate(src)]
    res = meta('sam2', mid, dt, peak_gib(), {'prompt_source': 'raw/dino.json boxes', 'mask_file': 'sam2_masks.npz'})
    res['panels'] = {'sam2_on_dino': dets}
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print('sam2', len(dets), 'masks', f'{dt*1000:.0f}ms')


# ---------------- Florence-2 ----------------
def run_florence():
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration
    out_path = target('florence')
    f = CFG['florence']
    img = load_rgb()
    result = {'stage': 'florence', 'models': {}, 'panels': {}, 'raw_generations': {}}
    for key in ('florence_base', 'florence_large'):
        mid = CFG['models'][key]
        torch.cuda.reset_peak_memory_stats()
        proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
        model = Florence2ForConditionalGeneration.from_pretrained(mid, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()

        def gen(task, text_input=''):
            inp = proc(text=task + text_input, images=img, return_tensors='pt').to('cuda', torch.bfloat16)
            with torch.inference_mode():
                ids = model.generate(**inp, max_new_tokens=f['max_new_tokens'], num_beams=f['num_beams'], do_sample=False)
            text = proc.batch_decode(ids, skip_special_tokens=False)[0]
            return text, proc.post_process_generation(text, task=task, image_size=img.size)[task]

        gen('<OD>')  # warmup
        latencies = {}
        for task in f['tasks']:
            name = f"{key}_{task.strip('<>')}"
            prompts = CFG['vocabulary'] if task == '<OPEN_VOCABULARY_DETECTION>' else ['']
            dets, raws = [], {}
            t = time.perf_counter()
            for p in prompts:
                text, parsed = gen(task, p)
                raws[p or task] = text
                labels = parsed.get('labels') or parsed.get('bboxes_labels') or []
                for b, l in zip(parsed.get('bboxes', []), labels):
                    dets.append({'label': l or p, 'score': None, 'bbox_px': [round(float(v), 1) for v in b], 'prompt': p or None})
            latencies[task] = round(time.perf_counter() - t, 4)
            result['panels'][name] = dets
            result['raw_generations'][name] = raws
            print(name, len(dets), f'{latencies[task]:.2f}s')
        result['models'][key] = meta(key, mid, sum(latencies.values()), peak_gib(), {'latency_s_per_task': latencies})
        del model
        torch.cuda.empty_cache()
    result.update({'input_sha256': sha256(CFG['image']), 'coordinate_system': CFG['coordinate_system'], 'florence_config': f})
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))


# ---------------- Ultralytics (yolo_world env) ----------------
def run_yolo_world():
    import torch
    from ultralytics import YOLOWorld
    import ultralytics.nn.text_model as tm
    out_path = target('yolo_world')
    mid, th = CFG['models']['yolo_world'], CFG['thresholds']
    original = tm.clip.load  # reuse installed CLIP, avoid automatic download (same as existing run_experiment.py)

    def local_clip(name, *a, **k):
        k['download_root'] = CFG['models']['yolo_world_clip_dir']
        return original(name, *a, **k)
    tm.clip.load = local_clip
    img = load_rgb()
    model = YOLOWorld(mid)
    model.set_classes(CFG['vocabulary'])
    kw = dict(device=0, conf=th['yolo_conf'], iou=th['yolo_iou'], imgsz=th['yolo_imgsz'], verbose=False)
    model.predict(img, **kw)
    torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    r = model.predict(img, **kw)[0]
    dt = time.perf_counter() - t
    dets = [{'label': r.names[int(c)], 'score': round(float(s), 4), 'bbox_px': [round(float(v), 1) for v in b]}
            for b, s, c in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)]
    res = meta('yolo_world', mid, dt, peak_gib(), {'vocabulary': CFG['vocabulary'], 'predict_kwargs': kw})
    res['panels'] = {'yolo_world': dets}
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print('yolo_world', len(dets), f'{dt*1000:.0f}ms')


def run_yoloe_pf():
    import torch
    from ultralytics import YOLOE
    out_path = target('yoloe_pf')
    mid, th = CFG['models']['yoloe_pf'], CFG['thresholds']
    img = load_rgb()
    model = YOLOE(mid)
    kw = dict(device=0, conf=th['yolo_conf'], iou=th['yolo_iou'], imgsz=th['yolo_imgsz'], verbose=False)
    model.predict(img, **kw)
    torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    r = model.predict(img, **kw)[0]
    dt = time.perf_counter() - t
    polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
    dets = [{'label': r.names[int(c)], 'score': round(float(s), 4), 'bbox_px': [round(float(v), 1) for v in b],
             'polygon_px': None if p is None else np.round(p, 1).tolist()}
            for b, s, c, p in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls, polys)]
    res = meta('yoloe_pf', mid, dt, peak_gib(), {'vocabulary_size': len(r.names), 'predict_kwargs': kw})
    res['panels'] = {'yoloe_pf': dets}
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print('yoloe_pf', len(dets), f'{dt*1000:.0f}ms')


# ---------------- render ----------------
def color(label):
    h = int(hashlib.md5(label.lower().encode()).hexdigest()[:6], 16)
    return ((h >> 16) & 255 | 64, (h >> 8) & 255 | 64, h & 255 | 64)


def draw_panel(img, dets, title, masks=None):
    base = img.convert('RGBA')
    over = Image.new('RGBA', base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    font = ImageFont.load_default(size=22)
    for i, det in enumerate(dets):
        c = color(det['label'])
        if masks is not None and det.get('mask_index') is not None:
            m = Image.fromarray(masks[det['mask_index']] * 110)
            over.paste(Image.new('RGBA', base.size, c + (255,)), (0, 0), m)
        if det.get('polygon_px'):
            d.polygon([tuple(p) for p in det['polygon_px']], fill=c + (90,))
        x1, y1, x2, y2 = det['bbox_px']
        d.rectangle([x1, y1, x2, y2], outline=c + (255,), width=3)
        tag = det['label'] + (f" {det['score']:.2f}" if det.get('score') is not None else '')
        tb = d.textbbox((x1, y1), tag, font=font)
        d.rectangle([tb[0], tb[1] - 2, tb[2] + 4, tb[3] + 2], fill=(0, 0, 0, 190))
        d.text((x1 + 2, y1), tag, fill=c + (255,), font=font)
    head = ImageFont.load_default(size=40)
    d.rectangle([0, 0, base.width, 56], fill=(0, 0, 0, 210))
    d.text((12, 6), f'{title}  (n={len(dets)})', fill=(255, 255, 255, 255), font=head)
    return Image.alpha_composite(base, over).convert('RGB')


def run_render():
    img = load_rgb()
    vis = RUN / 'overlays'
    vis.mkdir(exist_ok=True)
    rows, panels = [], []
    order = ['qwen', 'qwen_attempt02', 'qwen_attempt03', 'dino', 'sam2', 'yolo_world', 'yoloe_pf', 'florence']
    for stage in order:
        p = RAW / f'{stage}.json'
        if not p.exists():
            rows.append((stage, 'MISSING', '', '', '', ''))
            continue
        res = json.loads(p.read_text())
        masks = np.load(RAW / 'sam2_masks.npz')['masks'] if stage == 'sam2' else None
        for name, dets in res['panels'].items():
            if stage == 'florence':
                m = res['models']['florence_large' if name.startswith('florence_large') else 'florence_base']
                lat = m['latency_s_per_task']['<' + name.split('_', 2)[2] + '>']
            else:
                m = res
                lat = m['latency_s']
            panel = draw_panel(img, dets, name, masks)
            panel.save(vis / f'{name}.jpg', quality=90)
            panels.append(panel)
            labels = {}
            for d in dets:
                labels[d['label']] = labels.get(d['label'], 0) + 1
            top = ', '.join(f'{k}×{v}' for k, v in sorted(labels.items(), key=lambda kv: -kv[1])[:8])
            rows.append((name, len(dets), len(labels), f'{lat:.3f}', f"{m['peak_vram_gib']:.2f}", top))
        if stage.startswith('qwen') and res.get('parse_error'):
            # Derived view only: unique (label, bbox) pairs regex-recovered from the truncated raw text. Not a valid model output.
            import re
            raw = (RAW / res['raw_response_file']).read_text()
            pairs = list(dict.fromkeys(re.findall(r'"label":\s*"([^"]+)",\s*"bbox_2d":\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]', raw)))
            W, H = img.size
            sal = [{'label': l, 'score': None, 'bbox_norm1000': [int(v) for v in b],
                    'bbox_px': [int(b[0]) / 1000 * W, int(b[1]) / 1000 * H, int(b[2]) / 1000 * W, int(b[3]) / 1000 * H]} for l, *b in pairs]
            (RUN / 'processed').mkdir(exist_ok=True)
            (RUN / 'processed' / f'{stage}_salvaged.json').write_text(json.dumps(
                {'source': res['raw_response_file'], 'note': 'regex-recovered unique items from truncated invalid JSON; NOT model-valid output',
                 'detections': sal}, indent=2, ensure_ascii=False))
            panel = draw_panel(img, sal, f'{stage} SALVAGED (invalid JSON)')
            panel.save(vis / f'{stage}_SALVAGED.jpg', quality=90)
            panels.append(panel)
            rows.append((f'{stage} SALVAGED', len(sal), len({d["label"] for d in sal}), f'{lat:.3f}', f"{m['peak_vram_gib']:.2f}",
                         'parse_error; regex-recovered unique items'))
    cols, w = 2, 960
    h = round(img.height * w / img.width)
    sheet = Image.new('RGB', (cols * w, ((len(panels) + cols - 1) // cols) * h), 'white')
    for i, pnl in enumerate(panels):
        sheet.paste(pnl.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    sheet.save(RUN / 'contact_sheet.jpg', quality=85)
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=RUN, capture_output=True, text=True).stdout.strip() or 'NO_COMMIT'
    except FileNotFoundError:
        commit = 'GIT_NOT_FOUND'
    lines = ['# 2026-11-14 all-models detection (exploratory)', '',
             f"- input: `{CFG['image']}` sha256 `{sha256(CFG['image'])}`",
             f"- {CFG['image_note']}", f"- coordinates: {CFG['coordinate_system']}", f'- git commit: {commit}',
             '- latency = one timed call after warmup (Florence OVD = sum over 12 class prompts); VRAM = torch peak allocated in that process stage.', '',
             '| panel | n det | unique labels | latency s | peak VRAM GiB | top labels |', '|---|---|---|---|---|---|']
    lines += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    (RUN / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def selftest():
    dets, rej, err = parse_qwen('```json\n{"objects":[{"label":"box","bbox_2d":[100,200,500,800]},{"label":"x","bbox_2d":[5,5,1,1]}]}\n```', 2000, 1000)
    assert err is None and dets[0]['bbox_px'] == [200.0, 200.0, 1000.0, 800.0] and rej[0]['reason'] == 'bbox_2d not x1<x2, y1<y2', (dets, rej)
    assert parse_qwen('{"objects":[', 10, 10)[2].startswith('JSONDecodeError')
    print('selftest PASS')


if __name__ == '__main__':
    {'selftest': selftest, 'qwen': run_qwen, 'qwen_attempt02': lambda: run_qwen('qwen_attempt02', 'prompt_attempt02'),
     'qwen_attempt03': lambda: run_qwen('qwen_attempt03', 'prompt_attempt02', CFG['qwen']['attempt03_repetition_penalty']), 'dino': run_dino, 'sam2': run_sam2, 'florence': run_florence,
     'yolo_world': run_yolo_world, 'yoloe_pf': run_yoloe_pf, 'render': run_render}[sys.argv[1]]()
