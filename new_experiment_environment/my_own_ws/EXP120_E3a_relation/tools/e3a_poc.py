#!/usr/bin/env python3
"""EXP120 (= plan v3 E3a) feasibility, dev only: can the VLM pick the right small pallet from a relation/state phrase?

Conditions (same candidates, same phrase, same abstain options):
  J-vlm : numbered candidate pallets drawn on the image -> Qwen3-VL-4B picks an id or abstains (judges relations
          and states from the image; reference objects are NOT marked)
  J-geo : Qwen parses the phrase (text only) -> Python resolves it on the same candidates with the candidate states
          and the map reference boxes (pixel gaps, provisional rule fixed in config.json)
Candidate source:
  oracle   : answer-key small pallets with answer-key states (diagnostic upper bound; NOT a system result)
  detected : EXP111 predicted pallets + predicted states (config detected_source). Dev exploration under the limited
             AGENTS exception approved by the user on 2026-11-15; NOT an E3a PASS and NOT a system validation.
Reference objects for J-geo (both sources): answer-key boxes of fixed objects = M-map prior-map stand-in.
Referents: a JSON answer file (default referents_draft.json, user review pending). Raw outputs are saved; --rescore
re-judges a saved run against another referents file without calling the model.
Usage:
  <qwen_vl python> tools/e3a_poc.py --source oracle --out runs/<new>
  <qwen_vl python> tools/e3a_poc.py --rescore runs/<run> --referents referents_reviewed.json --out runs/<run>/rescore_<ts>
  <qwen_vl python> tools/e3a_poc.py --selftest
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXP110 = HERE.parent / 'EXP110_word_grounding'
sys.path.insert(0, str(EXP110 / 'tools'))
import score as S  # noqa: E402
import run_models as R  # noqa: E402

CFG = json.loads((HERE / 'config.json').read_text())
RULE = CFG['relation_rule_px']
RELATIONS = ('closest_to', 'next_to', 'empty', 'loaded', 'has_color', 'none')
CATS = ('correct_target', 'wrong_target', 'missed', 'correct_abstain', 'false_select', 'invalid')
NOT_IN_MAP = ('small_pallet', 'block')


def gap(a, b):
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def ref_class(name):
    """site name or class -> class (case-insensitive), else None."""
    if not isinstance(name, str):
        return None
    n = name.strip().lower()
    voc = {k.lower(): v for k, v in CFG['site_vocabulary'].items()}
    return voc.get(n) or (n if n in voc.values() else None)


def resolve(parse, cands, scene, cls_override=None):
    """Python relation rule. cands: [{'id', 'box', 'state', 'colors'}]; scene: class -> [boxes]. -> (id | None, reason).
    cls_override: scene key to use for the reference instead of the site-vocabulary lookup (J-gate detector source)."""
    rel = parse.get('relation')
    if rel in ('empty', 'loaded', 'has_color'):
        if rel == 'empty':
            ok = [c for c in cands if c['state'] == 'EMPTY']
        elif rel == 'loaded':
            ok = [c for c in cands if c['state'] == 'LOADED']
        else:
            ok = [c for c in cands if parse.get('color') in c['colors']]
        if not ok:
            return None, 'none'
        return (ok[0]['id'], 'ok') if len(ok) == 1 else (None, 'ambiguous')
    if rel in ('closest_to', 'next_to'):
        cls = cls_override or ref_class(parse.get('reference'))
        refs = scene.get(cls, []) if cls else []
        if not refs or not cands:
            return None, 'unknown'
        size = max(max(b[2] - b[0], b[3] - b[1]) for b in refs)
        gaps = sorted((min(gap(c['box'], r) for r in refs), c['id']) for c in cands)
        if rel == 'next_to' and gaps[0][0] > RULE['next_to_max_gap_factor'] * size:
            return None, 'none'
        if len(gaps) > 1 and gaps[1][0] - gaps[0][0] < RULE['ambiguity_margin_factor'] * size:
            if rel == 'closest_to' or gaps[1][0] <= RULE['next_to_max_gap_factor'] * size:
                return None, 'ambiguous'
        return gaps[0][1], 'ok'
    return None, 'no_relation'


def check_parse(p):
    if not isinstance(p, dict):
        return 'schema: not an object'
    if p.get('relation') not in RELATIONS:
        return f"schema: relation {p.get('relation')!r}"
    if p['relation'] == 'has_color' and p.get('color') not in ('red', 'blue'):
        return f"schema: color {p.get('color')!r}"
    if p['relation'] in ('closest_to', 'next_to') and not isinstance(p.get('reference'), str):
        return f"schema: reference {p.get('reference')!r}"
    return None


def check_vlm(p, ids):
    if not isinstance(p, dict) or 'id' not in p:
        return 'schema: no id key'
    v = p['id']
    if v is None:
        return None if p.get('reason') in (None, 'none', 'ambiguous') else f"schema: reason {p.get('reason')!r}"
    if isinstance(v, bool) or not isinstance(v, int):
        return f'schema: id {v!r} is not an int'
    return None if v in ids else f'id {v} is not a candidate'


def ask_text(qwen, text, max_new_tokens):
    msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}]
    inp = qwen.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
    with qwen.torch.inference_mode():
        out = qwen.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return qwen.proc.decode(out[0][inp['input_ids'].shape[-1]:], skip_special_tokens=True)


def parse_json(raw):
    try:
        return json.loads(R.strip_fence(raw)), None
    except json.JSONDecodeError as e:
        return None, f'JSONDecodeError: {e}'


def judge(pred_box, truth):
    """truth: box | 'none' | 'unknown' (any abstention is right for both). -> one of CATS except invalid."""
    if isinstance(truth, list):
        if pred_box is None:
            return 'missed'
        return 'correct_target' if S.center_in(pred_box, truth) else 'wrong_target'
    return 'correct_abstain' if pred_box is None else 'false_select'


def score_rows(rows, ref):
    """Adds 'truth' and per-judge 'result' to each row; returns the summary (by judge x group)."""
    for r in rows:
        r['truth'] = ref[r['layout']][r['qid']]
        for j in ('J-vlm', 'J-geo'):
            r[j]['result'] = 'invalid' if r[j]['error'] else judge(r[j]['box'], r['truth'])
    summ = {}
    for j in ('J-vlm', 'J-geo'):
        for g in ('spatial', 'state', 'all'):
            sel = [r for r in rows if g == 'all' or r['group'] == g]
            res = [r[j]['result'] for r in sel]
            summ[f'{j}/{g}'] = {'n': len(sel), 'n_target_present': sum(isinstance(r['truth'], list) for r in sel),
                                **{c: res.count(c) for c in CATS}}
    layouts = sorted({r['layout'] for r in rows})
    for j in ('J-vlm', 'J-geo'):  # layout-level: every query of the layout answered correctly (target or abstention)
        ok = [lay for lay in layouts if all(r[j]['result'] in ('correct_target', 'correct_abstain') for r in rows if r['layout'] == lay)]
        summ[f'{j}/layout_all_correct'] = {'n_layouts': len(layouts), 'all_correct': len(ok), 'layouts': ok}
    return summ


def write_summary(out, summ, rows, meta):
    lines = [f"# EXP120 E3a feasibility — {meta['source']} ({meta['label']})", '',
             f"- referents: `{meta['referents']}` ({meta['referents_status'][:60]}...)",
             '- 자동 생성 파일 (`tools/e3a_poc.py`). 숫자는 rows.json에서 다시 계산할 수 있다.', '',
             '| 조건/묶음 | 항목 | 정답 있음 | correct_target | wrong_target | missed | correct_abstain | false_select | invalid |',
             '|---|---|---|---|---|---|---|---|---|']
    for k, v in summ.items():
        if k.endswith('layout_all_correct'):
            continue
        lines.append(f"| {k} | {v['n']} | {v['n_target_present']} | " + ' | '.join(str(v[c]) for c in CATS) + ' |')
    lines += ['', '배치 단위 (그 배치의 질의 6개가 모두 정답):', '']
    for j in ('J-vlm', 'J-geo'):
        v = summ[f'{j}/layout_all_correct']
        lines.append(f"- {j}: {v['all_correct']}/{v['n_layouts']} {v['layouts']}")
    lines += ['', '| 배치 | 질의 | 정답 | J-vlm 번호 | J-vlm 결과 | J-geo 번호 (이유) | J-geo 결과 |', '|---|---|---|---|---|---|---|']
    for r in rows:
        t = r['truth'] if isinstance(r['truth'], str) else next((c['id'] for c in r['cands'] if S.center_in(c['box'], r['truth'])), 'box(후보 없음)')
        lines.append(f"| {r['layout']} | {r['qid']} | {t} | {r['J-vlm']['id']} | {r['J-vlm']['result']} | {r['J-geo']['id']} ({r['J-geo']['reason']}) | {r['J-geo']['result']} |")
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


def neutral_overlay(img, cands):
    """Diagnostic D1 (DESIGN §3a): same layout as run_models.overlay, but every box and label in white."""
    from PIL import ImageDraw, ImageFont
    im = img.copy()
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=22)
    for c in cands:
        x1, y1, x2, y2 = c['box']
        d.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=3)
        t = str(c['id'])
        tb = d.textbbox((0, 0), t, font=font)
        tx, ty = x1, max(0, y1 - (tb[3] - tb[1]) - 6)
        d.rectangle([tx, ty, tx + tb[2] - tb[0] + 6, ty + tb[3] - tb[1] + 6], fill=(0, 0, 0))
        d.text((tx + 3, ty + 3 - tb[1]), t, fill=(255, 255, 255), font=font)
    return im


def cands_for(source, img_path, gt):
    if source == 'oracle':
        pallets = sorted([o['box_xyxy'] for o in gt if o['label'] == 'small_pallet'], key=lambda b: b[0])
        blocks = [o for o in gt if o['label'] == 'block']
        cands = []
        for i, b in enumerate(pallets, 1):
            on = sorted({o.get('variant') for o in blocks if S.center_in(o['box_xyxy'], b)})
            cands.append({'id': i, 'box': b, 'state': 'LOADED' if on else 'EMPTY', 'colors': on})
        return cands
    so = json.loads((HERE / CFG['detected_source']).read_text())[img_path.stem[:3]]
    assert so['image_stem'] == img_path.stem, (so['image_stem'], img_path.stem)
    ps = sorted(so['pallets'], key=lambda p: p['box_raw_px'][0])
    return [{'id': i, 'box': p['box_raw_px'], 'state': p['state'], 'colors': sorted(p['block_colors'])} for i, p in enumerate(ps, 1)]


def selftest():
    c = [{'id': 1, 'box': [0, 0, 10, 10], 'state': 'EMPTY', 'colors': []},
         {'id': 2, 'box': [100, 0, 110, 10], 'state': 'LOADED', 'colors': ['blue', 'red']}]
    scene = {'charger': [[112, 0, 122, 10]], 'truck': [[300, 0, 400, 50]]}
    assert gap([0, 0, 10, 10], [13, 14, 20, 20]) == 5.0
    assert resolve({'relation': 'next_to', 'reference': 'charging station'}, c, scene) == (2, 'ok')
    assert resolve({'relation': 'next_to', 'reference': 'Charger'}, c, scene) == (2, 'ok')            # class name accepted
    assert resolve({'relation': 'next_to', 'reference': 'truck'}, c, scene) == (None, 'none')      # 190 px > 100 px gate
    assert resolve({'relation': 'closest_to', 'reference': 'truck'}, c, scene) == (2, 'ok')
    assert resolve({'relation': 'next_to', 'reference': 'forklift'}, c, scene) == (None, 'unknown')
    assert resolve({'relation': 'empty'}, c, scene) == (1, 'ok') and resolve({'relation': 'has_color', 'color': 'blue'}, c, scene) == (2, 'ok')
    assert check_parse({'relation': 'near'}) and check_parse({'relation': 'next_to', 'reference': None}) and not check_parse({'relation': 'empty'})
    assert check_vlm({'id': 3}, {1, 2}) and check_vlm({'id': '1'}, {1, 2}) and check_vlm([1], {1, 2}) and check_vlm({'id': True}, {1, 2})
    assert not check_vlm({'id': None, 'reason': 'none'}, {1, 2}) and not check_vlm({'id': 2}, {1, 2})
    assert judge(None, 'none') == 'correct_abstain' and judge([0, 0, 10, 10], [0, 0, 10, 10]) == 'correct_target'
    assert judge([100, 0, 110, 10], [0, 0, 10, 10]) == 'wrong_target' and judge(None, [0, 0, 10, 10]) == 'missed'
    assert judge([0, 0, 10, 10], 'unknown') == 'false_select'
    rows = [{'layout': 'X', 'qid': 'e01', 'group': 'spatial', 'J-vlm': {'error': 'bad', 'box': None}, 'J-geo': {'error': None, 'box': [0, 0, 10, 10]}}]
    s = score_rows(rows, {'X': {'e01': [0, 0, 10, 10]}})
    assert s['J-vlm/spatial']['invalid'] == 1 and s['J-geo/all']['correct_target'] == 1 and s['J-geo/state']['n'] == 0
    assert s['J-vlm/layout_all_correct']['all_correct'] == 0 and s['J-geo/layout_all_correct']['layouts'] == ['X']
    print('selftest PASS')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['oracle', 'detected'])
    ap.add_argument('--rescore', help='saved run dir: re-judge rows.json with --referents (no model call)')
    ap.add_argument('--referents', default='referents_draft.json')
    ap.add_argument('--overlay', default='palette', choices=['palette', 'neutral'], help='neutral = diagnostic D1 (DESIGN §3a)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ref = json.loads((HERE / a.referents).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if a.rescore:
        rows = json.loads((Path(a.rescore) / 'rows.json').read_text())
        meta = json.loads((Path(a.rescore) / 'MANIFEST.json').read_text())
        meta.update({'referents': a.referents, 'referents_status': ref['status'], 'rescored_from': a.rescore, 'rescored_at': R.now()})
    else:
        if not a.source:
            raise SystemExit('--source is required for a model run')
        from PIL import Image
        label = 'diagnostic upper bound, not a system result' if a.source == 'oracle' else \
            'dev exploration under the limited AGENTS exception (user, 2026-11-15); not E3a PASS, not system validation'
        if a.overlay == 'neutral':
            label += '; diagnostic D1: white candidate boxes (DESIGN §3a)'
        meta = {'id': 'EXP120-feasibility', 'source': a.source, 'label': label, 'overlay': a.overlay, 'started': R.now(), 'env': R.env_info(), 'config': CFG,
                'referents': a.referents, 'referents_status': ref['status'],
                'sha256': {p: R.sha256(HERE / p) for p in ['tools/e3a_poc.py', 'config.json', a.referents]}}
        if a.source == 'detected':
            meta['sha256'][CFG['detected_source']] = R.sha256(HERE / CFG['detected_source'])
        qwen = R.get_qwen()
        vocab = ', '.join(CFG['site_vocabulary'])
        parses = {}
        for q in CFG['queries']:  # the parse depends only on the text -> one call per phrase
            raw = ask_text(qwen, CFG['parser_prompt'].format(vocab=vocab, q=q['en']), CFG['max_new_tokens'])
            p, err = parse_json(raw)
            parses[q['qid']] = {'raw': raw, 'parsed': p, 'error': err or check_parse(p)}
            print('parse', q['qid'], raw.strip().replace('\n', ' ')[:150], '| error', parses[q['qid']]['error'], flush=True)
        rows = []
        for img_path in R.images_for('dev'):
            layout = img_path.stem[:3]
            img = Image.open(img_path).convert('RGB')
            gt = S.load_gt(img_path.stem)['objects']
            cands = cands_for(a.source, img_path, gt)
            ids = {c['id'] for c in cands}
            scene = {}
            for o in gt:
                if o['label'] not in NOT_IN_MAP:
                    scene.setdefault(o['label'], []).append(o['box_xyxy'])
            overlay = R.overlay(img, cands) if a.overlay == 'palette' else neutral_overlay(img, cands)
            overlay.save(out / f'overlay_{layout}.jpg', quality=90)
            for q in CFG['queries']:
                raw_v, ntok, nin, dt = qwen.ask(overlay, CFG['jvlm_prompt'].format(q=q['en']), CFG['max_new_tokens'])
                pv, err_v = parse_json(raw_v)
                err_v = err_v or check_vlm(pv, ids)
                vid = None if err_v else pv['id']
                pq = parses[q['qid']]
                gid, gwhy = (None, 'parse_error') if pq['error'] else resolve(pq['parsed'], cands, scene)
                rows.append({'layout': layout, 'qid': q['qid'], 'group': q['group'], 'text': q['en'], 'source': a.source, 'cands': cands,
                             'J-vlm': {'raw': raw_v, 'error': err_v, 'id': vid, 'box': next((c['box'] for c in cands if c['id'] == vid), None),
                                       'reason': pv.get('reason') if isinstance(pv, dict) else None, 'latency_s': round(dt, 2)},
                             'J-geo': {'error': pq['error'], 'id': gid, 'box': next((c['box'] for c in cands if c['id'] == gid), None), 'reason': gwhy}})
                print(layout, q['qid'], '| J-vlm', vid, err_v or '', '| J-geo', gid, gwhy, flush=True)
        (out / 'parses.json').write_text(json.dumps(parses, indent=1, ensure_ascii=False))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        meta['finished'] = R.now()
    summ = score_rows(rows, ref)
    (out / 'MANIFEST.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str))
    (out / 'scored.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    (out / 'summary.json').write_text(json.dumps(summ, indent=1))
    write_summary(out, summ, rows, meta)
    for k, v in summ.items():
        print(k, v)
    print('written', out)


if __name__ == '__main__':
    sys.exit(main())
