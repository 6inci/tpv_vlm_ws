#!/usr/bin/env python3
"""EXP120 J-gate exploration (dev, no model call): gate the saved J-vlm answers with deterministic checks.

Rules are fixed in config.json "jgate" (read it first). Two gates, named apart because plan v3 already uses "J-gate":
  J-gate-exist(S): spatial relation whose reference object is absent in source S -> abstain; else the J-vlm answer
  J-gate-agree(S): keep the J-vlm pick only if J-geo (reference positions from S) picks the same candidate (= plan v3 J-gate)
Existence / position sources S: L list, M prior map (dev: answer-key stand-in), D detector only (DINO C2 >= 0.5),
G per-image answer key (oracle diagnostic). Candidate sources: oracle (diagnostic) and detected (dev exploration).
Inputs are read only; the J-vlm and J-geo results of the input runs are not changed.
Usage: <qwen_vl python> tools/jgate.py --out runs/jgate_<ts>    |    --selftest
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3a_poc as E  # noqa: E402

HERE, CFG, S, R = E.HERE, E.CFG, E.S, E.R
JG = CFG['jgate']
OK = ('correct_target', 'correct_abstain')
WRONG = ('wrong_target', 'false_select')


def detector_boxes(stem, ref):
    qid = JG['detector_query_by_reference'].get((ref or '').strip().lower())
    if not qid:
        return []
    d = json.loads((HERE / JG['detector_run'] / f'{stem}.json').read_text())
    preds = next(r['preds'] for r in d['results'] if r['qid'] == qid)
    return [p['box'] for p in preds if p['score'] >= JG['detector_threshold']]


def ref_boxes(src, ref, stem, gt):
    """-> list of reference boxes (positions) or None when the source has no positions (L)."""
    cls = E.ref_class(ref)
    if src == 'L':
        return None
    if src == 'D':
        return detector_boxes(stem, ref)
    objs = [o for o in gt if o['label'] not in E.NOT_IN_MAP] if src == 'M' else gt
    return [o['box_xyxy'] for o in objs if cls and o['label'] == cls]


def exists(src, ref, stem, gt):
    return E.ref_class(ref) is not None if src == 'L' else bool(ref_boxes(src, ref, stem, gt))


def gate_exist(row, parse, src, stem, gt):
    if parse['relation'] in JG['gate_applies_to_relations'] and not exists(src, parse.get('reference'), stem, gt):
        return None, 'reference_absent'
    return row['J-vlm']['id'], 'pass'


def gate_agree(row, parse, src, stem, gt):
    vid = row['J-vlm']['id']
    if vid is None:
        return None, 'vlm_abstain'
    if parse['relation'] in JG['gate_applies_to_relations']:
        boxes = ref_boxes(src, parse.get('reference'), stem, gt)
        gid, why = E.resolve(parse, row['cands'], {'_ref': boxes}, cls_override='_ref')
    else:
        gid, why = E.resolve(parse, row['cands'], {})
    return (vid, 'agree') if gid == vid else (None, f'disagree (J-geo {gid}, {why})')


def evaluate(rows, parses, ref, stems, gts):
    out, summ = [], {}
    conds = [('J-gate-exist', s) for s in ('L', 'M', 'D', 'G')] + [('J-gate-agree', s) for s in JG['agree_sources']]
    for r in rows:
        truth = ref[r['layout']][r['qid']]
        base = 'invalid' if r['J-vlm']['error'] else E.judge(r['J-vlm']['box'], truth)
        item = {'layout': r['layout'], 'qid': r['qid'], 'group': r['group'], 'source': r['source'], 'truth': truth,
                'J-vlm': {'id': r['J-vlm']['id'], 'result': base}}
        p = parses[r['qid']]['parsed']
        for gate, src in conds:
            fn = gate_exist if gate == 'J-gate-exist' else gate_agree
            gid, why = fn(r, p, src, stems[r['layout']], gts[r['layout']])
            box = next((c['box'] for c in r['cands'] if c['id'] == gid), None)
            res = 'invalid' if r['J-vlm']['error'] else E.judge(box, truth)
            item[f'{gate}({src})'] = {'id': gid, 'why': why, 'result': res,
                                     'prevented_wrong': base in WRONG and gid is None,
                                     'blocked_correct': base == 'correct_target' and gid is None}
        out.append(item)
    for key in ['J-vlm'] + [f'{g}({s})' for g, s in conds]:
        for grp in ('spatial', 'state', 'all'):
            sel = [i for i in out if grp == 'all' or i['group'] == grp]
            res = [i[key]['result'] for i in sel]
            v = {'n': len(sel), **{c: res.count(c) for c in E.CATS}, 'correct': sum(x in OK for x in res), 'wrong_selection': sum(x in WRONG for x in res)}
            if key != 'J-vlm':
                v['prevented_wrong'] = sum(i[key]['prevented_wrong'] for i in sel)
                v['blocked_correct'] = sum(i[key]['blocked_correct'] for i in sel)
            summ[f'{key}/{grp}'] = v
    return out, summ


def label(cand_src, gate_key):
    if cand_src == 'oracle' or gate_key.endswith('(G)'):
        return 'diagnostic (answer-key information) - not system performance'
    if gate_key.endswith('(M)'):
        return 'map-type information, but the dev map is built from the answer key - not a system claim'
    return 'dev exploration with system-available information (EXP111 detections + list or detector)'


def selftest():
    c = [{'id': 1, 'box': [0, 0, 10, 10], 'state': 'EMPTY', 'colors': []}, {'id': 2, 'box': [100, 0, 110, 10], 'state': 'LOADED', 'colors': ['red']}]
    row = {'J-vlm': {'id': 2}, 'cands': c}
    gt = [{'label': 'charger', 'box_xyxy': [112, 0, 122, 10]}]
    assert gate_exist(row, {'relation': 'next_to', 'reference': 'forklift'}, 'L', 'x', gt) == (None, 'reference_absent')
    assert gate_exist(row, {'relation': 'next_to', 'reference': 'charging station'}, 'M', 'x', gt) == (2, 'pass')
    assert gate_exist(row, {'relation': 'empty'}, 'G', 'x', []) == (2, 'pass')                     # not a spatial relation
    assert gate_agree(row, {'relation': 'next_to', 'reference': 'charging station'}, 'G', 'x', gt) == (2, 'agree')
    assert gate_agree(row, {'relation': 'empty'}, 'G', 'x', gt)[0] is None                           # J-geo picks 1
    assert gate_agree({'J-vlm': {'id': None}, 'cands': c}, {'relation': 'empty'}, 'G', 'x', gt) == (None, 'vlm_abstain')
    print('selftest PASS')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=False)
    ref = json.loads((HERE / JG['referents']).read_text())
    stems = {p.stem[:3]: p.stem for p in R.images_for('dev')}
    gts = {k: S.load_gt(v)['objects'] for k, v in stems.items()}
    man = {'id': 'EXP120-jgate', 'started': R.now(), 'config_jgate': JG, 'referents_status': ref['status'],
           'sha256': {p: R.sha256(HERE / p) for p in ['tools/jgate.py', 'tools/e3a_poc.py', 'config.json', JG['referents']]}}
    all_items, summary = [], {}
    for cand_src, run in JG['inputs'].items():
        for f in ('rows.json', 'parses.json'):
            man['sha256'][f'{run}/{f}'] = R.sha256(HERE / run / f)
        rows = json.loads((HERE / run / 'rows.json').read_text())
        parses = json.loads((HERE / run / 'parses.json').read_text())
        items, summ = evaluate(rows, parses, ref, stems, gts)
        all_items += items
        summary[cand_src] = {k: {**v, 'label': label(cand_src, k.split('/')[0])} for k, v in summ.items()}
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1, ensure_ascii=False))
    (out / 'items.json').write_text(json.dumps(all_items, indent=1, ensure_ascii=False))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    L = ['# EXP120 J-gate 탐색 (dev, 모델 재호출 없음) — 자동 생성', '', f"- 정답: `{JG['referents']}` (초안, 사용자 검토 대기)",
         '- 규칙: `config.json` jgate. 입력 실행의 J-vlm, J-geo 결과는 바꾸지 않는다.', '',
         '| 후보 | 조건 | 묶음 | 정답 | 잘못된 선택 | 올바른 선택 | 놓침 | 올바른 보류 | 정답 없는데 선택 | 막은 오선택 | 막힌 정상 선택 | 표기 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for cand_src, summ in summary.items():
        for k, v in summ.items():
            L.append(f"| {cand_src} | {k.split('/')[0]} | {k.split('/')[1]} | {v['correct']}/{v['n']} | {v['wrong_selection']} | {v['correct_target']} | "
                     f"{v['missed']} | {v['correct_abstain']} | {v['false_select']} | {v.get('prevented_wrong', '-')} | {v.get('blocked_correct', '-')} | {v['label']} |")
    (out / 'SUMMARY.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('written', out)


if __name__ == '__main__':
    sys.exit(main())
