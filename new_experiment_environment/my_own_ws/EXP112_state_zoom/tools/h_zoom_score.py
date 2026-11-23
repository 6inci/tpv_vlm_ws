#!/usr/bin/env python3
"""EXP112 scoring (fixed rules in config.json: units, errors, screening_rule, bootstrap, destination_analysis).

Primary unit: the 16 dev pallets on frame 0. Frames 1-2 only give repeatability. Statistics resample layouts.
Per condition: state correct, A = E1 (empty judged loaded), B = E2 (loaded judged empty), E3 colour error on correctly
loaded pallets, E4 floor block reported on the pallet, E7 invalid; layout success; repeatability; destination selection.
Screening (dev only, not hypothesis proof): at most one of Z-crop / Z-both, see config screening_rule.
Usage: <qwen_vl python> tools/h_zoom_score.py --run runs/dev_<ts> [--no-images]   |   --selftest
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
WS = HERE.parent
sys.path.insert(0, str(WS / 'EXP120_E3a_relation' / 'tools'))
from e3a_poc import resolve  # noqa: E402

CFG = json.loads((HERE / 'config.json').read_text())
SR = CFG['screening_rule']
DEST = {'empty pallet': {'relation': 'empty'}, 'pallet with blocks': {'relation': 'loaded'},
        'pallet with the blue block': {'relation': 'has_color', 'color': 'blue'},
        'pallet with the red block': {'relation': 'has_color', 'color': 'red'}}


def truth_from_relations(rel):
    """relation sheet -> {pallet_id: {'on': set(colours), 'floor_near': set(colours)}}."""
    t = defaultdict(lambda: {'on': set(), 'floor_near': set()})
    for color, where in rel['blocks'].items():
        kind, _, pid = where.partition(':')
        if kind in ('on', 'floor_near'):
            t[int(pid)][kind].add(color)
    return t


def judge_pallet(rec, tr):
    gt_loaded = bool(tr['on'])
    if rec['error'] is not None or rec['answer'] is None:
        return {'state_ok': False, 'invalid': True, 'A': False, 'B': False, 'E3': False, 'E4': False, 'color_exact': False}
    ans = set(rec['answer'])
    pl = bool(rec['loaded'])
    return {'state_ok': pl == gt_loaded, 'invalid': False, 'A': not gt_loaded and pl, 'B': gt_loaded and not pl,
            'E3': gt_loaded and pl and ans != tr['on'], 'E4': bool(tr['floor_near'] & ans), 'color_exact': ans == tr['on']}


def dest_outcome(pred_id, true_id):
    if true_id is not None:
        return 'correct_target' if pred_id == true_id else ('missed' if pred_id is None else 'wrong_target')
    return 'correct_abstain' if pred_id is None else 'false_select'


def error_types(j):
    return [n for n, k in (('E1', 'A'), ('E2', 'B'), ('E3', 'E3'), ('E4', 'E4'), ('E7', 'invalid')) if j[k]]


def consequences(pc, tc, errs):
    """Per erroneous pallet and destination query: recompute with only that pallet set to the answer key.
    pc / tc: predicted / answer-key candidates of one layout; errs: {pallet_id: [error types]}. -> list of records."""
    out = []
    for name, q in DEST.items():
        truth = resolve(q, tc, {})[0]
        actual = dest_outcome(resolve(q, pc, {})[0], truth)
        for pid, types in errs.items():
            fixed = [next(t for t in tc if t['id'] == pid) if p['id'] == pid else p for p in pc]
            alt = dest_outcome(resolve(q, fixed, {})[0], truth)
            if alt == actual:
                cons = 'no_effect'
            elif actual in ('wrong_target', 'false_select'):
                cons = 'wrong_selection'
            elif actual == 'missed':
                cons = 'abstain'
            else:
                cons = 'other'
            out.append({'pallet_id': pid, 'errors': types, 'query': name, 'actual': actual, 'if_fixed': alt, 'consequence': cons})
    return out


def cluster_ci(diff_by_layout, n_boot, seed):
    rng = np.random.default_rng(seed)
    lays = sorted(diff_by_layout)
    v = np.array([diff_by_layout[k] for k in lays])
    s = [v[rng.integers(0, len(v), len(v))].sum() for _ in range(n_boot)]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def score(calls, truths, conds=('Z-full', 'Z-crop', 'Z-both', 'B-rule')):
    """calls: list of call records; truths: {layout: truth_from_relations(...)}. -> (summary, per-pallet rows, consequence records)."""
    f0 = CFG['data']['primary_frame']
    by = {(c['cond'], c['layout'], c['frame'], c['pallet_id']): c for c in calls}
    rows, summ, all_cons = [], {}, []
    for cond in conds:
        recs = [c for c in calls if c['cond'] == cond and c['frame'] == f0]
        if not recs:
            continue
        js = []
        for r in recs:
            j = judge_pallet(r, truths[r['layout']][r['pallet_id']])
            frames = [by.get((cond, r['layout'], fr, r['pallet_id'])) for fr in CFG['data']['frames'] if fr != f0]
            j['repeat_same'] = all(x is not None and x['error'] is None and x['answer'] == r['answer'] for x in frames) if frames else None
            rows.append({'cond': cond, 'layout': r['layout'], 'pallet_id': r['pallet_id'], 'answer': r['answer'], 'error': r['error'],
                         'truth_on': sorted(truths[r['layout']][r['pallet_id']]['on']), **j})
            js.append((r, j))
        lay_ok = defaultdict(list)
        for r, j in js:
            lay_ok[r['layout']].append(j['state_ok'] and j['color_exact'])
        dest, cons_rows = defaultdict(int), []
        for lay in sorted({r['layout'] for r, _ in js}):
            pc = [{'id': r['pallet_id'], 'box': r['box'], 'state': 'UNKNOWN' if j['invalid'] else ('LOADED' if r['loaded'] else 'EMPTY'),
                   'colors': [] if j['invalid'] else r['answer']} for r, j in js if r['layout'] == lay]
            tc = [{'id': p['id'], 'box': p['box'], 'state': 'LOADED' if truths[lay][p['id']]['on'] else 'EMPTY',
                   'colors': sorted(truths[lay][p['id']]['on'])} for p in pc]
            for q in DEST.values():
                dest[dest_outcome(resolve(q, pc, {})[0], resolve(q, tc, {})[0])] += 1
            errs = {r['pallet_id']: error_types(j) for r, j in js if r['layout'] == lay and error_types(j)}
            cons_rows += [{'cond': cond, 'layout': lay, **c} for c in consequences(pc, tc, errs)]
        link = defaultdict(lambda: defaultdict(int))
        for c in cons_rows:
            for t in c['errors']:
                link[t][c['consequence']] += 1
        all_cons += cons_rows
        lat = sorted(r['latency_s'] for r, _ in js)
        n_loaded = sum(bool(truths[r['layout']][r['pallet_id']]['on']) for r, _ in js)
        summ[cond] = {'pallets': len(js), 'state_correct': sum(j['state_ok'] for _, j in js),
                      'A_E1_empty_as_loaded': sum(j['A'] for _, j in js), 'n_empty': len(js) - n_loaded,
                      'B_E2_loaded_as_empty': sum(j['B'] for _, j in js), 'n_loaded': n_loaded,
                      'E3_colour_error': sum(j['E3'] for _, j in js), 'E4_floor_block_on': sum(j['E4'] for _, j in js),
                      'E7_invalid': sum(j['invalid'] for _, j in js), 'colour_exact': sum(j['color_exact'] for _, j in js),
                      'E7_on_loaded': sum(j['invalid'] and bool(truths[r['layout']][r['pallet_id']]['on']) for r, j in js),
                      'error_to_destination': {t: dict(v) for t, v in link.items()},
                      'layouts_all_correct': sum(all(v) for v in lay_ok.values()), 'n_layouts': len(lay_ok),
                      'repeat_same': sum(bool(j['repeat_same']) for _, j in js), 'latency_median_s': lat[len(lat) // 2],
                      'destination': dict(dest), '_by_layout_correct': {k: sum(j['state_ok'] for r, j in js if r['layout'] == k) for k in lay_ok}}
    return summ, rows, all_cons


def screen(summ):
    """config screening_rule -> (chosen condition or None, per-condition details)."""
    base = summ['Z-full']
    det = {}
    for c in ('Z-crop', 'Z-both'):
        if c not in summ:
            continue
        s = summ[c]
        net = s['state_correct'] - base['state_correct']
        diff = {k: s['_by_layout_correct'][k] - base['_by_layout_correct'][k] for k in base['_by_layout_correct']}
        guard = s['B_E2_loaded_as_empty'] + s['E7_on_loaded']          # v4: invalid on a loaded pallet cannot lower B
        det[c] = {'net': net, 'B_guard_count': guard, 'B_guard_count_Z-full': base['B_E2_loaded_as_empty'] + base['E7_on_loaded'],
                  'B_not_increased': guard <= base['B_E2_loaded_as_empty'] + base['E7_on_loaded'],
                  'net_ge_k': net >= SR['k'], 'latency_median_s': s['latency_median_s'],
                  'net_ci95_layout_bootstrap': cluster_ci(diff, CFG['bootstrap']['n_boot'], CFG['bootstrap']['seed'])}
        det[c]['eligible'] = det[c]['B_not_increased'] and det[c]['net_ge_k']
    ok = [c for c in det if det[c]['eligible']]
    chosen = min(ok, key=lambda c: (-det[c]['net'], det[c]['latency_median_s'])) if ok else None
    return chosen, det


def selftest():
    tr = {'L1': truth_from_relations({'blocks': {'red': 'on:1', 'blue': 'floor_near:2'}}),
          'L2': truth_from_relations({'blocks': {'red': 'on:2', 'blue': 'on:2'}})}
    assert tr['L1'][1]['on'] == {'red'} and tr['L1'][2]['floor_near'] == {'blue'} and not tr['L1'][2]['on']

    def rec(cond, lay, pid, ans, err=None, lat=1.0, frame=0):
        return {'cond': cond, 'layout': lay, 'frame': frame, 'pallet_id': pid, 'box': [pid * 100, 0, pid * 100 + 50, 50],
                'answer': ans, 'loaded': bool(ans) if ans is not None else None, 'error': err, 'latency_s': lat}
    calls = []
    for cond, a in (('Z-full', {('L1', 1): [], ('L1', 2): ['blue'], ('L2', 1): [], ('L2', 2): []}),
                    ('Z-crop', {('L1', 1): ['red'], ('L1', 2): [], ('L2', 1): [], ('L2', 2): ['blue', 'red']}),
                    ('Z-both', {('L1', 1): ['red'], ('L1', 2): [], ('L2', 1): [], ('L2', 2): None})):
        for (lay, pid), ans in a.items():
            calls.append(rec(cond, lay, pid, ans, err='bad' if ans is None else None, lat=2.0 if cond == 'Z-both' else 1.0))
    summ, rows, cons = score(calls, tr, conds=('Z-full', 'Z-crop', 'Z-both'))
    # error type vs final destination outcome, recorded separately (Z-full: L1 p1 B, L1 p2 A+E4, L2 p2 B)
    assert summ['Z-full']['error_to_destination'] == {'E2': {'wrong_selection': 2, 'abstain': 5, 'no_effect': 1},
                                                       'E1': {'wrong_selection': 3, 'no_effect': 1},
                                                       'E4': {'wrong_selection': 3, 'no_effect': 1}}, summ['Z-full']['error_to_destination']
    assert summ['Z-crop']['error_to_destination'] == {} and summ['Z-both']['E7_on_loaded'] == 1
    f, c, b = summ['Z-full'], summ['Z-crop'], summ['Z-both']
    assert (f['state_correct'], f['A_E1_empty_as_loaded'], f['B_E2_loaded_as_empty'], f['E4_floor_block_on']) == (1, 1, 2, 1), f
    assert (c['state_correct'], c['B_E2_loaded_as_empty'], c['colour_exact'], c['layouts_all_correct']) == (4, 0, 4, 2), c
    assert (b['state_correct'], b['E7_invalid'], b['B_E2_loaded_as_empty']) == (3, 1, 0), b   # invalid counts as E7 only
    # all-correct Z-crop: 2 layouts x 4 destination queries = 7 targets + 1 right abstention (L1 blue block is on the floor)
    assert c['destination'] == {'correct_target': 7, 'correct_abstain': 1} and f['destination']['wrong_target'] >= 1, (f, c)
    old = SR['k']
    SR['k'] = 3
    chosen, det = screen(summ)
    assert det['Z-crop']['net'] == 3 and det['Z-crop']['eligible'] and det['Z-both']['net'] == 2 and not det['Z-both']['eligible']
    assert chosen == 'Z-crop'
    summ['Z-both'].update({'state_correct': 4, 'B_E2_loaded_as_empty': 0, 'latency_median_s': 0.5,
                           '_by_layout_correct': {'L1': 2, 'L2': 2}})
    assert screen(summ)[0] == 'Z-both'           # same net 3, faster -> tie broken by latency
    summ['Z-both']['B_E2_loaded_as_empty'] = 3
    assert screen(summ)[0] == 'Z-crop'           # B increased -> not eligible
    summ['Z-crop']['state_correct'] = 3
    summ['Z-crop']['_by_layout_correct'] = {'L1': 2, 'L2': 1}
    assert screen(summ)[0] is None               # net 2 < k and the other has B up -> nothing chosen
    # v4 guard: lowering B by invalid outputs on loaded pallets does not pass (B 0 + E7 on loaded 3 > Z-full 2)
    summ['Z-crop'].update({'state_correct': 4, 'B_E2_loaded_as_empty': 0, 'E7_on_loaded': 3, '_by_layout_correct': {'L1': 2, 'L2': 2}})
    assert screen(summ)[1]['Z-crop']['net'] == 3 and not screen(summ)[1]['Z-crop']['B_not_increased']
    SR['k'] = old
    print('selftest PASS')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--no-images', action='store_true')
    a = ap.parse_args()
    run = HERE / a.run
    man = json.loads((run / 'MANIFEST.json').read_text())
    if not man['id'].endswith('dev'):
        raise SystemExit('only dev runs are scored (smoke runs are format and memory checks)')
    calls = [json.loads(x) for x in (run / 'calls.jsonl').read_text().splitlines()]
    truths = {lay: truth_from_relations(json.loads((HERE / CFG['data']['relations'].format(layout=lay)).read_text()))
              for lay in CFG['data']['layouts']}
    summ, rows, cons = score(calls, truths)
    chosen, det = screen(summ)
    out = run / 'score'
    out.mkdir(exist_ok=False)
    res = {'rule': SR, 'summary': {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')} for k, v in summ.items()},
           'screening': {'details': det, 'chosen_for_test': chosen,
                         'meaning': 'dev screening only; not hypothesis proof; the test criterion is fixed before the test'}}
    (out / 'summary.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (out / 'pallets.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    (out / 'error_consequences.json').write_text(json.dumps(cons, indent=1, ensure_ascii=False))
    if not a.no_images:
        sys.path.insert(0, str(HERE / 'tools'))
        from h_zoom import enlarged
        from PIL import Image
        fd = out / 'failures'
        fd.mkdir()
        for r in rows:
            if not (r['state_ok'] and r['color_exact']):
                img = Image.open(HERE / CFG['data']['images'].format(layout=r['layout'], frame=CFG['data']['primary_frame'])).convert('RGB')
                box = next(c['box'] for c in calls if c['layout'] == r['layout'] and c['pallet_id'] == r['pallet_id'])
                enlarged(img, box)[0].save(fd / f"{r['cond']}_{r['layout']}_p{r['pallet_id']}.jpg", quality=90)
                (fd / f"{r['cond']}_{r['layout']}_p{r['pallet_id']}.json").write_text(json.dumps(r, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main())
