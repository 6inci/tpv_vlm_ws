#!/usr/bin/env python3
"""EXP110-X6 (exploratory, dev only): compare two models on the same dev queries, images and C6 candidates.

Scoring is the registered EXP110 rule (tools/score.py rows_for: Hungarian IoU>=0.5, strict success).
Per (condition, model) it separates, on failed queries: output format error / missed instances only /
wrong objects (false positives) only / both; multi-object queries (>=2 answers): partial and total misses;
absent-object queries (Q5): hallucinated boxes; task objects (small_pallet, block); latency and peak VRAM.
Paired: per (query, layout) success of A vs B; query-cluster bootstrap CI (score.paired) is the main basis.
Usage: <qwen_vl python> explore/X6_model_size/x6_compare.py --a run02:M1 --b x6_M3_dev:M3 --out explore/X6_model_size/compare_<ts>
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import score as S  # noqa: E402

CONDS = ('C1', 'C6-dino')
TASK = ('small_pallet', 'block')


def fail_kind(r):
    if r['success']:
        return 'success'
    if r['invalid']:
        return 'invalid'
    if r['fn'] and r['fp']:
        return 'miss_and_wrong'
    return 'miss_only' if r['fn'] else 'wrong_only'


def block(rs):
    kinds = defaultdict(int)
    for r in rs:
        kinds[fail_kind(r)] += 1
    multi = [r for r in rs if r['n_gt'] >= 2 and not r['invalid']]
    absent = [r for r in rs if r['n_gt'] == 0 and not r['invalid']]
    tp, fp, ngt = sum(r['tp'] for r in rs), sum(r['fp'] for r in rs), sum(r['n_gt'] for r in rs)
    return {'n': len(rs), 'success_rate': round(float(np.mean([r['success'] for r in rs])), 4) if rs else None, **dict(kinds),
            'instance_recall': round(tp / ngt, 4) if ngt else None, 'false_positive_boxes': fp,
            'multi_n': len(multi), 'multi_success': sum(r['success'] for r in multi),
            'multi_partial_miss': sum(0 < r['tp'] < r['n_gt'] for r in multi), 'multi_total_miss': sum(r['tp'] == 0 for r in multi),
            'multi_instance_recall': round(sum(r['tp'] for r in multi) / sum(r['n_gt'] for r in multi), 4) if multi else None,
            'absent_n': len(absent), 'absent_hallucinated': sum(r['fp'] > 0 for r in absent)}


def timing(run_dir, cond):
    lat, tok, hit, vram = [], [], 0, []
    for f in sorted((run_dir / cond).glob('P0[1-4]_*.json')):
        d = json.loads(f.read_text())
        vram.append(d.get('peak_vram_gib'))
        for r in d['results']:
            if r.get('raw') is None:  # C6 with no candidates: no model call
                continue
            lat.append(r['latency_s'])
            tok.append(r.get('generated_tokens') or 0)
            hit += bool(r.get('hit_max_new_tokens'))
    return {'model_calls': len(lat), 'latency_median_s': round(float(np.median(lat)), 2), 'latency_p90_s': round(float(np.percentile(lat, 90)), 2),
            'latency_max_s': round(max(lat), 2), 'latency_sum_s': round(sum(lat), 1), 'generated_tokens_median': float(np.median(tok)),
            'hit_max_new_tokens': hit, 'peak_vram_gib_reported': max(v for v in vram if v is not None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='run:label, e.g. run02:M1')
    ap.add_argument('--b', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=False)
    queries = json.loads((ROOT / 'queries.json').read_text())['queries']
    runs = dict(x.split(':')[::-1] for x in (a.a, a.b))  # label -> run
    labels = [a.a.split(':')[1], a.b.split(':')[1]]
    rows = []
    for lab in labels:
        for r in S.rows_for(ROOT / 'results' / runs[lab], 'dev', {}, queries):
            if r['cond'] in CONDS:
                rows.append({**r, 'cond': f"{lab}:{r['cond']}", 'base_cond': r['cond'], 'model': lab})
    if S.INCOMPLETE:
        raise SystemExit(f'incomplete runs: {S.INCOMPLETE}')
    res = {'runs': runs, 'scoring': 'tools/score.py rows_for (registered EXP110 rule), split dev, no threshold', 'by': {}, 'paired': {}, 'timing': {}, 'diff_queries': {}}
    for cond in CONDS:
        for lab in labels:
            rs = [r for r in rows if r['cond'] == f'{lab}:{cond}']
            for lang in ('en', 'ko', 'all'):
                sel = [r for r in rs if lang == 'all' or r['lang'] == lang]
                res['by'][f'{lab}|{cond}|{lang}'] = block(sel)
                res['by'][f'{lab}|{cond}|{lang}|task_objects'] = block([r for r in sel if r['class'] in TASK])
                for t in ('Q1', 'Q2', 'Q3', 'Q5'):
                    res['by'][f'{lab}|{cond}|{lang}|{t}'] = block([r for r in sel if r['type'] == t])
            res['timing'][f'{lab}|{cond}'] = timing(ROOT / 'results' / runs[lab], cond)
        for lang in ('en', 'ko'):
            res['paired'][f'{cond}|{lang}'] = S.paired(rows, f'{labels[1]}:{cond}', f'{labels[0]}:{cond}', lang=lang)
        ka = {(r['qid'], r['layout']): r for r in rows if r['cond'] == f'{labels[0]}:{cond}'}
        kb = {(r['qid'], r['layout']): r for r in rows if r['cond'] == f'{labels[1]}:{cond}'}
        byq = defaultdict(lambda: [0, 0, 0, 0, '', '', ''])  # a_only, b_only, both, neither, lang, class, type
        for k in ka:
            x, y = ka[k]['success'], kb[k]['success']
            byq[k[0]][0 if x and not y else 1 if y and not x else 2 if x else 3] += 1
            byq[k[0]][4:] = [ka[k]['lang'], ka[k]['class'], ka[k]['type']]
        res['diff_queries'][cond] = sorted(([q, *v] for q, v in byq.items() if v[0] or v[1]), key=lambda v: -(v[1] + v[2]))
    (out / 'compare.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    A, B = labels
    L = [f'# X6 비교 {A} ({runs[A]}) vs {B} ({runs[B]}) — dev, 탐색', '',
         '- 자동 생성 (`explore/X6_model_size/x6_compare.py`). 채점 규칙은 EXP110 등록 규칙과 같다.',
         '- 실패 분류 (실패한 질의만, 서로 겹치지 않음): invalid = 출력 형식 오류, miss_only = 정답 일부/전부 누락, wrong_only = 정답은 다 찾았지만 잘못된 객체 추가, miss_and_wrong = 둘 다.', '']
    for cond in CONDS:
        L += [f'## {cond}', '', f'| 지표 | {A} en | {B} en | {A} ko | {B} ko |', '|---|---|---|---|---|']
        for key in ('n', 'success_rate', 'success', 'invalid', 'miss_only', 'wrong_only', 'miss_and_wrong', 'instance_recall', 'false_positive_boxes',
                    'multi_n', 'multi_success', 'multi_partial_miss', 'multi_total_miss', 'multi_instance_recall', 'absent_n', 'absent_hallucinated'):
            L.append(f'| {key} | ' + ' | '.join(str(res['by'][f'{m}|{cond}|{lg}'].get(key, 0)) for lg, m in (('en', A), ('en', B), ('ko', A), ('ko', B))) + ' |')
        L += ['', f'소형 팔레트·블록 질의 (task objects)', '', f'| 지표 | {A} en | {B} en | {A} ko | {B} ko |', '|---|---|---|---|---|']
        for key in ('n', 'success', 'instance_recall', 'false_positive_boxes', 'invalid', 'miss_only', 'wrong_only', 'miss_and_wrong'):
            L.append(f'| {key} | ' + ' | '.join(str(res['by'][f'{m}|{cond}|{lg}|task_objects'].get(key, 0)) for lg, m in (('en', A), ('en', B), ('ko', A), ('ko', B))) + ' |')
        L += ['', f'짝 비교 ({B} − {A}, 질의 묶음 bootstrap 95% CI가 주 근거, McNemar는 참고)', '']
        for lang in ('en', 'ko'):
            p = res['paired'][f'{cond}|{lang}']
            L.append(f"- {lang}: 차이 {p['diff']:+.3f}, CI {p['ci95_query_bootstrap']}, {B}만 성공 {p['a_only']}, {A}만 성공 {p['b_only']}, 짝 {p['pairs']}개, McNemar p={p['mcnemar_p']}")
        t = res['timing']
        L += ['', '| 시간·메모리 | ' + ' | '.join((A, B)) + ' |', '|---|---|---|']
        for key in ('model_calls', 'latency_median_s', 'latency_p90_s', 'latency_max_s', 'latency_sum_s', 'generated_tokens_median', 'hit_max_new_tokens', 'peak_vram_gib_reported'):
            L.append(f'| {key} | ' + ' | '.join(str(t[f'{m}|{cond}'][key]) for m in (A, B)) + ' |')
        L += ['', f'성공이 갈린 질의 ({A}만 성공 / {B}만 성공 배치 수)', '', '| 질의 | 언어 | 클래스 | 유형 | ' + f'{A}만 | {B}만 | 둘 다 | 둘 다 실패 |', '|---|---|---|---|---|---|---|---|']
        for v in res['diff_queries'][cond]:
            L.append(f'| {v[0]} | {v[5]} | {v[6]} | {v[7]} | {v[1]} | {v[2]} | {v[3]} | {v[4]} |')
        L.append('')
    (out / 'COMPARE.md').write_text('\n'.join(L) + '\n')
    print('written', out)


if __name__ == '__main__':
    sys.exit(main())
