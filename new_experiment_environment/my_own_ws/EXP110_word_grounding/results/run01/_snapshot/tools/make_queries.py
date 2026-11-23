#!/usr/bin/env python3
"""Build the EXP110 query set deterministically from the object registry + config.json -> queries.json.

Each query: id, type (Q1 name | Q2 attribute | Q3 synonym | Q5 absent), lang (ko|en), text,
target_labels (GT labels that count as correct), target_variant (None = any variant),
ignore_labels (GT labels neither required nor penalised for this query), source_class.
Usage: python tools/make_queries.py [--selftest]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config.json').read_text())


def build(registry, qcfg):
    out, seen = [], set()

    def add(qtype, lang, text, targets, variant, ignore, cls):
        if (lang, text) in seen:  # same string twice -> keep the first definition
            return
        seen.add((lang, text))
        out.append({'id': f'q{len(out) + 1:03d}', 'type': qtype, 'lang': lang, 'text': text,
                    'target_labels': targets, 'target_variant': variant, 'ignore_labels': ignore, 'source_class': cls})

    for c in registry['classes']:
        k = c['key']
        targets = qcfg['multi_class_targets'].get(k, [k])
        ignore = qcfg['query_ignore'].get(k, [])
        add('Q1', 'ko', c['ko'], targets, None, ignore, k)
        add('Q1', 'en', c['en'], targets, None, ignore, k)
        for v in c.get('variants') or []:
            ko_col, en_col = qcfg['colors'][v]
            add('Q2', 'ko', f"{ko_col} {c['ko']}", [k], v, ignore, k)
            add('Q2', 'en', f"{en_col} {c['en']}", [k], v, ignore, k)
        for s in c.get('syn_ko', []):
            add('Q3', 'ko', s, targets, None, ignore, k)
        for s in c.get('syn_en', []):
            add('Q3', 'en', s, targets, None, ignore, k)
    for ko, en in qcfg['negatives']:
        add('Q5', 'ko', ko, [], None, [], None)
        add('Q5', 'en', en, [], None, [], None)
    return out


def selftest():
    reg = {'classes': [
        {'key': 'drum', 'ko': '드럼통', 'en': 'drum', 'syn_ko': ['드럼'], 'syn_en': ['barrel'], 'variants': ['blue']},
        {'key': 'pallet', 'ko': '팔레트', 'en': 'pallet', 'syn_ko': [], 'syn_en': []},
        {'key': 'storage_rack', 'ko': '선반', 'en': 'rack', 'syn_ko': ['선반'], 'syn_en': []}]}
    q = build(reg, CFG['queries'])
    by = {(x['lang'], x['text']): x for x in q}
    assert by[('en', 'blue drum')]['target_variant'] == 'blue' and by[('ko', '파란 드럼통')]['type'] == 'Q2'
    assert by[('en', 'pallet')]['target_labels'] == ['pallet', 'small_pallet']
    assert by[('ko', '선반')]['type'] == 'Q1' and by[('ko', '선반')]['ignore_labels'] == ['storage_cubby']  # duplicate synonym dropped
    assert by[('en', 'forklift')]['target_labels'] == [] and len({x['id'] for x in q}) == len(q)
    print('selftest PASS')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    reg = json.loads(Path(CFG['registry']).read_text())
    q = build(reg, CFG['queries'])
    (ROOT / 'queries.json').write_text(json.dumps({'registry': CFG['registry'], 'n': len(q), 'queries': q}, indent=1, ensure_ascii=False))
    from collections import Counter
    print('queries', len(q), dict(Counter((x['type'], x['lang']) for x in q)))


if __name__ == '__main__':
    sys.exit(main())
