#!/usr/bin/env python3
"""EXP111 pilot (dev only): end-to-end structured output and image-level check (no new model calls).

Combines saved outputs: pallets = s3A rows answered true (false positives kept); blocks = s4 English query boxes.
Structured output per image (raw distorted pixels 1920x1080):
  {"pallets": [{"id", "box_raw_px", "state": "LOADED"|"EMPTY", "block_colors": [...], "blocks": [{"color", "box_raw_px"}]}]}
Rules (Python, fixed before this evaluation was computed):
  state  = config s4_blocks_on_predicted.state_rule (any English block-query box center inside the pallet box)
  colors = English "red block" / "blue block" boxes whose center lies inside the pallet box
Image is TASK-READY only if: every answer pallet is found, no extra (false) pallet, every state correct, and for
every loaded pallet the predicted colour set equals the answer colour set. Answer keys are used only here.
Usage: python tools/pipeline_eval.py --s3 <s3A dir> --s4 <s4 dir> --out <new dir>
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXP110 = HERE.parent / 'EXP110_word_grounding'
sys.path.insert(0, str(EXP110 / 'tools'))
import score as S  # noqa: E402

EN_BLOCK_Q = {'q029': None, 'q031': 'blue', 'q033': 'red', 'q035': None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s3', required=True)
    ap.add_argument('--s4', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    s3 = json.loads((Path(a.s3) / 'rows.json').read_text())
    s4 = json.loads((Path(a.s4) / 'rows.json').read_text())
    report, structured = [], {}
    for layout in sorted({r['layout'] for r in s3}):
        stem = sorted((EXP110 / 'annotations' / 'pilot').glob(f'{layout}_*.json'))[0].stem
        gt = S.load_gt(stem)['objects']
        gp = [o['box_xyxy'] for o in gt if o['label'] == 'small_pallet']
        gb = [o for o in gt if o['label'] == 'block']
        preds = [r['box_raw_px'] for r in s3 if r['layout'] == layout and r['yes']]
        blk = {qid: next(r['boxes_raw_px'] for r in s4 if r['layout'] == layout and r['qid'] == qid) for qid in EN_BLOCK_Q}
        pallets = []
        for i, p in enumerate(preds, 1):
            inside = lambda bs: [b for b in bs if S.center_in(b, p)]
            any_blk = [b for qid in EN_BLOCK_Q for b in inside(blk[qid])]
            colored = [{'color': c, 'box_raw_px': b} for qid, c in EN_BLOCK_Q.items() if c for b in inside(blk[qid])]
            pallets.append({'id': i, 'box_raw_px': p, 'state': 'LOADED' if any_blk else 'EMPTY',
                            'block_colors': sorted({b['color'] for b in colored}), 'blocks': colored})
        structured[layout] = {'image_stem': stem, 'image_space': 'raw distorted px 1920x1080', 'pallets': pallets}
        checks, extra = [], [q for q in pallets if not any(S.iou(q['box_raw_px'], g) >= 0.5 or S.center_in(q['box_raw_px'], g) for g in gp)]
        for g in gp:
            true_colors = sorted({o.get('variant') for o in gb if S.center_in(o['box_xyxy'], g)})
            true_state = 'LOADED' if true_colors else 'EMPTY'
            m = [q for q in pallets if S.iou(q['box_raw_px'], g) >= 0.5 or S.center_in(q['box_raw_px'], g)]
            q = m[0] if m else None
            checks.append({'gt_pallet': g, 'found': q is not None, 'true_state': true_state, 'pred_state': q and q['state'],
                           'state_ok': bool(q) and q['state'] == true_state, 'true_colors': true_colors,
                           'pred_colors': q and q['block_colors'],
                           'colors_ok': bool(q) and (true_state == 'EMPTY' or q['block_colors'] == true_colors)})
        ready = all(c['found'] and c['state_ok'] and c['colors_ok'] for c in checks) and not extra
        report.append({'layout': layout, 'n_gt_pallets': len(gp), 'n_pred_pallets': len(pallets), 'n_extra_pallets': len(extra),
                       'extra_pallets_state': [q['state'] for q in extra], 'checks': checks, 'task_ready': ready})
        print(layout, '| found', sum(c['found'] for c in checks), '/', len(gp), '| extra', len(extra), [q['state'] for q in extra],
              '| state ok', sum(c['state_ok'] for c in checks), '| colors ok', sum(c['colors_ok'] for c in checks),
              '| colors', [(c['true_colors'], c['pred_colors']) for c in checks if c['true_state'] == 'LOADED'], '| TASK-READY', ready)
    (out / 'structured_output.json').write_text(json.dumps(structured, indent=1))
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    (out / 'INPUTS.json').write_text(json.dumps({'s3': a.s3, 's4': a.s4}, indent=1))
    print('task-ready images', sum(r['task_ready'] for r in report), '/', len(report), '| written', out)


if __name__ == '__main__':
    sys.exit(main())
