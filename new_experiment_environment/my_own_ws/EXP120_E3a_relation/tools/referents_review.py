#!/usr/bin/env python3
"""EXP120: draw the referent-draft review sheets for the user (one image per dev layout, CPU only).

Left: the dev image with the candidate small pallets numbered exactly as the experiment numbers them (left to right),
and the reference objects (truck, charger) outlined in white. Right: zoomed pallets with their answer-key state,
then every query with the draft answer and its review flag (clear / judgment, from referents_draft.json review_flags).
Usage: <qwen_vl python> tools/referents_review.py --out review
"""
import argparse, json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / 'EXP110_word_grounding' / 'tools'))
import score as S  # noqa: E402
import run_models as R  # noqa: E402

CFG = json.loads((HERE / 'config.json').read_text())
REF = json.loads((HERE / 'referents_draft.json').read_text())
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
PAL = [(255, 64, 64), (64, 200, 255), (255, 210, 0), (0, 230, 120)]
KO = {'e01': '트럭에 가장 가까운 소형 팔레트', 'e02': '충전소 옆 소형 팔레트', 'e03': '빈 소형 팔레트',
      'e04': '블록이 있는 소형 팔레트', 'e05': '파란 블록이 있는 소형 팔레트', 'e06': '지게차 옆 소형 팔레트'}
STATE_KO = {'LOADED': '적재', 'EMPTY': '빈 팔레트'}
COLOR_KO = {'red': '빨강', 'blue': '파랑'}


def answer_label(truth, pallets):
    if isinstance(truth, list):
        return str(next(i for i, b in enumerate(pallets, 1) if S.center_in(b, truth)))
    return {'none': '없음', 'unknown': '확인 불가(기준 물체 없음)'}[truth]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out = HERE / ap.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    f = {s: ImageFont.truetype(FONT, s) for s in (22, 26, 30, 36)}
    for img_path in R.images_for('dev'):
        lay = img_path.stem[:3]
        img = Image.open(img_path).convert('RGB')
        gt = S.load_gt(img_path.stem)['objects']
        pallets = sorted([o['box_xyxy'] for o in gt if o['label'] == 'small_pallet'], key=lambda b: b[0])
        blocks = [o for o in gt if o['label'] == 'block']
        canvas = Image.new('RGB', (img.width + 900, img.height), (24, 24, 24))
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)
        for o in gt:
            if o['label'] in ('truck', 'charger'):
                d.rectangle(o['box_xyxy'], outline=(255, 255, 255), width=2)
                d.text((o['box_xyxy'][0], o['box_xyxy'][3] + 2), o['label'], fill=(255, 255, 255), font=f[22])
        x0, y = img.width + 24, 16
        d.text((x0, y), f'{lay}  정답 초안 검토 (Claude 초안, 사용자 확인 필요)', fill=(255, 255, 255), font=f[30])
        y += 52
        for i, b in enumerate(pallets, 1):
            col = PAL[i - 1]
            d.rectangle(b, outline=col, width=3)
            d.text((b[0], b[1] - 34), str(i), fill=col, font=f[30], stroke_width=3, stroke_fill=(0, 0, 0))
            on = sorted({o.get('variant') for o in blocks if S.center_in(o['box_xyxy'], b)})
            m = 25
            crop = img.crop((int(b[0]) - m, int(b[1]) - m, int(b[2]) + m, int(b[3]) + m)).resize((200, 200), Image.LANCZOS)
            cx = x0 + (i - 1) * 430
            canvas.paste(crop, (cx, y))
            d.rectangle([cx, y, cx + 199, y + 199], outline=col, width=3)
            state = 'LOADED' if on else 'EMPTY'
            txt = f'{i}: {STATE_KO[state]}' + (f" ({', '.join(COLOR_KO[c] for c in on)})" if on else '')
            d.text((cx + 210, y + 80), txt, fill=col, font=f[26])
        y += 230
        flags = REF.get('review_flags', {}).get(lay, {})
        for q in CFG['queries']:
            fl = flags.get(q['qid'], {'flag': 'clear'})
            judg = fl['flag'] == 'judgment'
            col = (255, 170, 40) if judg else (140, 230, 140)
            d.text((x0, y), f"{q['qid']} {KO[q['qid']]}", fill=(230, 230, 230), font=f[26])
            d.text((x0 + 24, y + 34), f"\"{q['en']}\"", fill=(170, 170, 170), font=f[22])
            ans = answer_label(REF[lay][q['qid']], pallets)
            d.text((x0 + 24, y + 64), f"초안 정답: {ans}" + (f"   [판단 필요, 대안: {fl.get('alt')}]" if judg else '   [명확]'), fill=col, font=f[26])
            y += 104
            if fl.get('why'):
                why = fl['why']
                for k in range(0, len(why), 36):
                    d.text((x0 + 24, y - 6), why[k:k + 36], fill=(200, 200, 200), font=f[22])
                    y += 28
                y += 8
        canvas.save(out / f'referents_review_{lay}.jpg', quality=92)
        print('wrote', out / f'referents_review_{lay}.jpg')


if __name__ == '__main__':
    sys.exit(main())
