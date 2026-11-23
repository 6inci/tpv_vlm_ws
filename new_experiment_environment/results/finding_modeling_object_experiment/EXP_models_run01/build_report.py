"""Combine recorded outputs and visual review into one standalone HTML."""
import base64,io,json,html,hashlib
from pathlib import Path
from PIL import Image,ImageOps
RUN=Path(__file__).resolve().parent; ROOT=RUN.parent
MODELS=['qwen_vlm','yolo_world','grounding_dino','sam2']
COLORS={'pallet':'#ff7043','water bottle':'#29b6f6','shelf':'#66bb6a'}
NAMES={'pallet':'팔레트','water bottle':'물병','shelf':'선반'}
# INFERRED visual review, not human-validated ground-truth annotations.
REF={'IMG_8557':{'pallet':True,'water bottle':False,'shelf':True},'IMG_8558':{'pallet':True,'water bottle':False,'shelf':False},'IMG_8559':{'pallet':True,'water bottle':False,'shelf':True},'IMG_8560':{'pallet':True,'water bottle':True,'shelf':True},'IMG_8561':{'pallet':True,'water bottle':True,'shelf':True}}
NOTES={'IMG_8557':'앞의 큰 흰색 슬랫 팔레트와 뒤의 흰색 선반이 대상이다. 배경의 정수기는 물병과 구분해야 한다. Qwen은 팔레트와 선반에 위치가 맞는 박스를 만들었다. YOLO의 배경 선반 박스는 오탐으로 보인다.', 'IMG_8558':'흰색 IBC 탱크 모형과 팔레트가 보인다. Qwen은 실제 팔레트를 찾았지만 IBC 탱크도 팔레트로 잡았다. IBC 탱크는 물병도 선반도 아니다. Grounding DINO가 탱크를 선반으로, 탱크 부근 작은 부분을 물병으로 잡은 것은 오탐으로 보인다.', 'IMG_8559':'원형 테이블 위 팔레트와 작은 선반이 보인다. 오른쪽 IBC 탱크 모형은 물병이 아니다. Qwen은 배경 서랍장만 선반으로 잡고 테이블 위 팔레트와 작은 선반을 놓쳤다. YOLO와 DINO 모두 작은 흰색 선반을 놓치고 배경 서랍장/캐비닛을 선반으로 잡았다. DINO의 팔레트 박스는 오른쪽 검은 장비 케이스를 가리켜 위치 오탐이다. 테이블 위 실제 팔레트를 찾지 못했다.', 'IMG_8560':'팔레트, 선반, 투명 물병이 책상 위에 보인다. Qwen은 물병과 실제 선반을 찾지만 팔레트를 놓치고 IBC 탱크도 선반으로 잡았다. DINO는 세 대상에 위치가 맞는 박스를 만든다. YOLO는 팔레트를 놓치며 왼쪽 컵을 물병으로 잡고 넓은 배경 선반 박스를 추가한다.', 'IMG_8561':'팔레트, 흰색 선반, 가운데 투명 물병이 보인다. Qwen은 가운데 실제 물병을 찾지만 왼쪽 컵도 물병으로 잡고 팔레트와 선반은 놓쳤다. DINO는 실제 팔레트를 찾지만 IBC 탱크와 흰색 선반에도 팔레트 박스를 만든다. 선반 박스는 책상 대부분을 덮어 실제 작은 선반의 위치를 찾았다고 보기 어렵다. 왼쪽 컵을 물병으로 잡은 오탐도 있다. YOLO는 가운데 실제 물병만 잡았다.'}
def uri(image,fmt='JPEG'):
 b=io.BytesIO(); image.save(b,format=fmt); return 'data:image/'+('jpeg' if fmt=='JPEG' else 'png')+';base64,'+base64.b64encode(b.getvalue()).decode()
inventory=json.loads((RUN/'inventory.json').read_text()); bundle={'inventory':inventory,'visual_reference':{'evidence':'INFERRED','definition':'물류용 pallet, 일반 물병, 선반/랙; IBC 탱크는 물병에서 제외','presence':REF,'notes':NOTES},'results':{},'failed_attempts':{'qwen_first_image':{'raw_response':(RUN/'qwen_vlm_IMG_8557_raw.txt').read_text(),'error':'schema validation failed: invalid composite label pallet|water bottle|shelf; original response retained; retry uses explicit single-label instruction'}}}
rows=[]; sections=[]; totals={m:0 for m in MODELS}
for p in sorted(ROOT.glob('*.jpeg')):
 image=ImageOps.exif_transpose(Image.open(p)).convert('RGB'); w,h=image.size
 thumb=image.copy(); thumb.thumbnail((1000,1000)); src=uri(thumb)
 cards=[]
 for m in MODELS:
  record=json.loads((RUN/f'{m}_{p.stem}.json').read_text()); bundle['results'][f'{m}/{p.name}']=record
  raw=RUN/f'{m}_{p.stem}_attempt02_raw.txt'
  if raw.exists(): record['raw_response']=raw.read_text()
  ds=record['detections']; totals[m]+=len(ds); boxes=[]; masks=[]; counts=[]
  for c in COLORS:
   detected=[d for d in ds if d['label']==c]; ref=REF[p.stem][c]
   state='탐지' if detected else '미탐지'; review='존재 참고와 일치' if bool(detected)==ref else ('오탐 후보' if detected else '누락 후보')
   color='#f5a623' if bool(detected)!=ref else '#a5d6a7'
   counts.append(f'<span style="color:{color}">{NAMES[c]} {len(detected)} · {review}</span>')
   rows.append(f'<tr><td>{p.name}</td><td>{m}</td><td>{NAMES[c]}</td><td>{state} ({len(detected)})</td><td>{"보임" if ref else "보이지 않음"} · INFERRED</td><td style="color:{color}">{review}</td></tr>')
  for i,d in enumerate(ds):
   x1,y1,x2,y2=d['bbox_px']; color=COLORS[d['label']]; score=d.get('score'); label=f'{i+1} {d["label"]}'+(f' {score:.3f}' if score is not None else '')
   boxes.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{color}" stroke-width="14"/><text x="{x1+8}" y="{max(65,y1-20)}" fill="{color}" stroke="#111" stroke-width="3" paint-order="stroke" font-size="65">{html.escape(label)}</text>')
   if m=='sam2':
    mask=Image.open(RUN/d['mask_path']).convert('L'); mask.thumbnail(thumb.size)
    tint=Image.new('RGBA',mask.size,color); tint.putalpha(mask.point(lambda v: int(v*.35)))
    masks.append(f'<image width="{w}" height="{h}" href="{uri(tint,"PNG")}"/>')
  svg=f'<svg viewBox="0 0 {w} {h}" role="img"><image width="{w}" height="{h}" href="{src}"/>{"".join(masks)}{"".join(boxes)}</svg>'
  cards.append(f'<article><h3>{m}{" (DINO 박스 → 분할)" if m=="sam2" else ""}</h3>{svg}<p>{"<br>".join(counts)}</p><small>추론 {record["seconds"]:.2f}초 · 박스 {len(ds)}개</small><details><summary>원문/박스/점수</summary><pre>{html.escape(json.dumps(record,ensure_ascii=False,indent=2))}</pre></details></article>')
 sections.append(f'<section><h2>{p.name}</h2><p>{NOTES[p.stem]}</p><div class="grid">{"".join(cards)}</div></section>')
unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in inventory['original_sha256'].items()); assert unchanged
bundle['provenance']=json.loads((RUN/'provenance.json').read_text())
bundle['verification']={'git_commit':'UNKNOWN: no accessible git repository at source workspace','original_files_unchanged':unchanged,'model_image_results':len(bundle['results']),'status':'TESTING','accuracy_metrics':'not computed: no verified ground-truth boxes or masks','visual_reference_is_not_ground_truth':True}
assert len(bundle['results'])==20
for r in bundle['results'].values():
 w,h=r['upright_size']
 for d in r['detections']:
  x1,y1,x2,y2=d['bbox_px']; assert 0<=x1<x2<=w+1 and 0<=y1<y2<=h+1
style='body{margin:0;background:#101820;color:#e8edf2;font-family:system-ui,sans-serif}main{max-width:1700px;margin:auto;padding:28px}h1{font-size:30px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}article{background:#1d2b38;border:1px solid #405367;border-radius:12px;padding:12px}svg{width:100%;background:#000}small{color:#b0bec5}section{margin:32px 0}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #405367;padding:8px;text-align:left}pre{white-space:pre-wrap;max-height:500px;overflow:auto;font-size:11px}.note{padding:16px;background:#293947;border-radius:10px}button{padding:12px;background:#90caf9;border:0;border-radius:6px;cursor:pointer}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.grid{grid-template-columns:1fr}}'
payload=json.dumps(bundle,ensure_ascii=False).replace('</','<\\/')
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>모델별 팔레트·물병·선반 실험</title><style>{style}</style><main><h1>모델별 팔레트 · 물병 · 선반 탐지 비교</h1><p>사진 5장 × 모델 4종 · CPU · {inventory['timestamp']} · EXP_models_run01</p><div class="note"><b>실행 완료 ≠ 정확도 PASS.</b> Qwen 첫 시도는 라벨 schema 오류로 실패해 원문을 보존하고, 단일 라벨 선택을 명시한 프롬프트로 재실행했다. 최초/재시도 프롬프트와 원문은 통합 JSON에 포함된다. 모든 모델 출력은 CONFIRMED 실행 기록이며, 육안 존재 참고와 오류 해석은 INFERRED다. 검증된 정답 박스/마스크가 없어 precision, recall, mAP, mask IoU를 계산하지 않았다. ‘존재 참고와 일치’는 박스 위치·개수의 정확성을 보장하지 않는다.<br>팔레트는 물류용 pallet이며 palette가 아니다. IBC 탱크는 물병에서 제외했다. SAM2는 독립 의미 탐지 모델이 아니므로 DINO 오탐/누락을 이어받는다. SAM 마스크 점수는 분할 품질 추정치이며 객체 존재 확률이 아니다.<br>모든 사진은 EXIF 회전을 적용한 직립 이미지 기준 픽셀 좌표다. 왜곡 보정/metric 변환은 하지 않았다. 기존 파일 SHA256 보존 확인: {unchanged}.</div><p><b>육안 비교 요약 (INFERRED):</b> YOLO-World는 이 설정에서 5장 모두 팔레트를 놓쳤다. Grounding DINO는 가까운 팔레트에 강하지만 넓은 장면에서 오탐·위치 오류가 나온다. Qwen도 작은 모형과 IBC 탱크에서 누락·오분류가 나온다. SAM2는 IMG_8560의 올바른 DINO 박스로 세 대상의 윤곽을 잘 분할하지만 탐지 오류 자체는 해결하지 않는다.</p><p>주황: 팔레트 · 파랑: 물병 · 초록: 선반. 점수는 모델별 의미가 달라 직접 비교하지 않는다.</p><button onclick="download()">통합 원시 결과 JSON 내려받기</button><details><summary>설정 · 모델 · 실행 환경</summary><pre>{html.escape(json.dumps(inventory,ensure_ascii=False,indent=2))}</pre><p>Qwen3-VL-4B-Instruct / YOLOv8s-World / Grounding DINO tiny / SAM2.1 Hiera tiny. 로컬 가중치만 사용. Qwen 입력은 직립 이미지 672×896 이내 축소, SAM2 입력은 1024×1024 이내 축소. Grounding DINO와 YOLO는 각 processor 내부 리사이즈 적용.</p><p><a href="https://huggingface.co/docs/transformers/model_doc/grounding-dino">Grounding DINO 공식 API</a> · <a href="https://huggingface.co/docs/transformers/model_doc/sam2">SAM2 공식 API</a></p></details>{''.join(sections)}<h2>사진 × 모델 × 객체 존재 비교</h2><p>아래 판정은 정답 주석이 아닌 육안 참고다. 박스 위치는 위 시각화에서 확인한다.</p><div style="overflow:auto"><table><tr><th>사진</th><th>모델</th><th>대상</th><th>모델 출력</th><th>육안 참고</th><th>존재 비교</th></tr>{''.join(rows)}</table></div><p>모델별 출력 박스 수: {html.escape(str(totals))}. SAM2 개수는 DINO 프롬프트 개수이며 독립 탐지 성능이 아니다.</p><script id="raw-data" type="application/json">{payload}</script><script>function download(){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([document.getElementById('raw-data').textContent],{{type:'application/json'}}));a.download='experiment_results.json';a.click();URL.revokeObjectURL(a.href);}}</script></main></html>'''
with (RUN/'model_comparison_report.html').open('x') as f:f.write(page)
with (RUN/'verification.json').open('x') as f:json.dump(bundle['verification'],f,indent=2)
print('Verified 20 model-image results; original hashes unchanged; report bytes',len(page.encode()))
