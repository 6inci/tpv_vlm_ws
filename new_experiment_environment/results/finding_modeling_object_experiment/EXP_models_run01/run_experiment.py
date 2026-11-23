"""Offline local models only. Originals and existing reports are read-only."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', YOLO_CONFIG_DIR='/tmp/object_detection_ultralytics', MPLCONFIGDIR='/tmp/object_detection_mpl')
import sys, json, time, hashlib, math, traceback, base64, io, html
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image, ImageOps, ImageDraw
ROOT = Path(__file__).resolve().parent.parent
RUN = Path(__file__).resolve().parent
CLASSES = ['pallet','water bottle','shelf']
COLORS = ['#ff7043','#29b6f6','#66bb6a']
PROMPT = '''Find visible miniature or full-size logistics pallets, water bottles, and storage shelves/racks in this image. A pallet is a slatted platform for carrying goods, not a painting palette. White 3D-printed models count. Return JSON only: {"objects":[{"label":"pallet","bbox_2d":[x1,y1,x2,y2]}]}. Each label must be exactly one of "pallet", "water bottle", or "shelf". The example label "pallet" is only an example, choose the correct single label for each object. Coordinates are integers normalized to 0..1000 relative to the entire displayed upright image. Include every distinct visible target; no unrelated objects. If none, return an empty objects list.'''

def save(name,data):
    with (RUN/name).open('x') as f: json.dump(data,f,ensure_ascii=False,indent=2)

def load_image(p): return ImageOps.exif_transpose(Image.open(p)).convert('RGB')

def validate(obj,size):
    assert isinstance(obj,dict) and isinstance(obj.get('objects'),list)
    result=[]
    for d in obj['objects']:
        assert isinstance(d,dict) and d.get('label') in CLASSES
        b=d.get('bbox_2d'); assert isinstance(b,list) and len(b)==4
        assert all(type(v) in (float,int) and math.isfinite(v) and 0<=v<=1000 for v in b)
        assert b[0]<b[2] and b[1]<b[3]
        result.append({'label':d['label'],'bbox_px':[b[i]*size[i%2]/1000 for i in range(4)],'bbox_normalized_1000':b})
    return result

def category(label):
    if 'pallet' in label: return 'pallet'
    if 'bottle' in label: return 'water bottle'
    if any(x in label for x in ['shelf','shelves','rack']): return 'shelf'
    return None

def main(stage):
    import torch
    torch.set_num_threads(8)
    images=sorted(ROOT.glob('*.jpeg'))
    if stage=='inventory':
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file()}
        save('inventory.json',{'timestamp':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'experiment_id':'EXP_models','run_id':'run01','original_sha256':files,'images':[{'name':p.name,'stored_size':Image.open(p).size,'exif_orientation':Image.open(p).getexif().get(274),'upright_size':load_image(p).size} for p in images],'device':'cpu','torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'coordinate_system':'EXIF-transposed upright image pixels; distorted original imagery; no metric conversion','ground_truth':'unavailable','parameters':{'classes':CLASSES,'yolo_conf':0.1,'yolo_iou':0.5,'yolo_imgsz':1280,'dino_threshold':0.25,'dino_text_threshold':0.25,'qwen_max_new_tokens':900,'qwen_max_pixels':602112,'sam_prompt_source':'Grounding DINO boxes'},'qwen_prompt':PROMPT})
        return
    if stage=='qwen_vlm':
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        model_id='Qwen/Qwen3-VL-4B-Instruct'
        processor=AutoProcessor.from_pretrained(model_id,local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(model_id,local_files_only=True,dtype=torch.bfloat16).eval()
    elif stage=='grounding_dino':
        from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
        model_id='IDEA-Research/grounding-dino-tiny'
        processor=AutoProcessor.from_pretrained(model_id,local_files_only=True)
        model=AutoModelForZeroShotObjectDetection.from_pretrained(model_id,local_files_only=True).eval()
    elif stage=='sam2':
        from transformers import Sam2Processor,Sam2Model
        model_id='facebook/sam2.1-hiera-tiny'
        processor=Sam2Processor.from_pretrained(model_id,local_files_only=True)
        model=Sam2Model.from_pretrained(model_id,local_files_only=True).eval()
    elif stage=='yolo_world':
        from ultralytics import YOLOWorld
        model_id='/home/binci/vlm_ws/new_experiment_environment/results/YOLO_WORLD_INTEGRATION_20260929T012518+0900/07_yolo_world/yolov8s-world.pt'
        model=YOLOWorld(model_id)
        # Reuse installed CLIP and read-only local checkpoint; avoid automatic download.
        import ultralytics.nn.text_model as tm
        original=tm.clip.load
        def local_clip(name,*args,**kwargs):
            kwargs['download_root']='/home/binci/weights/clip'
            return original(name,*args,**kwargs)
        tm.clip.load=local_clip
        model.set_classes(CLASSES)
    for p in images:
        target=RUN/f'{stage}_{p.stem}.json'
        if target.exists(): continue
        image=load_image(p); started=time.perf_counter(); masks=[]
        print(stage,p.name,'start',flush=True)
        if stage=='qwen_vlm':
            small=image.copy(); small.thumbnail((672,896))
            messages=[{'role':'user','content':[{'type':'image','image':small},{'type':'text','text':PROMPT}]}]
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
            with torch.inference_mode(): output=model.generate(**inputs,max_new_tokens=900,do_sample=False)
            raw=processor.decode(output[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
            (RUN/f'{stage}_{p.stem}_attempt02_raw.txt').write_text(raw)
            text=raw.strip()
            if text.startswith('```'): text=text.split('\n',1)[1].rsplit('```',1)[0]
            detections=validate(json.loads(text),image.size)
        elif stage=='yolo_world':
            r=model.predict(image,device='cpu',imgsz=1280,conf=0.1,iou=0.5,verbose=False)[0]
            detections=[{'label':r.names[int(c)],'bbox_px':b.tolist(),'score':float(s)} for b,s,c in zip(r.boxes.xyxy,r.boxes.conf,r.boxes.cls)]
        elif stage=='grounding_dino':
            inputs=processor(images=image,text='a pallet. a water bottle. a storage shelf.',return_tensors='pt')
            with torch.inference_mode(): out=model(**inputs)
            r=processor.post_process_grounded_object_detection(out,inputs.input_ids,threshold=0.25,text_threshold=0.25,target_sizes=[image.size[::-1]])[0]
            labels=r.get('text_labels',r.get('labels'))
            detections=[{'label':category(l),'raw_label':l,'bbox_px':b.tolist(),'score':float(s)} for b,s,l in zip(r['boxes'],r['scores'],labels) if category(l)]
        else:
            detections=json.loads((RUN/f'grounding_dino_{p.stem}.json').read_text())['detections']
            if detections:
                # Process resized upright image to bound memory; prompt scaling is deterministic.
                small=image.copy(); small.thumbnail((1024,1024)); sx,sy=small.width/image.width,small.height/image.height
                boxes=[[v*(sx if i%2==0 else sy) for i,v in enumerate(d['bbox_px'])] for d in detections]
                inputs=processor(images=small,input_boxes=[boxes],return_tensors='pt')
                with torch.inference_mode(): out=model(**{k:v for k,v in inputs.items() if k in ['pixel_values','input_boxes']},multimask_output=False)
                mask_tensor=processor.post_process_masks(out.pred_masks.cpu(),inputs['original_sizes'])[0]
                for i,d in enumerate(detections):
                    mask=Image.fromarray((mask_tensor[i,0].numpy().astype('uint8')*255)).resize(image.size,Image.Resampling.NEAREST)
                    mask.save(RUN/f'sam2_{p.stem}_{i}.png'); masks.append(mask)
                    d['sam_mask_quality']=float(out.iou_scores.flatten()[i]); d['mask_path']=f'sam2_{p.stem}_{i}.png'
                    d['mask_area_px']=sum(mask.histogram()[1:]); d['prompt_source']='grounding_dino'
        save(target.name,{'status':'TESTING','model':model_id,'image':p.name,'upright_size':image.size,'coordinate_system':'upright distorted image pixels','seconds':time.perf_counter()-started,'prompt':PROMPT if stage=='qwen_vlm' else None,'detections':detections})
        print(stage,p.name,'done',len(detections),round(time.perf_counter()-started,2),flush=True)

if __name__=='__main__':
    assert validate({'objects':[{'label':'pallet','bbox_2d':[100,200,500,800]}]},(300,400))[0]['bbox_px']==[30,80,150,320]
    main(sys.argv[1])
