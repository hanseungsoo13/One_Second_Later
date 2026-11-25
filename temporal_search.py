"""Extract CUCPS + motion scores, fit a scalar reranker, and evaluate it."""
import argparse,json,math,sys
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch,yaml
from scipy.io import loadmat
from PIL import Image
from torchvision.ops import box_iou
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from osl.data import AnnotationDataset
from osl.model import DinoBackbone,Model
from osl.temporal_search import location_scores,search_metrics

def ann_boxes(path):
 data=loadmat(path)
 for k in ('box_new','anno_file','anno_previous'):
  if k in data: arr=np.asarray(data[k]);break
 else:raise ValueError(f'No annotations in {path}')
 out=[]
 for ident,x,y,w,h in arr:
  if not np.isfinite([x,y,w,h]).all() or w<=0 or h<=0:continue
  out.append([x,y,x+w,y+h])
 return np.asarray(out,dtype=np.float32).reshape(-1,4)

def extract(cfg,part):
 from osl.cucps import CUCPS
 c=cfg['cucps'];m=cfg['motion'];dev=torch.device(cfg['runtime']['device']);out=Path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);dest=out/f'{part}_candidates.jsonl'
 if dest.exists():raise FileExistsError(f'Refusing to overwrite {dest}; choose a new output_root')
 split='train' if part=='train' else 'val';data=AnnotationDataset(Path(cfg['pairs_root'])/part,['PRW-temporal'],split,m['size'],m['person_crop']);
 search=CUCPS(c['repo'],c['args'],c['checkpoint'],dev); print('CUCPS unused checkpoint entries:',search.unused_keys,flush=True)
 model=Model(DinoBackbone(m['repo'],m['weights'],m['name']),True,person_branch=True).to(dev); ckpt=torch.load(m['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ckpt['model']);model.eval()
 @lru_cache(maxsize=16)
 def get_candidates(path):
  im=search.image(path)
  if c['candidate_mode']=='detected':return search.detections(im,c['detection_threshold'])
  if c['candidate_mode']=='ground_truth':
   a=ann_boxes(Path(cfg['prw_root'])/'annotations'/(Path(path).name+'.mat'));return a,search.embeddings(im,a),np.ones(len(a))
  raise ValueError('candidate_mode must be detected or ground_truth')
 with dest.open('x') as f, torch.no_grad():
  for i,record in enumerate(data.records):
   sample=data[i];query=search.image(record['image_path']); x,y,w,h=record['bbox_xywh'];q=search.embeddings(query,[[x,y,x+w,y+h]])[0]; boxes,features,det_scores=get_candidates(record['target_image_path']);
   image=sample['image'].unsqueeze(0).to(dev);bbox=sample['bbox'].unsqueeze(0).to(dev);crop=sample['person_crop'].unsqueeze(0).to(dev)
   with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=dev.type=='cuda'): logits=model(image,bbox,crop)
   heat=logits.float().flatten().softmax(0).reshape(logits.shape[-2:]).cpu().numpy();position=location_scores(heat,boxes,sample['scale'].numpy(),sample['offset'].numpy(),m['size'],m['heatmap_radius_cells']); appearance=(features@q).tolist();
   x,y,w,h=record['target_bbox_xywh']; target=torch.tensor([[x,y,x+w,y+h]],dtype=torch.float32); iou=box_iou(torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4),target).flatten().numpy() if len(boxes) else np.empty(0)
   row={'sequence':record['camera_segment'],'frame':record['frame'],'target_frame':record['target_frame'],'source_frame_gap':record['source_frame_gap'],'person_id':record['person_id'],'image_path':record['image_path'],'target_image_path':record['target_image_path'],'query_bbox_xywh':record['bbox_xywh'],'target_bbox_xywh':record['target_bbox_xywh'],'candidate_boxes_xyxy':boxes.tolist(),'detection_scores':np.asarray(det_scores).tolist(),'appearance':appearance,'position':position.tolist(),'positive':(iou>=c['iou_threshold']).astype(int).tolist()}
   f.write(json.dumps(row,separators=(',',':'))+'\n')
   if (i+1)%100==0: print(part,i+1,'/',len(data),flush=True)
 print('saved',dest,'queries',len(data),flush=True)

def read_rows(path):
 with path.open() as f:return [json.loads(line) for line in f if line.strip()]

def fit(cfg):
 out=Path(cfg['output_root']); out.mkdir(parents=True,exist_ok=True);cfgfit=cfg['fit']; input_root=Path(cfgfit.get('input_root',out));train=read_rows(input_root/'train_candidates.jsonl');val=read_rows(input_root/'val_candidates.jsonl');usable=[r for r in train if any(r['positive']) and len(r['positive'])>1]
 if not usable or not val:raise ValueError('Need labelled training candidates and separate validation records')
 dest=out/'ranker.json'
 if dest.exists():raise FileExistsError(f'Refusing to overwrite {dest}')
 weight=torch.nn.Parameter(torch.tensor(0.0));optimizer=torch.optim.Adam([weight],lr=cfgfit['learning_rate']);best={'epoch':0,'weight':0.0,'train_loss':None,'validation':search_metrics(val,0)};torch.manual_seed(cfgfit['seed']);order=torch.Generator().manual_seed(cfgfit['seed'])
 for epoch in range(cfgfit['epochs']):
  losses=[]
  for ix in torch.randperm(len(usable),generator=order).tolist():
   row=usable[ix]; a=torch.tensor(row['appearance']);p=torch.tensor(row['position']); positive=torch.tensor(row['positive'],dtype=torch.bool); logits=(a+weight.clamp(min=0)*p)/cfgfit['temperature'];loss=-(torch.logsumexp(logits[positive],0)-torch.logsumexp(logits,0));optimizer.zero_grad();loss.backward();optimizer.step();weight.data.clamp_(min=0);losses.append(loss.item())
  w=max(0.,float(weight.detach()));m=search_metrics(val,w); info={'epoch':epoch+1,'weight':w,'train_loss':float(np.mean(losses)),'validation':m};print(json.dumps(info),flush=True)
  if best is None or (m['rank1'],m['mrr'])>(best['validation']['rank1'],best['validation']['mrr']):best=info
 result={'method':'supervised scalar reranker; PRW train IDs used only to derive positives','usable_train_queries':len(usable),'validation_queries':len(val),'best':best,'baseline_validation':search_metrics(val,0)}
 dest.write_text(json.dumps(result,indent=2)+'\n');print('saved',dest,flush=True)

def evaluate(cfg,part):
 out=Path(cfg['output_root']);rows=read_rows(out/f'{part}_candidates.jsonl');ranker=json.loads((out/'ranker.json').read_text());weight=ranker['best']['weight']; result={'split':part,'weight':weight,'baseline':search_metrics(rows,0),'reranked':search_metrics(rows,weight),'by_frame_gap':{}}
 for key,cond in [('25',lambda n:n==25),('50_to_100',lambda n:50<=n<=100),('over_100',lambda n:n>100)]:
  subset=[r for r in rows if cond(r['source_frame_gap'])]
  if subset:result['by_frame_gap'][key]={'baseline':search_metrics(subset,0),'reranked':search_metrics(subset,weight)}
 path=out/f'{part}_metrics.json'
 if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
 path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','fit','evaluate']);p.add_argument('--config',required=True);p.add_argument('--split',choices=['train','val','test']);args=p.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
 if args.action=='fit':fit(cfg)
 elif not args.split:p.error('--split is required for extract and evaluate')
 elif args.action=='extract':extract(cfg,args.split)
 else:evaluate(cfg,args.split)
if __name__=='__main__':main()
