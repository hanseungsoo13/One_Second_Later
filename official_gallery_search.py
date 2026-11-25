"""CUCPS PRW official full-gallery baseline and temporal reranking experiment."""
import argparse,json,pickle,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch,yaml
from PIL import Image
from scipy.io import loadmat

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'src'))
from osl.data import AnnotationDataset
from osl.model import DinoBackbone,Model
from osl.cucps import CUCPS
from osl.temporal_search import location_scores
from osl.official_gallery import next_frame_map,prepare_gallery,evaluate_queries,fixed_fusion_weight

class Records:
 def __init__(self,records,data_path):self.record=records;self.data_path=str(data_path)
 def __len__(self):return len(self.record)

def load_prw(cfg):
 sys.path.insert(0,str(HERE/'external/python-deps'));sys.path.insert(0,str(Path(cfg['cucps']['repo']).resolve()))
 from lib.datasets import prw
 prw.xrange=range
 root=Path(cfg['prw_root']); gallery=prw.PRW(str(root),None,mode='test');queries=[]
 for line in (root/'query_info.txt').read_text().splitlines():
  if not line.strip():continue
  p=line.split();pid=int(p[0]);x,y,w,h=map(float,p[1:5]);name=p[5]+'.jpg';xyxy=np.clip(np.array([x,y,x+w,y+h]).astype(np.int32),0,None)
  queries.append({'im_name':name,'boxes':xyxy[None,:],'gt_pids':np.array([pid]),'cam_id':int(name.split('s',1)[0][1:])})
 names=[]
 for part in ('train','test'):
  names.extend(str(row[0][0])+'.jpg' for row in loadmat(root/f'frame_{part}.mat')[f'img_index_{part}'])
 return prw,gallery,Records(queries,root/'frames'),names

def make_motion_sample(query,path,size,crop):
 x1,y1,x2,y2=map(float,query['boxes'][0]);
 with Image.open(path) as image:w,h=image.size
 dataset=object.__new__(AnnotationDataset);dataset.size=size;dataset.person_crop=crop;dataset.poses=None;dataset.pose_keys=None
 dataset.records=[{'sequence':'query','person_id':int(query['gt_pids'][0]),'frame':0,'image_path':str(path),'image_size':[w,h],'bbox_xywh':[x1,y1,x2-x1,y2-y1],'target_bottom_center_xy':[(x1+x2)/2,y2]}]
 return dataset[0]

def extract(cfg):
 out=Path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);dest=out/'official_features.pkl'
 if dest.exists():raise FileExistsError(f'Refusing to overwrite {dest}')
 prw,gallery,probes,all_names=load_prw(cfg);c=cfg['cucps'];m=cfg['motion'];device=torch.device(cfg['runtime']['device']);cucps=CUCPS(c['repo'],c['args'],c['checkpoint'],device)
 print('CUCPS unused checkpoint entries:',cucps.unused_keys,flush=True);motion=Model(DinoBackbone(m['repo'],m['weights'],m['name']),True,person_branch=True).to(device);checkpoint=torch.load(m['checkpoint'],map_location='cpu',weights_only=False);motion.load_state_dict(checkpoint['model']);motion.eval()
 detections=[];features=[]
 with torch.no_grad():
  for i,item in enumerate(gallery.record):
   image=cucps.image(Path(gallery.data_path)/item['im_name']);boxes,feat,scores=cucps.detections(image,0.0);detections.append(np.column_stack((boxes,scores)));features.append(feat)
   if (i+1)%100==0:print('gallery',i+1,'/',len(gallery),flush=True)
  indices={r['im_name']:i for i,r in enumerate(gallery.record)};nexts=next_frame_map(all_names,indices);probe_features=[];biases={}
  for i,query in enumerate(probes.record):
   path=Path(gallery.data_path)/query['im_name'];image=cucps.image(path);probe_features.append(cucps.embeddings(image,query['boxes']))
   nxt=nexts.get(query['im_name'])
   if nxt is not None:
    sample=make_motion_sample(query,path,m['size'],m['person_crop']);im=sample['image'].unsqueeze(0).to(device);box=sample['bbox'].unsqueeze(0).to(device);crop=sample['person_crop'].unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):logits=motion(im,box,crop)
    heat=logits.float().flatten().softmax(0).reshape(logits.shape[-2:]).cpu().numpy();dets=detections[indices[nxt]];loc=location_scores(heat,dets[:,:4],sample['scale'].numpy(),sample['offset'].numpy(),m['size'],m['heatmap_radius_cells']);biases[i]=(nxt,loc)
   if (i+1)%100==0:print('queries',i+1,'/',len(probes),flush=True)
 result={'gallery_records':gallery.record,'query_records':probes.record,'detections':detections,'features':features,'probe_features':probe_features,'biases':biases,'data_path':gallery.data_path,'cucps_checkpoint':str(Path(c['checkpoint']).resolve()),'motion_checkpoint':str(Path(m['checkpoint']).resolve()),'parameters':{'motion_size':m['size'],'person_crop':m['person_crop'],'radius':m['heatmap_radius_cells']}}
 with dest.open('xb') as f:pickle.dump(result,f,protocol=pickle.HIGHEST_PROTOCOL)
 print('saved',dest,'gallery',len(gallery),'queries',len(probes),'queries_with_next',len(biases),flush=True)

def compare(cfg):
 out=Path(cfg['output_root']);dest=out/'official_gallery_metrics.json'
 if dest.exists():raise FileExistsError(f'Refusing to overwrite {dest}')
 # Only load files generated locally: pickle is not safe for untrusted inputs.
 with (out/'official_features.pkl').open('rb') as f:data=pickle.load(f)
 weight=fixed_fusion_weight(cfg);c=cfg['cucps']
 if Path(data['cucps_checkpoint'])!=Path(c['checkpoint']).resolve() or Path(data['motion_checkpoint'])!=Path(cfg['motion']['checkpoint']).resolve():raise ValueError('Cached checkpoint paths differ from configuration')
 expected={'motion_size':cfg['motion']['size'],'person_crop':cfg['motion']['person_crop'],'radius':cfg['motion']['heatmap_radius_cells']}
 if data['parameters']!=expected:raise ValueError('Cached motion preprocessing differs from configuration')
 prw,_,_,_=load_prw(cfg);records=data['gallery_records'];queries=data['query_records'];det=data['detections'];feat=data['features'];qfeat=data['probe_features'];gallery=Records(records,data['data_path']);probes=Records(queries,data['data_path'])
 print('Evaluating with original CUCPS function...',flush=True)
 original=prw.PRW.search_performance_calc(gallery,probes,det,feat,qfeat,det_thresh=c['detection_threshold'],gallery_size=-1,ignore_cam_id=c['ignore_cam_id'],remove_unlabel=c['remove_unlabel'])
 print('Preparing same candidates for reranking...',flush=True);cache=prepare_gallery(records,det,feat,c['detection_threshold'],c['remove_unlabel']);baseline=evaluate_queries(records,cache,queries,qfeat,{},0,c['ignore_cam_id']);
 if not np.isclose(original['mAP'],baseline['mAP'],rtol=0,atol=1e-10) or not np.allclose(original['accs'],baseline['top_k'],rtol=0,atol=1e-10):raise RuntimeError(f'Baseline does not match original CUCPS: original={original["mAP"],original["accs"]}, ours={baseline["mAP"],baseline["top_k"]}')
 print('Original CUCPS and baseline agree; evaluating temporal reranking...',flush=True);rerank=evaluate_queries(records,cache,queries,qfeat,data['biases'],weight,c['ignore_cam_id'])
 changed=[i for i,(a,b) in enumerate(zip(baseline['details'],rerank['details'])) if (a['ap'],a['top1'])!=(b['ap'],b['top1'])]
 groups={'no_next':[],'gap_25':[],'gap_26_to_100':[],'gap_over_100':[],'gap_other':[]}
 for i,q in enumerate(queries):
  if i not in data['biases']: group='no_next'
  else:
   gap=int(data['biases'][i][0].rsplit('_',1)[1][:-4])-int(q['im_name'].rsplit('_',1)[1][:-4]);group='gap_25' if gap==25 else 'gap_26_to_100' if 25<gap<=100 else 'gap_over_100' if gap>100 else 'gap_other'
  groups[group].append(i)
 def group_metrics(details,indices):
  if not indices:return None
  return {'queries':len(indices),'mAP':float(np.mean([details[i]['ap'] for i in indices])),'rank1':float(np.mean([details[i]['top1'] for i in indices]))}
 gap_results={k:{'baseline':group_metrics(baseline['details'],v),'reranked':group_metrics(rerank['details'],v)} for k,v in groups.items()}
 report={'protocol':'CUCPS official PRW regular full gallery; next-frame temporal score applied only where available','queries':len(queries),'gallery_images':len(records),'queries_with_next':len(data['biases']),'lambda':weight,'original':{'mAP':float(original['mAP']),'top_k':original['accs'].tolist()},'baseline':baseline,'reranked':rerank,'queries_with_changed_ap_or_top1':len(changed),'by_frame_gap':gap_results,'caveats':['CUCPS regular protocol uses GT labels to remove unlabeled detections','query bounding box is ground truth','frame gaps vary and elapsed time is unknown','fusion weight is fixed in advance without PRW future-location or ID supervision']}
 dest.write_text(json.dumps(report,indent=2)+'\n');print('saved',dest,'baseline',report['original'],'reranked mAP',rerank['mAP'],'top1',rerank['top_k'][0],flush=True)

def run(cfg):
 """Evaluate in one invocation, reusing a completed trusted extraction cache."""
 fixed_fusion_weight(cfg)
 out=Path(cfg['output_root'])
 if (out/'official_gallery_metrics.json').exists():raise FileExistsError(f'Refusing to overwrite {out / "official_gallery_metrics.json"}')
 if (out/'official_features.pkl').exists():print('Reusing completed feature cache',out/'official_features.pkl',flush=True)
 else:extract(cfg)
 compare(cfg)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','extract','compare']);parser.add_argument('--config',required=True);args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
 if args.action=='run':run(cfg)
 elif args.action=='extract':extract(cfg)
 else:compare(cfg)
if __name__=='__main__':main()
