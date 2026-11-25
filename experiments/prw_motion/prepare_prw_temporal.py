"""Create PRW temporal query pairs with train/validation/test isolation."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
import argparse, json, re
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from scipy.io import loadmat
import yaml

def read_annotations(item):
 root,name=item; data=loadmat(root/'annotations'/f'{name}.jpg.mat');
 for k in ('box_new','anno_file','anno_previous'):
  if k in data: arr=data[k];break
 else: raise ValueError(f'No annotation boxes: {name}')
 boxes={};duplicates=set()
 for ident,x,y,w,h in arr:
  if ident<=0 or int(ident)!=ident or not np.isfinite([ident,x,y,w,h]).all() or w<=0 or h<=0:continue
  ident=int(ident)
  if ident in boxes:duplicates.add(ident)
  else:boxes[ident]=[float(x),float(y),float(w),float(h)]
 for ident in duplicates:boxes.pop(ident,None)
 return name,boxes

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();cfg=yaml.safe_load(Path(args.config).read_text());root=Path(cfg['prw_root']); out=Path(cfg['pairs_root']);fraction=cfg['prepare']['train_fraction']
 if not 0<fraction<1:raise ValueError('train_fraction must be in (0,1)')
 names={s:[str(x[0][0]) for x in loadmat(root/f'frame_{s}.mat')[f'img_index_{s}']] for s in ('train','test')}; train=set(names['train']);test=set(names['test']);assert not train&test
 groups=defaultdict(list)
 for name in train|test:
  m=re.fullmatch(r'(c\d+s\d+)_(\d+)',name)
  if not m:raise ValueError(name)
  groups[m[1]].append((int(m[2]),name))
 parts={}
 for group in groups.values():
  group.sort();train_in_group=[name for _,name in group if name in train];cut=int(len(train_in_group)*fraction)
  for name in train_in_group[:cut]:parts[name]='train'
  for name in train_in_group[cut:]:parts[name]='val'
 for name in test:parts[name]='test'
 pairs=defaultdict(list)
 for group in groups.values():
  for (_,a),(_,b) in zip(group,group[1:]):
   if parts[a]==parts[b]:pairs[parts[a]].append((a,b))
 needed=sorted({n for ps in pairs.values() for pair in ps for n in pair});print('reading annotations',len(needed),flush=True)
 with ThreadPoolExecutor(max_workers=cfg['prepare']['annotation_workers']) as pool: boxes=dict(pool.map(read_annotations,((root,n) for n in needed)))
 summary={}
 for part in ('train','val','test'):
  mode='train' if part=='train' else 'val';folder=out/part/'PRW-temporal';folder.mkdir(parents=True,exist_ok=True);path=folder/f'{mode}.jsonl'
  if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
  count=0;img_sizes={}
  with path.open('x') as f:
   for a,b in sorted(pairs[part]):
    ids=sorted(boxes[a].keys() & boxes[b].keys())
    if not ids:continue
    src=root/'frames'/f'{a}.jpg';dst=root/'frames'/f'{b}.jpg'
    if not src.is_file() or not dst.is_file():raise FileNotFoundError(f'{src} or {dst}')
    if a not in img_sizes:
     with Image.open(src) as image: img_sizes[a]=list(image.size)
    for ident in ids:
     x,y,w,h=boxes[b][ident]; sequence=a.rsplit('_',1)[0];frame=int(a.rsplit('_',1)[1]);target=int(b.rsplit('_',1)[1])
     rec={'sequence':'PRW-temporal','camera_segment':sequence,'person_id':ident,'track_id':f'{sequence}:{ident}','frame':frame,'image_path':str(src),'image_size':img_sizes[a],'bbox_xywh':boxes[a][ident],'target_frame':target,'target_image_path':str(dst),'target_bbox_xywh':boxes[b][ident],'target_bottom_center_xy':[x+w/2,y+h],'source_frame_gap':target-frame}
     f.write(json.dumps(rec,separators=(',',':'))+'\n');count+=1
  summary[part]={'frame_pairs':len(pairs[part]),'queries':count,'file':str(path)};print(part,summary[part],flush=True)
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
