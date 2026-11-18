"""Metrics in original image coordinates; heatmap cells represent pixel centers."""
from collections import defaultdict
import math
import torch

def predictions(logits, size):
    b,_,h,w=logits.shape; ih,iw=size; p=logits.float().flatten(1).softmax(1)
    idx=p.argmax(1); ax=(idx%w+.5)*iw/w; ay=(idx//w+.5)*ih/h
    x=(torch.arange(w,device=p.device)+.5)*iw/w; y=(torch.arange(h,device=p.device)+.5)*ih/h
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    ex=(p*xx.flatten()).sum(1); ey=(p*yy.flatten()).sum(1)
    return torch.stack((ax,ay),1),torch.stack((ex,ey),1),p

class Metrics:
    def __init__(self): self.values=defaultdict(list)
    def update(self,batch,logits=None):
        target=batch['original_target']; foot=batch['original_foot']; height=batch['height'].clamp_min(1)
        movement=(target-foot).norm(dim=1)/height
        baseline=(target-foot).norm(dim=1)
        if logits is None: arg=exp=foot; nll=None
        else:
            arg,exp,p=predictions(logits,batch['image'].shape[-2:]); offset=batch['offset']; scale=batch['scale']
            arg=(arg-offset)/scale; exp=(exp-offset)/scale
            h,w=logits.shape[-2:]; ih,iw=batch['image'].shape[-2:]; xy=batch['target']
            ix=(xy[:,0]*w/iw).floor().long().clamp(0,w-1); iy=(xy[:,1]*h/ih).floor().long().clamp(0,h-1)
            nll=-p[torch.arange(len(p),device=p.device),iy*w+ix].clamp_min(1e-30).log()
        for i,seq in enumerate(batch['sequence']):
            for group in ('overall',seq):
                for threshold in (0,.1,.25,.5):
                    if movement[i] < threshold: continue
                    key=f'{group}/moving_{threshold:g}'
                    values={'zero_fde_px':baseline[i], 'argmax_fde_px':(arg[i]-target[i]).norm(), 'expected_fde_px':(exp[i]-target[i]).norm(), 'argmax_fde_height':(arg[i]-target[i]).norm()/height[i], 'expected_fde_height':(exp[i]-target[i]).norm()/height[i]}
                    if nll is not None: values['nll']=nll[i]
                    for k,v in values.items(): self.values[f'{key}/{k}'].append(float(v))
    def result(self):
        out={k:sum(v)/len(v) for k,v in self.values.items()}
        for k,v in self.values.items():
            if k.endswith('/zero_fde_px'): out[k.rsplit('/',1)[0]+'/count']=len(v)
        for k,v in list(out.items()):
            if k.endswith('/argmax_fde_px') or k.endswith('/expected_fde_px'):
                base=out[k.rsplit('/',1)[0]+'/zero_fde_px']; out[k.replace('_fde_px','_improvement_px')]=base-v
        return out
