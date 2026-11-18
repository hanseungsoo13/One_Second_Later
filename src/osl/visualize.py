"""Save examples; intermediate GT is used only to classify trajectories."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from .evaluate import predictions

def save_examples(model, loader, device, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); chosen={}; worst=-1
    model.eval()
    with torch.no_grad():
        for raw in loader:
            b={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in raw.items()}
            logits=model(b['image'],b['bbox'],b.get('person_crop')); arg,_,p=predictions(logits,b['image'].shape[-2:]); arg=(arg-b['offset'])/b['scale']
            for i in range(len(arg)):
                current=b['original_foot'][i]; target=b['original_target'][i]; mid=b['mid'][i]; height=b['height'][i].clamp_min(1)
                motion=float((target-current).norm()/height); error=float((arg[i]-target).norm()/height)
                v1=mid-current; v2=target-mid
                cosine=float(torch.dot(v1,v2)/(v1.norm()*v2.norm()).clamp_min(1e-6))
                category='stationary' if motion<.1 else ('turning' if v1.norm()/height>.05 and v2.norm()/height>.05 and cosine<.5 else 'straight' if cosine>.9 else None)
                categories=[]
                if category and category not in chosen: categories.append(category); chosen[category]=True
                if error>worst: categories.append('worst'); worst=error
                if not categories: continue
                image=Image.open(raw['path'][i]).convert('RGB'); w,h=image.size
                ih,iw=b['image'].shape[-2:]; heat=p[i].reshape(1,1,*logits.shape[-2:]); heat=F.interpolate(heat,(ih,iw),mode='bilinear',align_corners=False)[0,0]
                sx,sy=map(float,b['scale'][i]); ox,oy=map(int,b['offset'][i]); nw,nh=round(w*sx),round(h*sy)
                heat=heat[oy:oy+nh,ox:ox+nw].cpu().numpy(); heat=Image.fromarray(heat).resize((w,h)); heat=np.asarray(heat); heat=heat/(heat.max()+1e-12)
                a=np.asarray(image).astype(np.float32); color=np.zeros_like(a); color[:,:,0]=255; alpha=(heat*.6)[:,:,None]; out=Image.fromarray(np.uint8(a*(1-alpha)+color*alpha))
                draw=ImageDraw.Draw(out); box=b['bbox'][i].cpu().numpy(); box[[0,2]]=(box[[0,2]]-ox)/sx; box[[1,3]]=(box[[1,3]]-oy)/sy; draw.rectangle(tuple(box),outline='lime',width=3)
                for point,color in [(current,'cyan'),(target,'yellow'),(arg[i],'red')]:
                    x,y=map(float,point); draw.ellipse((x-7,y-7,x+7,y+7),fill=color)
                for name in categories: out.save(output/f'{name}.jpg')
    return {'categories':list(chosen),'worst_normalized_error':worst}
