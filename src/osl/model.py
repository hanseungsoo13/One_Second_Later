import math
import torch
from torch import nn
import torch.nn.functional as F

class DinoBackbone(nn.Module):
    def __init__(self, repo, weights, name='dinov3_vitb16'):
        super().__init__(); self.net=torch.hub.load(repo,name,source='local',weights=weights); self.net.requires_grad_(False); self.net.eval(); self.channels=self.net.embed_dim
    def train(self, mode=True):
        super().train(mode); self.net.eval(); return self
    def forward(self,x):
        with torch.no_grad(): f=self.net.forward_features(x)['x_norm_patchtokens']
        h,w=x.shape[-2]//16,x.shape[-1]//16
        return f.transpose(1,2).reshape(x.shape[0],self.channels,h,w)

def spatial_prompt(box,h,w,image_size,clip=5.,sigma=1.5):
    ih,iw=image_size; device=box.device
    y=(torch.arange(h,device=device)+.5)*ih/h; x=(torch.arange(w,device=device)+.5)*iw/w
    yy,xx=torch.meshgrid(y,x,indexing='ij'); xx=xx[None]; yy=yy[None]
    x1,y1,x2,y2=[v[:,None,None] for v in box.unbind(1)]
    fx=(x1+x2)/2; fy=y2; bh=(y2-y1).clamp_min(1)
    mask=((xx>=x1)&(xx<=x2)&(yy>=y1)&(yy<=y2)).float()
    gaussian=torch.exp(-.5*(((xx-fx)/(iw/w*sigma))**2+((yy-fy)/(ih/h*sigma))**2))
    return torch.stack((mask,gaussian,((xx-fx)/bh).clamp(-clip,clip).expand(-1,h,w),((yy-fy)/bh).clamp(-clip,clip).expand(-1,h,w)),1)

def position(h,w,c,device):
    assert c%4==0
    freq=torch.exp(torch.arange(c//4,device=device)*(-math.log(10000)/(c//4)))
    y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    a=x.flatten()[:,None]*freq; b=y.flatten()[:,None]*freq
    return torch.cat([a.sin(),a.cos(),b.sin(),b.cos()],1)[None]

class Model(nn.Module):
    def __init__(self,backbone,transformer=True,dim=256):
        super().__init__(); self.backbone=backbone
        self.projection=nn.Sequential(nn.Conv2d(backbone.channels+4,dim,1),nn.GroupNorm(32,dim),nn.GELU())
        self.transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,8,1024,.1,batch_first=True,norm_first=True),2,enable_nested_tensor=False) if transformer else None
        self.decoder=nn.Sequential(nn.Conv2d(dim,128,3,padding=1),nn.GroupNorm(32,128),nn.GELU(),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(128,64,3,padding=1),nn.GELU(),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(64,32,3,padding=1),nn.GELU(),nn.Conv2d(32,1,1))
    def forward(self,image,bbox):
        f=self.backbone(image); b,c,h,w=f.shape
        f=self.projection(torch.cat((f,spatial_prompt(bbox,h,w,image.shape[-2:])),1))
        if self.transformer is not None:
            f=self.transformer(f.flatten(2).transpose(1,2)+position(h,w,f.shape[1],f.device)).transpose(1,2).reshape(b,-1,h,w)
        return self.decoder(f)

def heatmap_loss(logits,target,image_size,sigma):
    b,_,h,w=logits.shape; ih,iw=image_size
    x=(torch.arange(w,device=logits.device)+.5)*iw/w; y=(torch.arange(h,device=logits.device)+.5)*ih/h
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    q=torch.exp(-.5*(((xx-target[:,0,None,None])/(iw/w*sigma))**2+((yy-target[:,1,None,None])/(ih/h*sigma))**2))
    q=q/q.sum((1,2),keepdim=True).clamp_min(1e-12)
    return -(q.flatten(1)*F.log_softmax(logits.float().flatten(1),dim=1)).sum(1).mean()
