"""Inference adapter for the unmodified official CUCPS repository."""
import json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

class CUCPS:
    def __init__(self, repo, args_json, checkpoint, device):
        repo=Path(repo).resolve()
        if not (repo/'lib/model/faster_rcnn.py').is_file(): raise FileNotFoundError(f'CUCPS repository missing: {repo}')
        sys.path.insert(0,str(repo))
        from lib.model.faster_rcnn import get_model
        args=SimpleNamespace()
        for key,value in json.loads(Path(args_json).read_text()).items():
            if '.' in key:
                group,key=key.split('.',1)
                if not hasattr(args,group): setattr(args,group,SimpleNamespace())
                setattr(getattr(args,group),key,value)
            else: setattr(args,key,value)
        self.device=torch.device(device);self.model=get_model(args,training=False,pretrained_backbone=False)
        checkpoint_data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        result=self.model.load_state_dict(checkpoint_data['model'],strict=False)
        if result.missing_keys: raise RuntimeError(f'CUCPS checkpoint missing model weights: {result.missing_keys}')
        self.unused_keys=result.unexpected_keys; self.model.to(self.device).eval()
    def image(self,path):
        with Image.open(path) as image: return to_tensor(image.convert('RGB')).to(self.device)
    @torch.no_grad()
    def embeddings(self,image,boxes):
        boxes=torch.as_tensor(boxes,dtype=torch.float32,device=self.device).reshape(-1,4)
        if not len(boxes): return np.empty((0,256),dtype=np.float32)
        outputs=self.model.ex_feat([image],[{'boxes':boxes}],mode='det')
        return torch.cat(outputs,dim=0).detach().cpu().numpy()
    @torch.no_grad()
    def detections(self,image,threshold):
        output=self.model(0,[image])[0]; mask=output['scores']>=threshold
        return (output['boxes'][mask].cpu().numpy(),output['embeddings'][mask].cpu().numpy(),output['scores'][mask].cpu().numpy())
