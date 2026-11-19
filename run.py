import argparse, json, math, random, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from osl.data import AnnotationDataset
from osl.model import DinoBackbone, Model, heatmap_loss
from osl.evaluate import Metrics

def device_batch(b,device): return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['train','eval','e0','overfit','visualize']); p.add_argument('--config',required=True); p.add_argument('--checkpoint'); p.add_argument('--steps',type=int,default=100); args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); d=cfg['data']; t=cfg['training']; m=cfg['model']
    if m['variant'] not in ('e2a','e2b','e2c'): raise ValueError(f"Unknown model variant: {m['variant']}")
    if m['variant']=='e2c' and not m.get('person_crop'): raise ValueError('E2-C requires model.person_crop configuration')
    fixed={'02','04','09'}; moving={'05','10','11','13'}
    for split in ('train','val'):
        kinds={name.split('-')[1] for name in d[split] if name.startswith('MOT17-')}
        if kinds & fixed and kinds & moving: raise ValueError('Fixed and moving camera sequences must be evaluated separately')
    random.seed(t['seed']); np.random.seed(t['seed']); torch.manual_seed(t['seed']); torch.cuda.manual_seed_all(t['seed'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.backends.cudnn.benchmark=False
    def dataset(names,split): return AnnotationDataset(d['root'],names,split,d['size'],m.get('person_crop'),d.get('train_sample_fps') if split=='train' else None)
    val=dataset(d['val'],'val'); val.records=val.records[:d.get('max_val_samples',len(val))]; print('validation samples:',len(val),flush=True)
    val_loader=DataLoader(val,batch_size=t['batch_size'],num_workers=t['workers'])
    if args.action=='e0':
        metrics=Metrics()
        for batch in val_loader: metrics.update(batch)
        print(json.dumps(metrics.result(),indent=2)); return
    model=Model(DinoBackbone(m['repo'],m['weights'],m['name']),m['variant'] in ('e2b','e2c'),person_branch=m['variant']=='e2c').to(device)
    if args.checkpoint: model.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=False)['model'])
    if args.action in ('eval','visualize') and not args.checkpoint: raise ValueError('eval requires --checkpoint')
    optimizer=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=t['lr'],weight_decay=t['weight_decay'])
    amp=device.type=='cuda'; dtype=torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=amp and dtype==torch.float16)
    output=Path(t['output'])
    if args.action=='train' and (output/'metrics.jsonl').exists(): raise FileExistsError(f'Output already contains a run: {output}; choose a new output in YAML')
    output.mkdir(parents=True,exist_ok=True)
    run=None
    if args.action == 'train' and cfg.get('wandb',{}).get('enabled'):
        sys.path.insert(0, str(Path('external/python-deps').resolve()))
        import wandb
        run=wandb.init(entity=cfg['wandb']['entity'],project=cfg['wandb']['project'],config=cfg,mode=cfg['wandb'].get('mode','online'))
    def evaluate():
        model.eval(); metrics=Metrics(); total=0.; n=0
        with torch.no_grad():
            for raw in val_loader:
                b=device_batch(raw,device)
                with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp): logits=model(b['image'],b['bbox'],b.get('person_crop')); loss=heatmap_loss(logits,b['target'],b['image'].shape[-2:],t['sigma'])
                metrics.update(b,logits); total+=loss.item()*len(b['image']); n+=len(b['image'])
        return {'loss':total/n,**metrics.result()}
    if args.action=='eval': print(json.dumps(evaluate(),indent=2)); return
    if args.action=='visualize':
        from osl.visualize import save_examples
        print(json.dumps(save_examples(model,val_loader,device,output/'visualizations'),indent=2)); return
    train=dataset(d['train'],'train'); train.records=train.records[:d.get('max_train_samples',len(train))]; print('train samples:',len(train),flush=True)
    loader=DataLoader(train,batch_size=t['batch_size'],shuffle=True,num_workers=t['workers'],generator=torch.Generator().manual_seed(t['seed']))
    if args.action=='overfit':
        batch=device_batch(next(iter(loader)),device); initial=None
        for step in range(args.steps):
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp): loss=heatmap_loss(model(batch['image'],batch['bbox'],batch.get('person_crop')),batch['target'],batch['image'].shape[-2:],t['sigma'])
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            if initial is None: initial=loss.item()
            if step%10==0 or step==args.steps-1: print(f'step {step}: {loss.item():.5f}',flush=True)
        if loss.item() >= initial: raise RuntimeError('Overfit test did not reduce loss')
        return
    best=float('inf')
    if t['warmup_epochs']:
        for group in optimizer.param_groups: group['lr']=t['lr']/t['warmup_epochs']
    for epoch in range(t['epochs']):
        model.train(); total=0.; n=0; train_metrics=Metrics()
        for raw in loader:
            b=device_batch(raw,device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp): logits=model(b['image'],b['bbox'],b.get('person_crop')); loss=heatmap_loss(logits,b['target'],b['image'].shape[-2:],t['sigma'])
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total+=loss.item()*len(b['image']); n+=len(b['image']); train_metrics.update(b,logits.detach())
        metrics=evaluate(); metrics['train_loss']=total/n; metrics['epoch']=epoch+1; metrics.update({'train/'+k:v for k,v in train_metrics.result().items()})
        print(json.dumps(metrics),flush=True)
        with (output/'metrics.jsonl').open('a') as log: log.write(json.dumps(metrics)+'\n')
        if run: run.log(metrics,step=epoch+1)
        score=metrics['overall/moving_0/argmax_fde_px']
        if score<best:
            best=score; torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch+1,'metrics':metrics},output/'best.pt')
        warm=t['warmup_epochs']; factor=(epoch+2)/warm if epoch+1<warm else .5*(1+math.cos(math.pi*(epoch+1-warm)/max(1,t['epochs']-warm)))
        for group in optimizer.param_groups: group['lr']=t['lr']*factor
    if run: run.finish()
if __name__=='__main__': main()
