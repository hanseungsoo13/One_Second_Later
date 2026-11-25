"""Train, evaluate, and visualize an OSL next-position predictor."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from osl.config import validate_osl_config
from osl.data import AnnotationDataset
from osl.evaluate import Metrics
from osl.model import DinoBackbone, Model, heatmap_loss
from osl.segmentation import cache_metadata


def device_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'eval', 'e0', 'overfit', 'visualize'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--shuffle-pose', action='store_true',
                        help='Deterministically shuffle validation poses')
    return parser.parse_args()


def check_camera_splits(data):
    fixed, moving = {'02', '04', '09'}, {'05', '10', '11', '13'}
    for split in ('train', 'val'):
        kinds = {name.split('-')[1] for name in data[split] if name.startswith('MOT17-')}
        if kinds & fixed and kinds & moving:
            raise ValueError('Fixed and moving camera sequences must be evaluated separately')


def make_dataset(data, model_config, variant, split, shuffle_pose=False):
    semantic = model_config.get('semantic') if variant.semantic else None
    metadata = None
    if semantic:
        metadata = cache_metadata(semantic['model'], semantic['revision'], data['size'],
                                  [dimension // 16 for dimension in data['size']])
    return AnnotationDataset(
        data['root'], data[split], split, data['size'], model_config.get('person_crop'),
        data.get('train_sample_fps') if split == 'train' else None,
        model_config['pose']['cache'][split] if variant.pose_fusion else None,
        shuffle_pose and split == 'val', semantic['cache'] if semantic else None, metadata,
    )


def make_model(model_config, variant, device):
    backbone = DinoBackbone(model_config['repo'], model_config['weights'], model_config['name'])
    return Model(backbone, variant.transformer, person_branch=variant.person_branch,
                 pose_fusion=variant.pose_fusion,
                 semantic_channels=150 if variant.semantic else None).to(device)


def forward_loss(model, batch, sigma):
    logits = model(batch['image'], batch['bbox'], batch.get('person_crop'),
                   batch.get('pose'), batch.get('semantic'))
    loss = heatmap_loss(logits, batch['target'], batch['image'].shape[-2:], sigma)
    return logits, loss


def evaluate(model, loader, device, dtype, amp, sigma):
    model.eval()
    metrics = Metrics()
    total, count = 0.0, 0
    with torch.no_grad():
        for raw in loader:
            batch = device_batch(raw, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                logits, loss = forward_loss(model, batch, sigma)
            metrics.update(batch, logits)
            total += loss.item() * len(batch['image'])
            count += len(batch['image'])
    return {'loss': total / count, **metrics.result()}


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data, training, model_config = cfg['data'], cfg['training'], cfg['model']
    variant = validate_osl_config(cfg)
    if args.shuffle_pose and (not variant.pose_fusion or args.action != 'eval'):
        raise ValueError('--shuffle-pose requires pose evaluation')
    check_camera_splits(data)
    random.seed(training['seed'])
    np.random.seed(training['seed'])
    torch.manual_seed(training['seed'])
    torch.cuda.manual_seed_all(training['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False

    val = make_dataset(data, model_config, variant, 'val', args.shuffle_pose)
    val.records = val.records[:data.get('max_val_samples', len(val))]
    print('validation samples:', len(val), flush=True)
    val_loader = DataLoader(val, batch_size=training['batch_size'], num_workers=training['workers'])
    if args.action == 'e0':
        metrics = Metrics()
        for batch in val_loader:
            metrics.update(batch)
        print(json.dumps(metrics.result(), indent=2))
        return

    model = make_model(model_config, variant, device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)['model'])
    if args.action in ('eval', 'visualize') and not args.checkpoint:
        raise ValueError('eval requires --checkpoint')
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=training['lr'], weight_decay=training['weight_decay'])
    amp = device.type == 'cuda'
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp and dtype == torch.float16)
    output = Path(training['output'])
    if args.action == 'train' and (output / 'metrics.jsonl').exists():
        raise FileExistsError(f'Output already contains a run: {output}; choose a new output in YAML')
    output.mkdir(parents=True, exist_ok=True)
    run = None
    if args.action == 'train' and cfg.get('wandb', {}).get('enabled'):
        sys.path.insert(0, str(Path(cfg['wandb'].get('python_deps', 'external/python-deps')).resolve()))
        import wandb
        run = wandb.init(entity=cfg['wandb']['entity'], project=cfg['wandb']['project'],
                         config=cfg, mode=cfg['wandb'].get('mode', 'online'))
    if args.action == 'eval':
        print(json.dumps(evaluate(model, val_loader, device, dtype, amp, training['sigma']), indent=2))
        return
    if args.action == 'visualize':
        from osl.visualize import save_examples
        print(json.dumps(save_examples(model, val_loader, device, output / 'visualizations'), indent=2))
        return

    train = make_dataset(data, model_config, variant, 'train')
    train.records = train.records[:data.get('max_train_samples', len(train))]
    print('train samples:', len(train), flush=True)
    loader = DataLoader(train, batch_size=training['batch_size'], shuffle=True,
                        num_workers=training['workers'],
                        generator=torch.Generator().manual_seed(training['seed']))
    if args.action == 'overfit':
        batch = device_batch(next(iter(loader)), device)
        initial = None
        for step in range(args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                _, loss = forward_loss(model, batch, training['sigma'])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if initial is None:
                initial = loss.item()
            if step % 10 == 0 or step == args.steps - 1:
                print(f'step {step}: {loss.item():.5f}', flush=True)
        if loss.item() >= initial:
            raise RuntimeError('Overfit test did not reduce loss')
        return

    best, stale_epochs = float('inf'), 0
    if training['warmup_epochs']:
        for group in optimizer.param_groups:
            group['lr'] = training['lr'] / training['warmup_epochs']
    for epoch in range(training['epochs']):
        model.train()
        total, count, train_metrics = 0.0, 0, Metrics()
        for raw in loader:
            batch = device_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                logits, loss = forward_loss(model, batch, training['sigma'])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(batch['image'])
            count += len(batch['image'])
            train_metrics.update(batch, logits.detach())
        metrics = evaluate(model, val_loader, device, dtype, amp, training['sigma'])
        metrics['train_loss'] = total / count
        metrics['epoch'] = epoch + 1
        metrics.update({f'train/{key}': value for key, value in train_metrics.result().items()})
        print(json.dumps(metrics), flush=True)
        with (output / 'metrics.jsonl').open('a') as log:
            log.write(json.dumps(metrics) + '\n')
        if run:
            run.log(metrics, step=epoch + 1)
        score = metrics['overall/moving_0/argmax_fde_px']
        if score < best:
            best, stale_epochs = score, 0
            torch.save({'model': model.state_dict(), 'config': cfg, 'epoch': epoch + 1,
                        'metrics': metrics}, output / 'best.pt')
        else:
            stale_epochs += 1
        if training.get('early_stopping_patience') and stale_epochs >= training['early_stopping_patience']:
            print(f'Early stopping after {epoch + 1} epochs', flush=True)
            break
        warmup = training['warmup_epochs']
        factor = ((epoch + 2) / warmup if epoch + 1 < warmup else
                  .5 * (1 + math.cos(math.pi * (epoch + 1 - warmup) /
                                     max(1, training['epochs'] - warmup))))
        for group in optimizer.param_groups:
            group['lr'] = training['lr'] * factor
    if run:
        run.finish()


if __name__ == '__main__':
    main()
