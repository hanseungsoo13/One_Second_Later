"""Evaluate fixed motion predictors on identity-matched PRW test pairs."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from cucps import experiment
from osl.data import AnnotationDataset
from osl.evaluate import predictions
from integration.variants import resolve_cucps_variant

HERE = _ROOT


def gap_group(gap, bounds):
    previous = 0
    for upper in bounds:
        if gap <= upper:
            return f'{previous + 1}-{upper}'
        previous = upper
    return f'>{previous}'


def summarize(rows, bounds):
    if not rows:
        raise ValueError('No identity-matched pairs were evaluated')
    groups = defaultdict(list)
    for row in rows:
        groups['overall'].append(row)
        groups['gap/' + gap_group(row['frame_gap'], bounds)].append(row)
        if row['displacement_px'] >= row['height_px'] * 0.1:
            groups['moving_0.1'].append(row)
    result = {}
    for group, items in groups.items():
        result[group] = {'count': len(items)}
        for model in items[0]['errors']:
            errors = np.asarray([item['errors'][model] for item in items])
            heights = np.asarray([item['height_px'] for item in items])
            result[group][model] = {
                'fde_px': float(errors.mean()),
                'fde_height': float((errors / heights).mean()),
            }
    return result


def resolve_evaluation_models(names):
    """Resolve legacy and descriptive names while rejecting duplicate models."""
    if not names:
        raise ValueError('models must contain at least one model')
    models = {}
    for name in names:
        if name in ('baseline', 'graph'):
            raise ValueError(f'{name} is not an OSL model variant')
        variant = resolve_cucps_variant(name)
        if variant.name in {item.name for item in models.values()}:
            raise ValueError(f'duplicate model variant: {name}')
        models[name] = variant
    return models


def run(cfg):
    output = HERE / cfg['output']
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    source_path = HERE / cfg['experiment_config']
    source = yaml.safe_load(source_path.read_text())
    paths = experiment._paths(source)
    device = experiment._resolve_device(cfg['device'])
    bounds = cfg['gap_upper_bounds']
    if not bounds or any(x <= 0 for x in bounds) or sorted(set(bounds)) != bounds:
        raise ValueError('gap_upper_bounds must be increasing positive integers')
    models = resolve_evaluation_models(cfg['models'])
    for name in models:
        experiment.validate_cucps_config(source, name)
    crop_config = source['motion'].get('person_crop') if any(v.person_branch for v in models.values()) else None
    dataset = AnnotationDataset(HERE / cfg['pairs_root'], [cfg['sequence']], 'val',
                                size=tuple(source['motion']['size']), person_crop=crop_config)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    predictors = {name: experiment._motion_model(source, device, paths, name) for name in models}
    pose_estimator = experiment._make_pose_estimator(source, device) if any(v.pose_fusion for v in models.values()) else None
    semantic_predictor = None
    if any(v.semantic for v in models.values()):
        semantic_cfg = source['motion']['semantic']
        size = source['motion']['size']
        if any(n % 16 for n in size):
            raise ValueError('motion.size must be divisible by 16 for segmentation')
        semantic_predictor = experiment.SegFormerPredictor(
            semantic_cfg['model'], semantic_cfg['revision'], size, [n // 16 for n in size],
            semantic_cfg.get('device', str(device)))
    rows = []
    with torch.inference_mode():
        for batch in loader:
            start = len(rows)
            records = dataset.records[start:start + len(batch['image'])]
            pose = None
            if pose_estimator:
                poses = []
                for record in records:
                    x, y, w, h = record['bbox_xywh']
                    with Image.open(record['image_path']) as image:
                        poses.append(experiment._predict_relative_poses(
                            pose_estimator, image, [[x, y, x + w, y + h]])[0])
                pose = torch.from_numpy(np.stack(poses)).to(device)
            semantic = None
            if semantic_predictor:
                maps = []
                for record in records:
                    with Image.open(record['image_path']) as image:
                        maps.append(semantic_predictor(image).astype(np.float16).astype(np.float32))
                semantic = torch.from_numpy(np.stack(maps)).to(device)
            predictions_by_model = {'stationary': batch['original_foot']}
            for name, model in predictors.items():
                variant = models[name]
                crop = batch['person_crop'].to(device) if variant.person_branch else None
                logits = model(batch['image'].to(device), batch['bbox'].to(device), crop,
                               pose if variant.pose_fusion else None, semantic if variant.semantic else None)
                argmax, _, _ = predictions(logits, batch['image'].shape[-2:])
                predictions_by_model[name] = (argmax.cpu() - batch['offset']) / batch['scale']
            for i, record in enumerate(records):
                target = batch['original_target'][i]
                foot = batch['original_foot'][i]
                gap = record['target_frame'] - record['frame']
                if gap <= 0:
                    raise ValueError('Expected a strictly later target frame')
                rows.append({
                    'frame_gap': gap,
                    'height_px': max(float(batch['height'][i]), 1.),
                    'displacement_px': float((target - foot).norm()),
                    'errors': {name: float((point[i] - target).norm())
                               for name, point in predictions_by_model.items()},
                })
            if len(rows) % 200 < len(records):
                print(f'Evaluated {len(rows)}/{len(dataset)} pairs', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'protocol': 'PRW identity-matched consecutive available frames, test split',
              'pairs_root': str((HERE / cfg['pairs_root']).resolve()),
              'experiment_config': str(source_path.resolve()),
              'checkpoints': {name: str(paths[('motion', 'pose_checkpoint' if models[name].pose_fusion else 'checkpoint')]) for name in models},
              'gap_upper_bounds': bounds, 'metrics': summarize(rows, bounds)}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['metrics'], indent=2))
    print(f'Saved {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()))


if __name__ == '__main__':
    main()
