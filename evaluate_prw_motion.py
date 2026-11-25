"""Evaluate fixed motion predictors on identity-matched PRW test pairs."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

import cucps_temporal_experiment as experiment
from osl.data import AnnotationDataset
from osl.evaluate import predictions

HERE = Path(__file__).resolve().parent


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
    models = cfg['models']
    if not models or len(set(models)) != len(models) or any(m not in ('e2c', 'pose_p1', 'pose_p2') for m in models):
        raise ValueError('models must contain unique choices from e2c, pose_p1, pose_p2')
    dataset = AnnotationDataset(HERE / cfg['pairs_root'], [cfg['sequence']], 'val',
                                size=tuple(source['motion']['size']),
                                person_crop=source['motion']['person_crop'])
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    predictors = {name: experiment._motion_model(source, device, paths, 'temporal' if name == 'e2c' else name)
                  for name in models}
    pose_estimator = experiment._make_pose_estimator(source, device) if any(m.startswith('pose_') for m in models) else None
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
            predictions_by_model = {'stationary': batch['original_foot']}
            for name, model in predictors.items():
                logits = model(batch['image'].to(device), batch['bbox'].to(device),
                               batch['person_crop'].to(device), pose if name.startswith('pose_') else None)
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
              'checkpoints': {name: str(paths[('motion', 'checkpoint' if name == 'e2c' else 'pose_checkpoint')]) for name in models},
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
