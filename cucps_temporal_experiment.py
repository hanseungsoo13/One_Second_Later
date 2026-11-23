"""Train and evaluate CUCPS with an E2-C next-frame prior in HNM clustering."""

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'src'))

from osl.cucps_temporal import add_temporal_prior, next_frame_indices
from osl.model import DinoBackbone, Model
from osl.temporal_search import location_scores


def _paths(cfg):
    base = HERE
    resolved = {}
    for group, keys in {
        'cucps': ('repo',),
        'prw': ('root',),
        'motion': ('repo', 'weights', 'checkpoint'),
        'temporal_prior': ('cache',),
        'training': ('output_root',),
    }.items():
        for key in keys:
            path = Path(cfg[group][key]).expanduser()
            resolved[(group, key)] = (path if path.is_absolute() else base / path).resolve()
    return resolved


def _load_cucps(cfg):
    paths = _paths(cfg)
    repo = paths[('cucps', 'repo')]
    if not (repo / 'lib/model/faster_rcnn.py').is_file():
        raise FileNotFoundError(f'CUCPS source not found: {repo}')
    sys.path.insert(0, str(HERE / 'external/python-deps'))
    sys.path.insert(0, str(repo))

    import lib.datasets as datasets
    from lib.datasets import PRW
    import lib.datasets.prw as prw_module
    from lib.utils.transforms import get_transform
    prw_module.xrange = range

    prw_root = paths[('prw', 'root')]

    def configured_dataset(args, train=True):
        if args.dataset != 'PRW':
            raise ValueError('This experiment supports PRW only')
        if train:
            return PRW(str(prw_root), get_transform(args.train.use_flipped), mode='train')
        return (
            PRW(str(prw_root), get_transform(False), mode='test'),
            PRW(str(prw_root), get_transform(False), mode='probe'),
        )

    # The official checkout hard-codes a different machine's PRW location.
    # Replacing this factory keeps its dataset and loader behavior intact.
    datasets.get_dataset = configured_dataset
    from lib.model.faster_rcnn import get_model
    return datasets, get_model, paths


def _motion_sample(image, box, output_size, person_crop_cfg):
    width, height = image.size
    out_h, out_w = output_size
    scale = min(out_w / width, out_h / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (out_w - resized_w) // 2, (out_h - resized_h) // 2
    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (out_w, out_h))
    canvas.paste(resized, (pad_x, pad_y))
    pixels = np.asarray(canvas, dtype=np.float32) / 255
    pixels = (pixels - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)

    x1, y1, x2, y2 = map(float, box)
    factor = person_crop_cfg['scale']
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    crop = image.crop((
        int(np.floor(center_x - (x2 - x1) * factor / 2)),
        int(np.floor(center_y - (y2 - y1) * factor / 2)),
        int(np.ceil(center_x + (x2 - x1) * factor / 2)),
        int(np.ceil(center_y + (y2 - y1) * factor / 2)),
    ))
    crop_h, crop_w = person_crop_cfg['size']
    crop_scale = min(crop_w / crop.width, crop_h / crop.height)
    crop_w_resized = max(1, round(crop.width * crop_scale))
    crop_h_resized = max(1, round(crop.height * crop_scale))
    crop = crop.resize((crop_w_resized, crop_h_resized), Image.Resampling.BILINEAR)
    crop_canvas = Image.new('RGB', (crop_w, crop_h))
    crop_canvas.paste(crop, ((crop_w - crop_w_resized) // 2, (crop_h - crop_h_resized) // 2))
    crop_pixels = np.asarray(crop_canvas, dtype=np.float32) / 255
    crop_pixels = (crop_pixels - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    transformed_box = np.array([
        x1 * resized_w / width + pad_x,
        y1 * resized_h / height + pad_y,
        x2 * resized_w / width + pad_x,
        y2 * resized_h / height + pad_y,
    ], dtype=np.float32)
    return pixels.transpose(2, 0, 1), transformed_box, crop_pixels.transpose(2, 0, 1), (resized_w / width, resized_h / height), (pad_x, pad_y)


def _motion_model(cfg, device, paths):
    motion_cfg = cfg['motion']
    model = Model(
        DinoBackbone(str(paths[('motion', 'repo')]), str(paths[('motion', 'weights')]), motion_cfg['name']),
        transformer=True,
        person_branch=True,
    ).to(device)
    checkpoint = torch.load(paths[('motion', 'checkpoint')], map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model


def _cache_metadata(cfg, paths, dataset):
    checkpoint = paths[('motion', 'checkpoint')]
    stat = checkpoint.stat()
    motion_cfg = cfg['motion']
    layout = '\n'.join(
        f"{record['im_name']}|{np.asarray(record['boxes']).tolist()}|{record['cnt']}"
        for record in dataset.record
    )
    return {
        'version': 1,
        'prw_root': str(paths[('prw', 'root')]),
        'train_record_count': len(dataset.get_data_info()[0]),
        'train_layout_sha256': hashlib.sha256(layout.encode()).hexdigest(),
        'motion_checkpoint': str(checkpoint),
        'motion_checkpoint_size': stat.st_size,
        'motion_checkpoint_mtime_ns': stat.st_mtime_ns,
        'motion_repo': str(paths[('motion', 'repo')]),
        'motion_weights': str(paths[('motion', 'weights')]),
        'motion_name': motion_cfg['name'],
        'image_size': motion_cfg['size'],
        'person_crop': motion_cfg['person_crop'],
        'heatmap_radius_cells': motion_cfg['heatmap_radius_cells'],
    }


def _read_prior_cache(path, metadata):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        saved = json.loads(str(data['metadata'].item()))
        if saved != metadata:
            raise ValueError(f'Temporal-prior cache does not match this configuration: {path}')
        row_ptr = data['row_ptr']
        columns = data['columns']
        scores = data['scores']
    return [(columns[row_ptr[i]:row_ptr[i + 1]], scores[row_ptr[i]:row_ptr[i + 1]]) for i in range(len(row_ptr) - 1)]


@torch.no_grad()
def prepare_temporal_prior(cfg, dataset, device, paths):
    """Predict next-image positions from boxes only; identity annotations are never read."""
    cache_path = paths[('temporal_prior', 'cache')]
    metadata = _cache_metadata(cfg, paths, dataset)
    cached = _read_prior_cache(cache_path, metadata)
    if cached is not None:
        print(f'Using temporal-prior cache: {cache_path}', flush=True)
        return cached

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    records = dataset.record
    next_indices = next_frame_indices(records)
    frame_gaps = [
        int(records[next_index]['im_name'].rsplit('_', 1)[1][:-4])
        - int(records[source_index]['im_name'].rsplit('_', 1)[1][:-4])
        for source_index, next_index in next_indices.items()
    ]
    print(
        'PRW next-frame number gaps (min/median/p90/max):',
        min(frame_gaps), float(np.median(frame_gaps)), float(np.percentile(frame_gaps, 90)), max(frame_gaps),
        flush=True,
    )
    num_instances = len(dataset.get_data_info()[0])
    prior_rows = [(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)) for _ in range(num_instances)]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    model = _motion_model(cfg, device, paths)
    # Loading the fixed predictor must not change CUCPS's seeded initialization
    # or data-order RNG relative to the baseline run.
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    motion_cfg = cfg['motion']
    output_size = tuple(motion_cfg['size'])
    chunk_size = int(motion_cfg['batch_size'])
    if chunk_size <= 0:
        raise ValueError('motion.batch_size must be positive')
    amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    radius = int(motion_cfg['heatmap_radius_cells'])
    image_root = Path(dataset.data_path)
    made = 0

    for source_index, next_index in next_indices.items():
        source_record, next_record = records[source_index], records[next_index]
        source_boxes = np.asarray(source_record['boxes'], dtype=np.float32).reshape(-1, 4)
        next_boxes = np.asarray(next_record['boxes'], dtype=np.float32).reshape(-1, 4)
        source_memory_ids = np.asarray(source_record['cnt'], dtype=np.int64).reshape(-1) - 1
        next_memory_ids = np.asarray(next_record['cnt'], dtype=np.int64).reshape(-1) - 1
        if not len(source_boxes) or not len(next_boxes):
            continue

        with Image.open(image_root / source_record['im_name']) as image:
            source_image = image.convert('RGB')
        with Image.open(image_root / next_record['im_name']) as image:
            next_size = image.size
        if source_image.size != next_size:
            raise ValueError(f'Adjacent PRW image dimensions differ: {source_record["im_name"]}, {next_record["im_name"]}')

        sample_data = [
            _motion_sample(source_image, box, output_size, motion_cfg['person_crop'])
            for box in source_boxes
        ]
        scale, offset = sample_data[0][3], sample_data[0][4]
        candidate_scores = [[] for _ in source_boxes]
        for start in range(0, len(source_boxes), chunk_size):
            chunk = sample_data[start:start + chunk_size]
            images = torch.from_numpy(np.stack([sample[0] for sample in chunk])).to(device)
            boxes = torch.from_numpy(np.stack([sample[1] for sample in chunk])).to(device)
            crops = torch.from_numpy(np.stack([sample[2] for sample in chunk])).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(images, boxes, crops)
            heatmaps = logits.float().flatten(1).softmax(1).reshape(logits.shape[0], *logits.shape[-2:]).cpu().numpy()
            for local_index, heatmap in enumerate(heatmaps):
                candidate_scores[start + local_index] = location_scores(
                    heatmap, next_boxes, scale, offset, output_size, radius
                )

        for source_id, scores in zip(source_memory_ids, candidate_scores):
            if source_id < 0 or source_id >= num_instances:
                raise ValueError('PRW instance index is outside the CUCPS feature memory')
            prior_rows[int(source_id)] = (next_memory_ids.copy(), np.asarray(scores, dtype=np.float32))
        made += len(source_boxes)
        if made % 1000 < len(source_boxes):
            print(f'predicted next-frame priors for {made} / {num_instances} train boxes', flush=True)

    row_ptr = np.zeros(num_instances + 1, dtype=np.int64)
    for index, (columns, _) in enumerate(prior_rows):
        row_ptr[index + 1] = row_ptr[index] + len(columns)
    columns = np.concatenate([row[0] for row in prior_rows]) if row_ptr[-1] else np.empty(0, dtype=np.int64)
    scores = np.concatenate([row[1] for row in prior_rows]) if row_ptr[-1] else np.empty(0, dtype=np.float32)
    np.savez_compressed(cache_path, row_ptr=row_ptr, columns=columns, scores=scores,
                        metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(f'Saved {made} source-box priors ({len(columns)} source-to-next candidates): {cache_path}', flush=True)
    return prior_rows


def _attach_temporal_prior(clustering, prior_rows, weight):
    if not clustering.use_hnm:
        raise ValueError('Temporal CUCPS requires HNM to remain enabled')
    original = clustering.hard_negative_mining

    def temporal_hnm(self, similarities, coapp, pids, memory, threshold):
        adjusted = add_temporal_prior(similarities, pids, prior_rows, weight)
        return original(adjusted, coapp, pids, memory, threshold)

    clustering.hard_negative_mining = MethodType(temporal_hnm, clustering)


def train(cfg, variant):
    if variant not in ('baseline', 'temporal'):
        raise ValueError('variant must be baseline or temporal')
    datasets, get_model, paths = _load_cucps(cfg)
    from configs import args_faster_rcnn
    from lib.utils.misc import Nestedspace, get_lr_scheduler, get_optimizer
    import lib.utils.trainer as trainer_module
    from lib.utils.trainer import Trainer

    # The upstream trainer references time.time() without importing time.
    trainer_module.time = time
    training = cfg['training']
    run_root = paths[('training', 'output_root')] / variant
    run_dir = run_root / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)
    cli = [
        '--dataset', 'PRW', '--num_pids', str(training['num_pids']),
        '--num_cq_size', str(training['num_cq_size']),
        '--lr_decay_step', str(training['lr_decay_step']),
        '--batch_size', str(training['batch_size']), '--nw', str(training['workers']),
        '--path', str(run_dir), '--seed', str(training['seed']), '--device', cfg['runtime']['device'],
    ]
    args = args_faster_rcnn().parse_args(cli, namespace=Nestedspace())
    if args.dataset != 'PRW':
        raise ValueError('Only PRW is supported')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    train_loader, train_info = datasets.get_data_loader(args, train=True)
    print(f'PRW train images: {len(train_loader.dataset)}; memory instances: {len(train_info[0])}', flush=True)
    frame_pairs = next_frame_indices(train_loader.dataset.record)
    frame_gaps = [
        int(train_loader.dataset.record[next_index]['im_name'].rsplit('_', 1)[1][:-4])
        - int(train_loader.dataset.record[source_index]['im_name'].rsplit('_', 1)[1][:-4])
        for source_index, next_index in frame_pairs.items()
    ]
    if variant == 'temporal':
        prior_rows = prepare_temporal_prior(cfg, train_loader.dataset, device, paths)
    else:
        prior_rows = None

    model = get_model(args, training=True, pretrained_backbone=True)
    model.roi_heads.reid_regressor.set_scene_vector(train_info)
    if prior_rows is not None:
        expected = model.roi_heads.reid_regressor.clustering.cnt2snum.numel()
        if len(prior_rows) != expected:
            raise ValueError(f'Temporal-prior rows ({len(prior_rows)}) do not match CUCPS memory ({expected})')
        _attach_temporal_prior(model.roi_heads.reid_regressor.clustering, prior_rows,
                               float(cfg['temporal_prior']['weight']))
    model.to(device)
    optimizer = get_optimizer(args, model)
    scheduler = get_lr_scheduler(args, optimizer)
    if args.resume is not None:
        from lib.utils.misc import resume_from_checkpoint
        args, model, optimizer, scheduler = resume_from_checkpoint(args, model, optimizer, scheduler)

    args.export_to_json(str(run_dir / 'args.json'))
    metadata = {
        'variant': variant,
        'temporal_weight': float(cfg['temporal_prior']['weight']) if variant == 'temporal' else 0.0,
        'temporal_prior_cache': str(paths[('temporal_prior', 'cache')]) if variant == 'temporal' else None,
        'motion_checkpoint': str(paths[('motion', 'checkpoint')]) if variant == 'temporal' else None,
        'prw_identity_labels_used_for_training': False,
        'next_frame_pairs': len(frame_pairs),
        'frame_number_gap': {
            'min': int(min(frame_gaps)),
            'median': float(np.median(frame_gaps)),
            'p90': float(np.percentile(frame_gaps, 90)),
            'max': int(max(frame_gaps)),
        },
    }
    (run_dir / 'experiment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'variant={variant} output={run_dir} temporal_weight={metadata["temporal_weight"]}', flush=True)
    Trainer(args, model, train_loader, optimizer, scheduler, device).run()
    print(f'TRAINING_COMPLETE checkpoint={run_dir / "checkpoint_epoch_last.pth"}', flush=True)


def evaluate(cfg, checkpoint):
    _, get_model, _ = _load_cucps(cfg)
    from lib.utils.misc import Nestedspace
    from runs.test import main as official_test

    # Build the same test arguments used by CUCPS and let it read run args.json.
    new_args = Nestedspace()
    new_args.test = Nestedspace()
    new_args.test.checkpoint_name = str(Path(checkpoint).resolve())
    result = official_test(new_args, get_model)
    sidecar = Path(checkpoint).with_name(Path(checkpoint).name.replace('.pth', '-regular.json'))
    if not sidecar.is_file():
        raise RuntimeError(f'Official CUCPS evaluation did not write {sidecar}; result={result}')
    print(f'OFFICIAL_EVALUATION_COMPLETE metrics={sidecar}', flush=True)


def compare(baseline_checkpoint, temporal_checkpoint, output):
    paths = [Path(baseline_checkpoint), Path(temporal_checkpoint)]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError('Both checkpoint files must exist')
    experiment_paths = [path.parent / 'experiment.json' for path in paths]
    if not all(path.is_file() for path in experiment_paths):
        raise FileNotFoundError(f'Run metadata is missing: {experiment_paths}')
    experiments = [json.loads(path.read_text()) for path in experiment_paths]
    if [item.get('variant') for item in experiments] != ['baseline', 'temporal']:
        raise ValueError('Pass the baseline checkpoint first and temporal checkpoint second')
    metrics_paths = [path.with_name(path.name.replace('.pth', '-regular.json')) for path in paths]
    if not all(path.is_file() for path in metrics_paths):
        missing = [str(path) for path in metrics_paths if not path.is_file()]
        raise FileNotFoundError(f'Run official test for both checkpoints first: {missing}')
    metrics = [json.loads(path.read_text()) for path in metrics_paths]
    report = {
        'protocol': 'Official CUCPS PRW regular full-gallery evaluation',
        'baseline_checkpoint': str(paths[0].resolve()),
        'temporal_checkpoint': str(paths[1].resolve()),
        'temporal_weight': experiments[1]['temporal_weight'],
        'baseline': metrics[0],
        'temporal': metrics[1],
        'delta': {
            'mAP_percentage_points': 100 * (metrics[1]['mAP'] - metrics[0]['mAP']),
            'rank1_percentage_points': 100 * (metrics[1]['top_k'][0] - metrics[0]['top_k'][0]),
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['delta'], indent=2), flush=True)
    print(f'Saved comparison: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='action', required=True)
    for action in ('train', 'test'):
        sub = subparsers.add_parser(action)
        sub.add_argument('--config', required=True)
        if action == 'train':
            sub.add_argument('--variant', required=True, choices=('baseline', 'temporal'))
        else:
            sub.add_argument('--checkpoint', required=True)
    sub = subparsers.add_parser('compare')
    sub.add_argument('--baseline-checkpoint', required=True)
    sub.add_argument('--temporal-checkpoint', required=True)
    sub.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'compare':
        compare(args.baseline_checkpoint, args.temporal_checkpoint, args.output)
        return
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.action == 'train':
        train(cfg, args.variant)
    else:
        evaluate(cfg, args.checkpoint)


if __name__ == '__main__':
    main()
