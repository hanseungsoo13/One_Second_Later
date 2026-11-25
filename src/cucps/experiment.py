"""Train and evaluate CUCPS with an OSL next-frame prior in HNM clustering."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / 'src'))


from cucps.compat import (
    ModernMemoryLayer, _install_memory_compatibility,
    _install_numpy_detection_compatibility, _load_prw_probes_utf8,
)
from integration.temporal import (add_temporal_prior, next_frame_indices,
                                build_temporal_graph, refine_temporal_positives)
from cucps.config import validate_cucps_config
from osl.model import DinoBackbone, Model
from integration.variants import resolve_cucps_variant
from osl.segmentation import SegFormerPredictor
from integration.motion_inputs import _motion_sample, _predict_relative_poses
from integration import prior as _prior
from integration import predictor as _predictor
from integration.prior import _cache_metadata, _read_prior_cache


def _paths(cfg):
    base = HERE
    resolved = {}
    for group, keys in {
        'cucps': ('repo',),
        'prw': ('root',),
        'motion': ('repo', 'weights', 'checkpoint', 'pose_checkpoint'),
        'temporal_prior': ('cache',),
        'training': ('output_root',),
    }.items():
        for key in keys:
            if key not in cfg[group]:
                continue
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
    PRW.load_probes = _load_prw_probes_utf8
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
    import lib.model.reID.regressor as regressor_module
    _install_memory_compatibility(regressor_module)
    return datasets, get_model, paths


def _motion_model(cfg, device, paths, variant='temporal'):
    """Compatibility wrapper for constructing the fixed OSL predictor."""
    return _predictor.build_motion_model(
        cfg, device, paths, variant, backbone_class=DinoBackbone, model_class=Model)


def _make_pose_estimator(cfg, device):
    return _predictor.make_pose_estimator(cfg, device)


def prepare_temporal_prior(cfg, dataset, device, paths, variant='temporal'):
    """Build or reuse the OSL location prior for CUCPS training."""
    return _prior.prepare_temporal_prior(
        cfg, dataset, device, paths, variant,
        motion_model=_motion_model, make_pose_estimator=_make_pose_estimator,
        predict_relative_poses=_predict_relative_poses,
        semantic_predictor_factory=SegFormerPredictor, metadata_factory=_cache_metadata,
    )


def _attach_temporal_prior(clustering, prior_rows, weight):
    if not clustering.use_hnm:
        raise ValueError('Temporal CUCPS requires HNM to remain enabled')
    original = clustering.hard_negative_mining

    def temporal_hnm(self, similarities, coapp, pids, memory, threshold):
        adjusted = add_temporal_prior(similarities, pids, prior_rows, weight)
        return original(adjusted, coapp, pids, memory, threshold)

    clustering.hard_negative_mining = MethodType(temporal_hnm, clustering)


def _attach_temporal_graph(clustering, prior_rows, cfg):
    original = clustering.predict

    def graph_predict(self, memory, targets):
        graph = build_temporal_graph(
            prior_rows, memory, float(cfg['location_weight']),
            float(cfg['min_edge_appearance']), float(cfg['min_edge_combined']),
            float(cfg['min_edge_margin']))
        return refine_temporal_positives(
            memory, targets, graph, self.cnt2snum, original,
            hops=int(cfg['hops']), alpha=float(cfg['alpha']),
            min_confidence=float(cfg['min_path_confidence']),
            min_appearance=float(cfg['min_positive_appearance']))

    clustering.predict = MethodType(graph_predict, clustering)


def _resolve_device(value):
    """Resolve CUDA's implicit device to the first visible GPU."""
    device = torch.device(value)
    if device.type == 'cuda' and device.index is None:
        return torch.device('cuda', 0)
    return device


def _exclusive_epoch_end(epochs, start_epoch=1):
    """Official CUCPS trainer uses an exclusive upper bound in range()."""
    epochs = int(epochs)
    start_epoch = int(start_epoch)
    if epochs <= 0 or start_epoch <= 0:
        raise ValueError('epochs and start_epoch must be positive')
    return start_epoch + epochs


def train(cfg, variant):
    specification = validate_cucps_config(cfg, variant)
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
        '--epochs', str(_exclusive_epoch_end(training['epochs'])),
        '--path', str(run_dir), '--seed', str(training['seed']), '--device', cfg['runtime']['device'],
    ]
    args = args_faster_rcnn().parse_args(cli, namespace=Nestedspace())
    if args.dataset != 'PRW':
        raise ValueError('Only PRW is supported')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _resolve_device(args.device)
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
    if variant != 'baseline':
        prior_rows = prepare_temporal_prior(cfg, train_loader.dataset, device, paths, variant)
    else:
        prior_rows = None

    model = get_model(args, training=True, pretrained_backbone=True)
    model.roi_heads.reid_regressor.set_scene_vector(train_info)
    if prior_rows is not None:
        expected = model.roi_heads.reid_regressor.clustering.cnt2snum.numel()
        if len(prior_rows) != expected:
            raise ValueError(f'Temporal-prior rows ({len(prior_rows)}) do not match CUCPS memory ({expected})')
        if variant == 'graph':
            _attach_temporal_graph(model.roi_heads.reid_regressor.clustering, prior_rows, cfg['temporal_graph'])
        else:
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
        'temporal_graph': cfg.get('temporal_graph') if variant == 'graph' else None,
        'requested_epochs': int(training['epochs']),
        'trainer_epoch_end_exclusive': int(args.train.epochs),
        'temporal_weight': float(cfg['temporal_prior']['weight']) if variant not in ('baseline', 'graph') else 0.0,
        'temporal_prior_cache': str(paths[('temporal_prior', 'cache')]) if variant != 'baseline' else None,
        'motion_checkpoint': str(paths[('motion', 'pose_checkpoint' if specification.pose_fusion else 'checkpoint')]) if specification else None,
        'pose_checkpoint': str(paths[('motion', 'pose_checkpoint')]) if specification and specification.pose_fusion else None,
        'pose_model': cfg['motion'].get('pose', {}).get('model') if specification and specification.pose_fusion else None,
        'pose_annotations_used': False,
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
    import lib.utils.evaluator as evaluator_module
    _install_numpy_detection_compatibility(evaluator_module)
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


def compare(checkpoints, output):
    """Compare any trained variants against the required baseline."""
    if 'baseline' not in checkpoints:
        raise ValueError('A baseline checkpoint is required')
    variants, metrics = {}, {}
    for variant, checkpoint in checkpoints.items():
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f'{variant} checkpoint does not exist: {path}')
        experiment_path = path.parent / 'experiment.json'
        metric_path = path.with_name(path.name.replace('.pth', '-regular.json'))
        if not experiment_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError(f'Run official test for {variant} first: {metric_path}')
        experiment = json.loads(experiment_path.read_text())
        stored_variant = experiment.get('variant')
        same_variant = stored_variant == variant
        if not same_variant and stored_variant not in (None, 'baseline', 'graph') and variant not in ('baseline', 'graph'):
            same_variant = resolve_cucps_variant(stored_variant) == resolve_cucps_variant(variant)
        if not same_variant:
            raise ValueError(f'{path} is tagged {experiment.get("variant")}, expected {variant}')
        variants[variant] = {
            'checkpoint': str(path.resolve()),
            'temporal_weight': experiment['temporal_weight'],
            'metrics': json.loads(metric_path.read_text()),
        }
        metrics[variant] = variants[variant]['metrics']
    base = metrics['baseline']
    for variant, row in variants.items():
        row['delta'] = {
            'mAP_percentage_points': 100 * (row['metrics']['mAP'] - base['mAP']),
            'rank1_percentage_points': 100 * (row['metrics']['top_k'][0] - base['top_k'][0]),
        }
    report = {
        'protocol': 'Official CUCPS PRW regular full-gallery evaluation',
        'variants': variants,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v['delta'] for k, v in variants.items()}, indent=2), flush=True)
    print(f'Saved comparison: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='action', required=True)
    for action in ('train', 'test'):
        sub = subparsers.add_parser(action)
        sub.add_argument('--config', required=True)
        if action == 'train':
            sub.add_argument('--variant', required=True, choices=('baseline', 'temporal', 'pose_p1', 'pose_p2', 'graph', 'full_bbox_a', 'full_bbox_b', 'full_bbox_c', 'full_bbox_pose1', 'full_bbox_pose2', 'full_bbox_pose_seg'))
        else:
            sub.add_argument('--checkpoint', required=True)
    sub = subparsers.add_parser('compare')
    sub.add_argument('--baseline-checkpoint', required=True)
    sub.add_argument('--temporal-checkpoint')
    sub.add_argument('--pose-p1-checkpoint')
    sub.add_argument('--pose-p2-checkpoint')
    sub.add_argument('--model-checkpoint', action='append', default=[], metavar='VARIANT=PATH',
                     help='Additional named model checkpoint; repeat for each variant')
    sub.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'compare':
        checkpoints = {'baseline': args.baseline_checkpoint}
        checkpoints.update({variant: path for variant, path in (
            ('temporal', args.temporal_checkpoint), ('pose_p1', args.pose_p1_checkpoint),
            ('pose_p2', args.pose_p2_checkpoint),
        ) if path})
        for item in args.model_checkpoint:
            name, separator, path = item.partition('=')
            if not separator or not path or name in checkpoints or name in ('baseline', 'graph'):
                parser.error('--model-checkpoint requires a unique model VARIANT=PATH')
            resolve_cucps_variant(name)
            checkpoints[name] = path
        compare(checkpoints, args.output)
        return
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.action == 'train':
        train(cfg, args.variant)
    else:
        evaluate(cfg, args.checkpoint)


if __name__ == '__main__':
    main()
