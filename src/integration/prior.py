"""Build and validate the OSL-to-CUCPS temporal prior cache."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from integration.location import location_scores
from integration.motion_inputs import _motion_sample
from integration.temporal import next_frame_indices
from integration.variants import resolve_cucps_variant


def _cache_metadata(cfg, paths, dataset, variant='temporal'):
    specification = resolve_cucps_variant(variant)
    if specification is None:
        raise ValueError('baseline does not use a temporal-prior cache')
    checkpoint_key = 'pose_checkpoint' if specification.pose_fusion else 'checkpoint'
    checkpoint = paths[('motion', checkpoint_key)]
    stat = checkpoint.stat()
    motion_cfg = cfg['motion']
    layout = '\n'.join(
        f"{record['im_name']}|{np.asarray(record['boxes']).tolist()}|{record['cnt']}"
        for record in dataset.record
    )
    metadata = {
        'version': 2,
        'model_variant': specification.name,
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
        'person_crop': motion_cfg.get('person_crop') if specification.person_branch else None,
        'heatmap_radius_cells': motion_cfg['heatmap_radius_cells'],
    }
    if variant == 'graph':
        metadata['graph_cache'] = {'max_frame_gap': cfg['temporal_graph']['max_frame_gap'],
                                   'min_heatmap_peak': cfg['temporal_graph']['min_heatmap_peak']}
    if specification.semantic:
        metadata['semantic'] = motion_cfg['semantic']
    if specification.pose_fusion:
        metadata['pose'] = motion_cfg.get('pose', {})
    return metadata


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
def prepare_temporal_prior(cfg, dataset, device, paths, variant='temporal', *,
                           motion_model, make_pose_estimator, predict_relative_poses,
                           semantic_predictor_factory, metadata_factory):
    """Predict next-image positions from boxes only; identity annotations are never read."""
    cache_path = paths[('temporal_prior', 'cache')]
    metadata = metadata_factory(cfg, paths, dataset, variant)
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
    model = motion_model(cfg, device, paths, variant)
    specification = resolve_cucps_variant(variant)
    pose_estimator = make_pose_estimator(cfg, device) if specification.pose_fusion else None
    semantic_predictor = None
    if specification.semantic:
        semantic = cfg['motion']['semantic']
        size = cfg['motion']['size']
        if any(n % 16 for n in size):
            raise ValueError('motion.size must be divisible by 16 for segmentation')
        semantic_predictor = semantic_predictor_factory(semantic['model'], semantic['revision'], size, [n // 16 for n in size], semantic.get('device', str(device)))
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
        if variant == 'graph':
            gap = int(next_record['im_name'].rsplit('_', 1)[1][:-4]) - int(source_record['im_name'].rsplit('_', 1)[1][:-4])
            if gap > int(cfg['temporal_graph']['max_frame_gap']):
                continue
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
            _motion_sample(source_image, box, output_size, motion_cfg['person_crop'] if specification.person_branch else None)
            for box in source_boxes
        ]
        poses = predict_relative_poses(pose_estimator, source_image, source_boxes) if pose_estimator else None
        semantic_map = semantic_predictor(source_image).astype(np.float16).astype(np.float32) if semantic_predictor else None
        scale, offset = sample_data[0][3], sample_data[0][4]
        candidate_scores = [[] for _ in source_boxes]
        for start in range(0, len(source_boxes), chunk_size):
            chunk = sample_data[start:start + chunk_size]
            images = torch.from_numpy(np.stack([sample[0] for sample in chunk])).to(device)
            boxes = torch.from_numpy(np.stack([sample[1] for sample in chunk])).to(device)
            crops = (torch.from_numpy(np.stack([sample[2] for sample in chunk])).to(device)
                     if specification.person_branch else None)
            pose_chunk = torch.from_numpy(poses[start:start + chunk_size]).to(device) if poses is not None else None
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                semantic_chunk = None if semantic_map is None else torch.from_numpy(semantic_map).unsqueeze(0).expand(len(chunk), -1, -1, -1).to(device)
                logits = model(images, boxes, crops, pose_chunk, semantic_chunk) if semantic_chunk is not None else model(images, boxes, crops, pose_chunk)
            heatmaps = logits.float().flatten(1).softmax(1).reshape(logits.shape[0], *logits.shape[-2:]).cpu().numpy()
            for local_index, heatmap in enumerate(heatmaps):
                if variant == 'graph' and heatmap.max() < float(cfg['temporal_graph']['min_heatmap_peak']):
                    continue
                candidate_scores[start + local_index] = location_scores(
                    heatmap, next_boxes, scale, offset, output_size, radius
                )

        for source_id, scores in zip(source_memory_ids, candidate_scores):
            if variant == 'graph' and not len(scores):
                continue
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
