"""Temporal location priors for CUCPS pseudo-label clustering."""

import re

import numpy as np
import torch


_PRW_FRAME = re.compile(r"(c\d+s\d+)_(\d+)\.jpg$")


def add_temporal_prior(similarities, query_ids, prior_rows, weight):
    """Return similarities with directed source-to-next-frame location scores."""
    if similarities.ndim != 2:
        raise ValueError('similarities must be a 2D tensor')
    if len(query_ids) != similarities.shape[0]:
        raise ValueError('query_ids must align with similarity rows')
    if similarities.shape[1] != len(prior_rows):
        raise ValueError('prior rows must align with memory columns')
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('weight must be finite and nonnegative')

    result = similarities.clone()
    if weight == 0:
        return result
    for row, source_id in enumerate(query_ids.detach().cpu().tolist()):
        columns, scores = prior_rows[int(source_id)]
        if len(columns) != len(scores):
            raise ValueError(f'prior row {source_id} has mismatched columns and scores')
        if not len(columns):
            continue
        columns = torch.as_tensor(columns, dtype=torch.long, device=result.device)
        scores = torch.as_tensor(scores, dtype=result.dtype, device=result.device)
        if torch.any(columns < 0) or torch.any(columns >= result.shape[1]):
            raise ValueError(f'prior row {source_id} contains an invalid memory index')
        if not torch.isfinite(scores).all() or torch.any(scores < 0):
            raise ValueError(f'prior row {source_id} contains an invalid location score')
        result[row].index_add_(0, columns, scores * float(weight))
    return result


def next_frame_indices(records):
    """Map each PRW train image to the next available image in its segment."""
    groups = {}
    for index, record in enumerate(records):
        name = record['im_name']
        match = _PRW_FRAME.fullmatch(name)
        if not match:
            raise ValueError(f'Invalid PRW frame name: {name}')
        groups.setdefault(match.group(1), []).append((int(match.group(2)), index))

    result = {}
    for frames in groups.values():
        frames.sort()
        for (frame_a, index_a), (frame_b, index_b) in zip(frames, frames[1:]):
            if frame_a == frame_b:
                raise ValueError('Duplicate frame number within a PRW segment')
            result[index_a] = index_b
    return result
