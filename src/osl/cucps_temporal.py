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


def build_temporal_graph(prior_rows, memory, location_weight, min_appearance,
                         min_combined, min_margin):
    """Build a sparse undirected graph using mutual matches within frame pairs."""
    if not 0 <= location_weight <= 1:
        raise ValueError('location_weight must be between 0 and 1')
    sources, destinations, locations = [], [], []
    groups = {}
    row_offsets = {}
    for source, (columns, scores) in enumerate(prior_rows):
        if len(columns) != len(scores):
            raise ValueError('Prior columns and scores must align')
        if len(columns):
            key = tuple(int(x) for x in columns)
            groups.setdefault(key, []).append(source)
            row_offsets[source] = len(sources)
            sources.extend([source] * len(columns))
            destinations.extend(key)
            locations.extend(scores)
    graph = [{} for _ in prior_rows]
    if not sources:
        return graph
    with torch.no_grad():
        src = torch.as_tensor(sources, device=memory.device)
        dst = torch.as_tensor(destinations, device=memory.device)
        appearance = (memory[src] * memory[dst]).sum(1).cpu().numpy()
    loc = np.asarray(locations, dtype=np.float32)
    combined = (1 - location_weight) * appearance + location_weight * loc
    for columns, rows in groups.items():
        indexes = [row_offsets[row] + j for row in rows for j in range(len(columns))]
        a = appearance[indexes].reshape(len(rows), len(columns))
        s = combined[indexes].reshape(len(rows), len(columns))
        def margin(values, index):
            return float('inf') if len(values) == 1 else float(values[index] - np.partition(values, -2)[-2])
        for i, source in enumerate(rows):
            j = int(np.argmax(s[i]))
            if int(np.argmax(s[:, j])) != i:
                continue
            if (a[i, j] < min_appearance or s[i, j] < min_combined or
                    margin(s[i], j) < min_margin or margin(s[:, j], i) < min_margin):
                continue
            destination = columns[j]
            weight = float(np.clip(s[i, j], 0, 1))
            graph[source][destination] = weight
            graph[destination][source] = weight
    return graph


def _reachable(graph, start, hops, alpha):
    best = {int(start): 1.0}
    frontier = dict(best)
    for _ in range(hops):
        following = {}
        for node, confidence in frontier.items():
            for neighbor, weight in graph[node].items():
                score = confidence * weight * alpha
                if score > best.get(neighbor, 0):
                    best[neighbor] = score
                    following[neighbor] = max(following.get(neighbor, 0), score)
        frontier = following
    return best


def refine_temporal_positives(memory, targets, graph, scenes, predict, *, hops,
                              alpha, min_confidence, min_appearance):
    """Expand original CUCPS positives through bounded, maximum-confidence paths."""
    if hops < 0 or not 0 <= alpha <= 1:
        raise ValueError('hops must be nonnegative and alpha must be in [0, 1]')
    target_list = targets.detach().cpu().tolist()
    paths = {int(t): _reachable(graph, t, hops, alpha) for t in target_list}
    seeds = sorted({node for path in paths.values() for node in path})
    seed_ids = torch.as_tensor(seeds, dtype=torch.long, device=targets.device)
    original_labels = predict(memory, targets)
    labels = predict(memory, seed_ids)
    seed_rows = {seed: i for i, seed in enumerate(seeds)}
    result = original_labels.clone()
    scene_list = scenes.detach().cpu().tolist()
    for row, target in enumerate(target_list):
        candidates = {}
        for seed, left_weight in paths[target].items():
            seed_labels = result[row] if seed == target else labels[seed_rows[seed]]
            positive_ids = torch.nonzero(seed_labels > 0).flatten().cpu().tolist()
            for positive in positive_ids:
                for candidate, right_weight in _reachable(graph, positive, hops, alpha).items():
                    confidence = left_weight * right_weight
                    if confidence >= min_confidence:
                        candidates[candidate] = max(candidates.get(candidate, 0), confidence)
        occupied = {scene_list[i] for i in torch.nonzero(result[row] > 0).flatten().cpu().tolist()}
        for candidate, confidence in sorted(candidates.items(), key=lambda item: (-item[1], item[0])):
            if scene_list[candidate] in occupied:
                continue
            if float(torch.dot(memory[target], memory[candidate])) < min_appearance:
                continue
            result[row, candidate] = 1
            occupied.add(scene_list[candidate])
    return result
