"""Candidate scoring and evaluation for a temporal person-search reranker."""
import numpy as np

def location_scores(heatmap, boxes, scale, offset, input_size, radius=1):
    """Read max-normalized heat near each original-image box's bottom centre."""
    heatmap = np.asarray(heatmap, dtype=float)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all() or radius < 0:
        raise ValueError('Invalid heatmap or radius')
    h, w = heatmap.shape
    ih, iw = input_size
    scale = np.asarray(scale)
    offset = np.asarray(offset)
    peak = heatmap.max(initial=0)
    heatmap = heatmap / peak if peak > 0 else np.zeros_like(heatmap)
    out = []
    for x1, y1, x2, y2 in boxes:
        x = (x1 + x2) / 2 * scale[0] + offset[0]
        y = y2 * scale[1] + offset[1]
        if x < 0 or y < 0 or x >= iw or (y >= ih):
            out.append(0.0)
            continue
        ix = min(w - 1, int(x * w / iw))
        iy = min(h - 1, int(y * h / ih))
        out.append(float(heatmap[max(0, iy - radius):min(h, iy + radius + 1), max(0, ix - radius):min(w, ix + radius + 1)].mean()))
    return np.asarray(out)

def rank_candidates(appearance, position, weight):
    a = np.asarray(appearance, dtype=float)
    p = np.asarray(position, dtype=float)
    if a.shape != p.shape or a.ndim != 1 or (not np.isfinite(a).all()) or (not np.isfinite(p).all()) or (not np.isfinite(weight)):
        raise ValueError('Invalid candidate scores')
    return np.argsort(-(a + weight * p), kind='stable')

def search_metrics(records, weight):
    rank1 = rr = recall = 0.0
    n = len(records)
    for r in records:
        pos = np.asarray(r['positive'], dtype=bool)
        order = rank_candidates(r['appearance'], r['position'], weight)
        if len(pos) != len(order):
            raise ValueError('Candidate labels do not match scores')
        if pos.any():
            recall += 1
            rank = int(np.flatnonzero(pos[order])[0]) + 1
            rank1 += rank == 1
            rr += 1 / rank
    return {'queries': n, 'detector_recall': recall / n if n else 0.0, 'rank1': rank1 / n if n else 0.0, 'mrr': rr / n if n else 0.0}

def select_weight(records, weights):
    if not records:
        raise ValueError('No validation records')
    scores = [{'weight': float(w), **search_metrics(records, float(w))} for w in weights]
    if not scores:
        raise ValueError('No weights supplied')
    best = max(scores, key=lambda r: (r['rank1'], r['mrr'], -abs(r['weight'])))
    return {**best, 'trials': scores}
