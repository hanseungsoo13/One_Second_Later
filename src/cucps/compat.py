"""Compatibility shims for the pinned upstream CUCPS release.

These adaptations preserve upstream behavior while supporting modern PyTorch,
NumPy, and the official PRW probe text format.
"""
from pathlib import Path

import numpy as np
import torch


def _install_memory_compatibility(regressor_module):
    def memory_forward(self, inputs, targets, epoch=None):
        return ModernMemoryLayer.apply(inputs, targets, self.mem, 0.5)

    regressor_module.Memory.forward = memory_forward


class ModernMemoryLayer(torch.autograd.Function):
    """PyTorch 2-compatible form of CUCPS's memory update autograd function."""

    @staticmethod
    def forward(ctx, inputs, targets, memory, alpha):
        ctx.memory = memory
        ctx.alpha = float(alpha)
        ctx.save_for_backward(inputs, targets)
        return inputs.mm(memory.t())

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        memory = ctx.memory
        grad_inputs = grad_outputs.mm(memory) if ctx.needs_input_grad[0] else None
        with torch.no_grad():
            for feature, target in zip(inputs, targets):
                memory[target] = ctx.alpha * memory[target] + (1.0 - ctx.alpha) * feature
                memory[target] /= memory[target].norm()
        return grad_inputs, None, None, None

def _load_prw_probes_utf8(dataset):
    """Read the official CRLF probe list as decoded text on Python 3."""
    query_info = Path(dataset.root) / 'query_info.txt'
    with query_info.open('r', encoding='utf-8', newline='') as handle:
        lines = handle.readlines()

    probes = []
    for line in lines:
        fields = line.split()
        pid = int(fields[0])
        x, y, width, height = map(float, fields[1:5])
        roi = np.array([x, y, x + width, y + height], dtype=np.int32)
        roi = np.clip(roi, 0, None)
        image_name = fields[5].strip() + '.jpg'
        probes.append({
            'im_name': image_name,
            'boxes': roi[np.newaxis, :],
            'gt_pids': np.array([pid]),
            'flipped': False,
            'cam_id': dataset._get_cam_id(image_name),
        })
    return probes


def _install_numpy_detection_compatibility(evaluator_module):
    legacy_metric = evaluator_module.detection_performance_calc

    def detection_performance_calc(dataset, gallery_det, *args, **kwargs):
        detections = [det.tolist() if isinstance(det, np.ndarray) else det for det in gallery_det]
        return legacy_metric(dataset, detections, *args, **kwargs)

    evaluator_module.detection_performance_calc = detection_performance_calc
