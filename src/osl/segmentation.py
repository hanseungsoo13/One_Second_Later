"""Image-aligned semantic probabilities and a versioned per-image cache."""
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def image_key(record):
    values = [str(record['image_path']), list(record['image_size'])]
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def cache_metadata(model, revision, image_size, grid_size):
    return {'format': 1, 'model': model, 'revision': revision, 'image_size': list(image_size),
            'grid_size': list(grid_size), 'classes': 150, 'dtype': 'float16',
            'alignment': 'letterbox-before-segformer-bilinear-logits-then-softmax'}


def letterbox(image, size):
    """Match AnnotationDataset's RGB scene resizing and black padding exactly."""
    w, h = image.size; oh, ow = size
    scale = min(ow / w, oh / h)
    nw, nh = round(w * scale), round(h * scale)
    canvas = Image.new('RGB', (ow, oh))
    canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), ((ow-nw)//2, (oh-nh)//2))
    return canvas


class SegFormerPredictor:
    """Frozen SegFormer-B2 produces 150 class probabilities at the scene grid."""
    def __init__(self, model, revision, image_size, grid_size, device):
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        self.image_size, self.grid_size, self.device = image_size, grid_size, torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model, revision=revision, use_fast=False)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model, revision=revision).to(self.device)
        if self.model.config.num_labels != 150:
            raise ValueError('Expected a 150-class ADE20K SegFormer checkpoint')
        self.model.requires_grad_(False).eval()

    @torch.inference_mode()
    def __call__(self, image):
        canvas = letterbox(image.convert('RGB'), self.image_size)
        inputs = self.processor(images=canvas, return_tensors='pt')
        logits = self.model(pixel_values=inputs['pixel_values'].to(self.device)).logits.float()
        logits = F.interpolate(logits, size=self.grid_size, mode='bilinear', align_corners=False)
        return logits.softmax(dim=1)[0].cpu().numpy()


class SegmentationCache:
    def __init__(self, path, metadata, writable=False):
        self.path = Path(path)
        self.metadata = metadata
        if writable:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path)
            self.db.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1), json TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS maps (key TEXT PRIMARY KEY, data BLOB NOT NULL)')
        else:
            self.db = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True)
        raw = json.dumps(metadata, sort_keys=True, separators=(',', ':'))
        row = self.db.execute('SELECT json FROM metadata WHERE id=1').fetchone()
        if row is None:
            if not writable: raise ValueError('Segmentation cache metadata is missing')
            self.db.execute('INSERT INTO metadata VALUES (1, ?)', (raw,)); self.db.commit()
        elif row[0] != raw:
            self.db.close(); raise ValueError(f'Segmentation cache metadata mismatch: {path}')

    def keys(self): return {row[0] for row in self.db.execute('SELECT key FROM maps')}
    def contains(self, key): return self.db.execute('SELECT 1 FROM maps WHERE key=?', (key,)).fetchone() is not None
    def put(self, key, values):
        array = np.asarray(values, dtype=np.float32)
        expected = (self.metadata['classes'], *self.metadata['grid_size'])
        if array.shape != expected or not np.isfinite(array).all() or np.any(array < 0) or not np.allclose(array.sum(axis=0), 1, atol=1e-3):
            raise ValueError(f'Invalid semantic probability map; expected {expected}')
        self.db.execute('INSERT INTO maps VALUES (?, ?)', (key, array.astype('<f2').tobytes()))
    def get(self, key):
        row = self.db.execute('SELECT data FROM maps WHERE key=?', (key,)).fetchone()
        if row is None: raise KeyError(f'Missing semantic cache for image {key}')
        shape = (self.metadata['classes'], *self.metadata['grid_size'])
        arr = np.frombuffer(row[0], dtype='<f2')
        if arr.size != int(np.prod(shape)): raise ValueError(f'Invalid cached semantic map: {key}')
        return arr.reshape(shape).astype(np.float32)
    def commit(self): self.db.commit()
    def close(self): self.db.close()
