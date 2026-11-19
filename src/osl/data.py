"""Load the supplied per-sequence JSONL annotations without re-splitting."""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class AnnotationDataset(Dataset):
    def __init__(self, root, sequences, split, size=(384, 640), person_crop=None, sample_fps=None):
        if split not in ('train', 'val'):
            raise ValueError(f'Invalid split: {split}')
        if sample_fps is not None and (split != 'train' or sample_fps <= 0):
            raise ValueError('sample_fps must be positive and applies only to training')
        self.size = size
        self.person_crop = person_crop
        self.records = []
        for sequence in sequences:
            path = Path(root) / sequence / f'{split}.jsonl'
            with path.open() as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip(): continue
                    record = json.loads(line)
                    if record['sequence'] != sequence:
                        raise ValueError(f'{path}:{line_number}: unexpected sequence')
                    if sample_fps is not None:
                        seconds = record['time_delta_seconds']
                        fps = (record['target_frame'] - record['frame']) / seconds if seconds > 0 else 0
                        if fps < sample_fps:
                            raise ValueError(f'{path}:{line_number}: invalid time delta or sample FPS exceeds source FPS')
                        frame = record['frame']
                        if math.floor((frame - 1) * sample_fps / fps) == math.floor((frame - 2) * sample_fps / fps):
                            continue
                    self.records.append(record)
        if not self.records:
            raise ValueError(f'No records in {split} annotations')

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        w, h = record['image_size']; oh, ow = self.size
        scale = min(ow / w, oh / h); nw, nh = round(w * scale), round(h * scale)
        px, py = (ow - nw) // 2, (oh - nh) // 2; sx, sy = nw / w, nh / h
        image = Image.open(record['image_path']).convert('RGB')
        if image.size != (w, h): raise ValueError(f'Image size mismatch: {record["image_path"]}')
        person_tensor = None
        if self.person_crop is not None:
            x, y, bw, bh = record['bbox_xywh']; factor = self.person_crop['scale']
            cx, cy = x + bw / 2, y + bh / 2
            left, top = math.floor(cx-bw*factor/2), math.floor(cy-bh*factor/2)
            right, bottom = math.ceil(cx+bw*factor/2), math.ceil(cy+bh*factor/2)
            crop = image.crop((left, top, right, bottom))
            ch, cw = self.person_crop['size']; ratio = min(cw/crop.width, ch/crop.height)
            rw, rh = max(1, round(crop.width*ratio)), max(1, round(crop.height*ratio))
            crop = crop.resize((rw, rh), Image.Resampling.BILINEAR)
            crop_canvas = Image.new('RGB', (cw, ch)); crop_canvas.paste(crop, ((cw-rw)//2, (ch-rh)//2))
            person = np.asarray(crop_canvas, dtype=np.float32) / 255
            person = (person - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
            person_tensor = torch.from_numpy(person.transpose(2, 0, 1).copy())
        image = image.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (ow, oh)); canvas.paste(image, (px, py))
        a = np.asarray(canvas, dtype=np.float32) / 255
        a = (a - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        x, y, bw, bh = record['bbox_xywh']; tx, ty = record['target_bottom_center_xy']
        # The image heatmap cannot represent a point outside the image: train and
        # NLL at its nearest image boundary, but retain the true point for FDE.
        clipped_x = min(max(tx, 0), w - 1); clipped_y = min(max(ty, 0), h - 1)
        result = {'image': torch.from_numpy(a.transpose(2, 0, 1).copy()),
                'bbox': torch.tensor([x*sx+px, y*sy+py, (x+bw)*sx+px, (y+bh)*sy+py]),
                'target': torch.tensor([clipped_x*sx+px, clipped_y*sy+py]),
                'original_target': torch.tensor([tx, ty]),
                'original_foot': torch.tensor([x+bw/2, y+bh]), 'height': torch.tensor(bh),
                'scale': torch.tensor([sx, sy]), 'offset': torch.tensor([px, py]),
                'sequence': record['sequence'], 'frame': record['frame'],
                'person': record['person_id'], 'path': record['image_path'],
                'mid': torch.tensor([float('nan'), float('nan')])}
        if person_tensor is not None: result['person_crop'] = person_tensor
        return result
