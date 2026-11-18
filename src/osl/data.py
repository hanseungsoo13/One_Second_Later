"""MOT17 adapter: sequence-separated, continuous one-second tracks."""
import configparser
import csv
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class MOT17Dataset(Dataset):
    def __init__(self, root, sequences, size=(384, 640), sample_fps=2, min_visibility=0.0):
        self.size = size
        self.records = []
        for name in sequences:
            seq = Path(root) / name
            cfg = configparser.ConfigParser(); cfg.read(seq / 'seqinfo.ini')
            info = cfg['Sequence']; fps = int(info['frameRate'])
            width, height = int(info['imWidth']), int(info['imHeight'])
            tracks = {}
            with (seq / 'gt/gt.txt').open() as f:
                for row in csv.reader(f):
                    frame, person = int(row[0]), int(row[1]); box = list(map(float, row[2:6])); vis = float(row[8])
                    if int(row[6]) != 1 or int(row[7]) != 1 or min(box[2:]) <= 0 or not all(map(math.isfinite, box + [vis])): continue
                    track = tracks.setdefault(person, {})
                    if frame in track: raise ValueError(f'duplicate GT: {name}/{person}/{frame}')
                    track[frame] = (box, vis)
            for person, track in tracks.items():
                for frame, (box, vis) in sorted(track.items()):
                    if round((frame-1)*sample_fps/fps) == round((frame-2)*sample_fps/fps): continue
                    end = frame + fps
                    if any(t not in track or track[t][1] < min_visibility for t in range(frame, end+1)): continue
                    future = track[end][0]; foot = [future[0]+future[2]/2, future[1]+future[3]]
                    # A probability distribution cannot represent points outside the image.
                    if not (0 <= foot[0] < width and 0 <= foot[1] < height): continue
                    path = seq / 'img1' / f'{frame:06d}.jpg'
                    if not path.is_file(): raise FileNotFoundError(path)
                    mid = track[frame + fps//2][0]
                    self.records.append((name, person, frame, path, box, foot, mid, width, height))
        if not self.records: raise ValueError('No valid samples; check paths, sequences and visibility')
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        name, person, frame, path, box, foot, mid, w, h = self.records[i]
        oh, ow = self.size; scale = min(ow/w, oh/h); nw, nh = round(w*scale), round(h*scale)
        px, py = (ow-nw)//2, (oh-nh)//2; sx, sy = nw/w, nh/h
        image = Image.open(path).convert('RGB').resize((nw,nh), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB',(ow,oh)); canvas.paste(image,(px,py))
        a = np.asarray(canvas, dtype=np.float32)/255
        a = (a - np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        x,y,bw,bh = box
        return {'image':torch.from_numpy(a.transpose(2,0,1).copy()), 'bbox':torch.tensor([x*sx+px,y*sy+py,(x+bw)*sx+px,(y+bh)*sy+py]),
                'target':torch.tensor([foot[0]*sx+px,foot[1]*sy+py]), 'original_target':torch.tensor(foot),
                'original_foot':torch.tensor([x+bw/2,y+bh]), 'height':torch.tensor(bh),
                'scale':torch.tensor([sx,sy]), 'offset':torch.tensor([px,py]), 'sequence':name, 'frame':frame,
                'person':person, 'path':str(path), 'mid':torch.tensor([mid[0]+mid[2]/2,mid[1]+mid[3]])}
