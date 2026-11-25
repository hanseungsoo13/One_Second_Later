"""Cache frozen RTMPose-M predictions for annotation bounding boxes."""
import argparse
import sqlite3
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from osl.data import AnnotationDataset
from osl.pose import pose_key, relative_pose

DEFAULT_MODEL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--max-samples', type=int, help='Limit for smoke tests; use a separate cache path')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    pcfg = cfg['model']['pose']; data = cfg['data']
    records = AnnotationDataset(data['root'], data[args.split], args.split).records
    if args.max_samples is not None: records = records[:args.max_samples]
    path = Path(pcfg['cache'][args.split]); path.parent.mkdir(parents=True, exist_ok=True)
    from rtmlib import RTMPose
    model = RTMPose(pcfg.get('model', DEFAULT_MODEL), model_input_size=(192, 256),
                    backend='onnxruntime', device=pcfg.get('device', 'cpu'))
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS poses (key TEXT PRIMARY KEY, pose BLOB NOT NULL)')
        identity = pcfg.get('model', DEFAULT_MODEL)
        existing = db.execute("SELECT value FROM metadata WHERE name='model'").fetchone()
        if existing and existing[0] != identity: raise ValueError('Cache belongs to a different pose model')
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('model', ?)", (identity,))
        for i, record in enumerate(records):
            key = pose_key(record)
            if db.execute('SELECT 1 FROM poses WHERE key=?', (key,)).fetchone(): continue
            image = np.asarray(Image.open(record['image_path']).convert('RGB'))[:, :, ::-1].copy()
            x, y, w, h = record['bbox_xywh']
            if w <= 0 or h <= 0: raise ValueError(f'Invalid bbox in {key}')
            points, scores = model(image, [[x, y, x+w, y+h]])
            pose = relative_pose(points[0], scores[0], record['bbox_xywh'])
            db.execute('INSERT INTO poses VALUES (?, ?)', (key, pose.astype('<f4').tobytes()))
            if (i+1) % 100 == 0:
                db.commit(); print(f'{args.split}: {i+1}/{len(records)}', flush=True)
        db.commit()
    print(f'{args.split}: cached {len(records)} records in {path}')

if __name__ == '__main__': main()
