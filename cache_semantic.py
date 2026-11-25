"""Cache frozen SegFormer probabilities once per distinct scene image."""
import argparse
from pathlib import Path
import sys
import yaml
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from osl.data import AnnotationDataset
from osl.segmentation import SegFormerPredictor, SegmentationCache, cache_metadata, image_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', required=True, choices=('train', 'val'))
    parser.add_argument('--max-images', type=int, help='Limit for smoke testing; use a separate cache')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    data, seg = config['data'], config['model']['semantic']
    if any(n % 16 for n in data['size']): raise ValueError('Scene size must be divisible by 16')
    grid = [n // 16 for n in data['size']]
    metadata = cache_metadata(seg['model'], seg['revision'], data['size'], grid)
    records = AnnotationDataset(data['root'], data[args.split], args.split).records
    unique = {image_key(r): r for r in records}
    if args.max_images is not None:
        if args.max_images <= 0: raise ValueError('--max-images must be positive')
        unique = dict(list(unique.items())[:args.max_images])
    cache = SegmentationCache(seg['cache'], metadata, writable=True)
    existing = cache.keys()
    pending = [(k, r) for k, r in unique.items() if k not in existing]
    print(f'{args.split}: {len(unique)} unique images, {len(pending)} to process', flush=True)
    if pending:
        predictor = SegFormerPredictor(seg['model'], seg['revision'], data['size'], grid, seg['device'])
        for i, (key, record) in enumerate(pending, 1):
            with Image.open(record['image_path']) as image:
                if list(image.size) != record['image_size']: raise ValueError(f'Image size mismatch: {record["image_path"]}')
                semantic = predictor(image)
            cache.put(key, semantic)
            if i % 10 == 0:
                cache.commit(); print(f'{args.split}: {i}/{len(pending)} new images', flush=True)
    cache.commit(); cache.close()
    print(f'{args.split}: cache complete at {seg["cache"]}', flush=True)

if __name__ == '__main__': main()
