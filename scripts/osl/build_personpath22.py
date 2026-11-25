#!/usr/bin/env python3
"""Generate one-second person pairs and source frames for selected PersonPath22 videos."""
import argparse
import bisect, json, math
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2

DEFAULT_IDS = Path(__file__).resolve().parents[2] / 'data/personpath22_ground_view_ids.txt'
TOLERANCE_MS=100

def make_pairs(entities):
    tracks=defaultdict(list); seen=set()
    for e in entities:
        labels=e.get('labels',{}); box=e.get('bb',[]); frame=e.get('blob',{}).get('frame_idx')
        if not labels.get('person') or labels.get('fully_occluded') or frame is None or len(box)!=4 or not all(math.isfinite(float(x)) for x in box) or box[2]<=0 or box[3]<=0: continue
        key=(e['id'],int(frame))
        if key in seen: raise ValueError(f'Duplicate person/frame: {key}')
        seen.add(key); tracks[e['id']].append(e)
    pairs=[]
    for track in tracks.values():
        track.sort(key=lambda e:(e['time'],e['blob']['frame_idx']))
        times=[e['time'] for e in track]
        for e in track:
            desired=e['time']+1000; i=bisect.bisect_left(times,desired)
            candidates=[track[j] for j in (i-1,i) if 0<=j<len(track) and track[j]['blob']['frame_idx']>e['blob']['frame_idx']]
            if candidates:
                target=min(candidates,key=lambda x:(abs(x['time']-desired),x['time']))
                if abs(target['time']-desired)<=TOLERANCE_MS: pairs.append((e,target))
    return pairs

def process(row, output_root):
    name=row['video_id']; directory=output_root/name; frames_dir=directory/'images'; frames_dir.mkdir(parents=True,exist_ok=True)
    with open(row['amodal_annotation_path']) as f: ann=json.load(f)
    meta=ann['metadata']; width=int(meta['resolution']['width']); height=int(meta['resolution']['height']); length=int(meta['number_of_frames']); boundary=int(length*.8)
    pairs=make_pairs(ann['entities']); records={'train':[],'val':[]}; needed=set(); excluded=0
    for a,b in pairs:
        frame=int(a['blob']['frame_idx']); target=int(b['blob']['frame_idx'])
        if target < boundary: split='train'
        elif frame >= boundary: split='val'
        else: excluded+=1; continue
        x,y,w,h=map(float,a['bb']); fx,fy,fw,fh=map(float,b['bb']); ident=a['id']
        path=frames_dir/f'{frame:06d}.jpg'; target_path=frames_dir/f'{target:06d}.jpg'
        records[split].append({'sequence':name,'person_id':ident,'track_id':f'{name}:{ident}','frame':frame,'image_path':str(path),'image_size':[width,height],'bbox_xywh':[x,y,w,h],'target_frame':target,'target_image_path':str(target_path),'target_bbox_xywh':[fx,fy,fw,fh],'target_bottom_center_xy':[fx+fw/2,fy+fh],'time_delta_seconds':(b['time']-a['time'])/1000,'source_time_ms':a['time'],'target_time_ms':b['time']})
        needed.update((frame,target))
    cap=cv2.VideoCapture(row['video_path'])
    if not cap.isOpened(): raise RuntimeError(f'Cannot open {name}')
    requested=sorted(needed); next_idx=0; frame_idx=0
    while next_idx<len(requested):
        if not cap.grab(): break
        if frame_idx==requested[next_idx]:
            ok,img=cap.retrieve()
            if not ok or img.shape[:2]!=(height,width): raise RuntimeError(f'Cannot decode {name} frame {frame_idx} with expected size')
            if not cv2.imwrite(str(frames_dir/f'{frame_idx:06d}.jpg'),img,[cv2.IMWRITE_JPEG_QUALITY,90]): raise RuntimeError(f'Cannot write {name} frame {frame_idx}')
            next_idx+=1
        frame_idx+=1
    cap.release()
    if next_idx!=len(requested): raise RuntimeError(f'{name}: missing frames {requested[next_idx:next_idx+5]}')
    for split,items in records.items():
        items.sort(key=lambda x:(x['frame'],x['person_id']))
        with (directory/f'{split}.jsonl').open('w') as f:
            for item in items: f.write(json.dumps(item,separators=(',',':'))+'\n')
    return {'sequence':name,'source_video':row['video_path'],'annotation':row['amodal_annotation_path'],'fps':meta['fps'],'frames':length,'split_boundary_frame':boundary,'train':len(records['train']),'val':len(records['val']),'boundary_excluded':excluded,'images':len(needed),'train_person_ids':len({r['person_id'] for r in records['train']}),'val_person_ids':len({r['person_id'] for r in records['val']})}

def load_sequence_ids(path):
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]
    if len(ids) != len(set(ids)) or any(not item.startswith('uid_vid_') or not item[8:].isdigit() for item in ids):
        raise ValueError('Sequence IDs must be unique uid_vid_<number> values')
    return ids


def prepare_output(path):
    path.mkdir(parents=True, exist_ok=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-dir', type=Path, required=True, help='Directory containing uid_vid_*.mp4')
    parser.add_argument('--annotation-dir', type=Path, required=True, help='Directory containing amodal uid_vid_*.mp4.json')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; must not exist')
    parser.add_argument('--ids', type=Path, default=DEFAULT_IDS)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error('--workers must be positive')
    rows = []
    for name in load_sequence_ids(args.ids):
        video = (args.video_dir / f'{name}.mp4').resolve()
        annotation = (args.annotation_dir / f'{name}.mp4.json').resolve()
        for path in (video, annotation):
            if not path.is_file():
                raise FileNotFoundError(path)
        rows.append({'video_id': name, 'video_path': str(video), 'amodal_annotation_path': str(annotation)})
    output = args.output.resolve()
    prepare_output(output)
    cv2.setNumThreads(1)
    from functools import partial
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for result in pool.map(partial(process, output_root=output), rows):
            results.append(result)
            print(result['sequence'], result['train'], result['val'], result['images'], flush=True)
    (output / 'summary.json').write_text(json.dumps({'tolerance_ms': TOLERANCE_MS, 'sequences': results}, indent=2) + '\n')
    print('total:', {key: sum(row[key] for row in results) for key in ('train', 'val', 'images', 'boundary_excluded')}, flush=True)


if __name__ == '__main__':
    main()
