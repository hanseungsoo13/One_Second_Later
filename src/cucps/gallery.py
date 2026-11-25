"""CUCPS-compatible PRW full-gallery score fusion and evaluation."""
import re
from collections import defaultdict
import numpy as np
from sklearn.metrics import average_precision_score

def fixed_fusion_weight(cfg):
    """Use only the prespecified weight; never load PRW-trained ranker artifacts."""
    try:
        value=float(cfg['fusion']['weight'])
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('fusion.weight must be explicitly specified') from exc
    if not np.isfinite(value) or value<0:raise ValueError('fusion.weight must be finite and nonnegative')
    return value

def next_frame_map(all_names, gallery_names):
    """Next available image within a camera segment, if it belongs to gallery."""
    groups=defaultdict(list); gallery_names=set(gallery_names)
    for name in all_names:
        m=re.fullmatch(r'(c\d+s\d+)_(\d+)\.jpg',name)
        if not m:raise ValueError(f'Invalid PRW image name: {name}')
        groups[m[1]].append((int(m[2]),name))
    result={}
    for group in groups.values():
        group.sort()
        for a,b in zip(group,group[1:]):
            if b[1] in gallery_names:result[a[1]]=b[1]
    return result

def iou_matrix(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
    lt=np.maximum(a[:,None,:2],b[None,:,:2]);rb=np.minimum(a[:,None,2:],b[None,:,2:]);wh=np.maximum(rb-lt,0);inter=wh[:,:,0]*wh[:,:,1]
    area_a=np.maximum(a[:,2]-a[:,0],0)*np.maximum(a[:,3]-a[:,1],0);area_b=np.maximum(b[:,2]-b[:,0],0)*np.maximum(b[:,3]-b[:,1],0)
    union=area_a[:,None]+area_b[None,:]-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)

def prepare_gallery(records,detections,features,threshold=.9,remove_unlabel=True):
    if not (len(records)==len(detections)==len(features)):raise ValueError('gallery inputs differ in length')
    result={}
    for gt,det,feat in zip(records,detections,features):
        det=np.asarray(det).reshape(-1,5);feat=np.asarray(feat)
        if len(det)!=len(feat):raise ValueError('candidate features not aligned with detections')
        inds=np.flatnonzero(det[:,4]>=threshold)
        if len(inds) and remove_unlabel:
            ious=iou_matrix(gt['boxes'],det[:,:4]).astype(np.float32)
            if not len(ious):inds=inds[:0]
            else:
                labels=np.asarray(gt['gt_pids'])[ious.argmax(axis=0)]
                labels[ious.max(axis=0)<.5]=-2
                inds=inds[labels[inds]!=-2]
        result[gt['im_name']]={'det':det[inds],'features':feat[inds],'indices':inds,'original_count':len(det)}
    return result

def evaluate_queries(records,cache,queries,probe_features,biases,weight,ignore_cam_id=True):
    if len(queries)!=len(probe_features):raise ValueError('probe lengths differ')
    aps=[];acc=[];details=[]
    for qi,(query,probe) in enumerate(zip(queries,probe_features)):
        pid=int(np.asarray(query['gt_pids']).reshape(-1)[0]); probe=np.asarray(probe).ravel();gt={}
        for item in records:
            if item['im_name']!=query['im_name'] and pid in item['gt_pids']:
                matches=np.asarray(item['boxes'])[np.asarray(item['gt_pids'])==pid]
                # The upstream evaluator flattens matches and uses the first box.
                gt[item['im_name']]=matches.reshape(-1)[:4]
        true=[];scores=[];count_gt=count_tp=0
        for item in records:
            name=item['im_name']
            if name==query['im_name'] or (not ignore_cam_id and item['cam_id']==query['cam_id']):continue
            count_gt+=int(name in gt); cached=cache[name];det=cached['det'];feat=cached['features']
            if not len(det):continue
            sim=feat.reshape(len(feat),-1).dot(probe).ravel()
            if qi in biases and biases[qi][0]==name:
                loc=np.asarray(biases[qi][1]);
                if len(loc)!=cached['original_count']:raise ValueError('Bias values must match original candidate count')
                sim=sim+float(weight)*loc[cached['indices']]
            labels=np.zeros(len(sim),dtype=np.int32)
            if name in gt:
                box=gt[name];w,h=box[2]-box[0],box[3]-box[1];threshold=min(.5,(w*h)/((w+10)*(h+10)))
                inds=np.argsort(sim)[::-1];sim=sim[inds];det=det[inds];ious=iou_matrix(det[:,:4],box[None,:]).ravel();hit=np.flatnonzero(ious>=threshold)
                if len(hit):labels[hit[0]]=1;count_tp+=1
            true.extend(labels.tolist());scores.extend(sim.tolist())
        if not count_gt:raise ValueError(f'Query {qi} has no GT matches in selected gallery')
        y=np.asarray(true);s=np.asarray(scores);recall=count_tp/count_gt
        ap=0. if count_tp==0 else float(average_precision_score(y,s)*recall)
        order=np.argsort(s)[::-1];top=[int(bool(y[order[:k]].sum())) for k in (1,5,10)];aps.append(ap);acc.append(top);details.append({'query_index':qi,'ap':ap,'top1':top[0],'gt':count_gt,'detected_gt':count_tp})
    if not aps:raise ValueError('No queries')
    return {'mAP':float(np.mean(aps)),'top_k':np.mean(acc,axis=0).tolist(),'queries':len(aps),'details':details}
