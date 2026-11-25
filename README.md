# One Second Later

Single-image pedestrian future localization with a frozen DINOv3 encoder. E0 predicts the current foot point; E2-A uses spatial conditioning and a convolutional decoder; E2-B adds two self-attention layers. E2-C additionally encodes a 2× person/context crop from the original image with the same frozen DINOv3 backbone, pools its patch features, and appends a projected person token to the scene transformer. The token carries the foot-point positional encoding; the box spatial maps and decoder are unchanged. The crop is aspect-preserving letterboxed to 224×224 and can include black padding outside the image. Only the current image and bounding box are model inputs.

## Data

The default configurations directly load the user's existing `../MOT17_prediction/MOT17-{02,04,09}-FRCNN/{train,val}.jsonl` files. All records are used: 52,015 training and 11,999 validation examples. The files use a chronological 80:20 split within each sequence; train and validation share scenes and can share people. Validation measures later frames in the same scenes, not generalization to unseen scenes. Adjacent examples overlap heavily. No extra FPS sampling, visibility threshold or intermediate-track filtering is applied. The loader does not regenerate pairs from MOT17 GT.

The JSONL contains some future foot positions outside the source image (1,874 train and 418 validation records). These examples are kept. A finite image heatmap cannot represent an outside location, so the training target and NLL use the nearest point on the original image boundary. FDE uses the original, unclipped coordinate. Therefore NLL and the learning target for these examples measure the projected boundary location, while FDE measures the actual annotated location. Heatmap cell NLL uses floor after transforming to the heatmap grid. FDE and movement use original-image pixels and current bounding-box height.

The JSONL does not contain intermediate positions. Visualizations can show stationary cases and worst errors, but cannot infer straight or turning categories from these annotations alone. The old sequence-separated experiments in `outputs/e2a` and `outputs/e2b` used different samples and must not be compared as if they used these splits. New configurations write to `outputs/annotations-e2a` and `outputs/annotations-e2b`.

## Setup

The following uses the existing `flypose` environment, images at the absolute paths in the JSONL, and the DINOv3 ViT-B/16 checkpoint. Obtain the official implementation (excluded from this repository):

```bash
mkdir -p external
git clone https://github.com/facebookresearch/dinov3.git external/dinov3
git -C external/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
/root/workplace/miniconda3/envs/flypose/bin/pip install --target external/python-deps wandb
```

Set annotation root, sequences, batch size, output directory and W&B mode in the YAML files. W&B defaults to offline mode; authenticate and set `wandb.mode: online` for online tracking, or sync the offline run later. Do not put credentials in configuration or Git.

```bash
PY=/root/workplace/miniconda3/envs/flypose/bin/python
$PY -m unittest discover -s tests -v
$PY run.py e0 --config configs/e2a.yaml
$PY run.py overfit --config configs/e2a.yaml --steps 100
$PY run.py overfit --config configs/e2b.yaml --steps 100
$PY run.py train --config configs/e2a.yaml
$PY run.py train --config configs/e2b.yaml
$PY run.py overfit --config configs/e2c.yaml --steps 100
$PY run.py train --config configs/e2c.yaml
$PY run.py eval --config configs/e2b.yaml --checkpoint outputs/annotations-e2b/best.pt
$PY run.py visualize --config configs/e2b.yaml --checkpoint outputs/annotations-e2b/best.pt
```

E2-C uses the same annotation split, loss, decoder and optimizer settings as E2-B; crop size, context scale and output path are configurable in `configs/e2c.yaml`. Compare their respective validation metrics only after the runs complete.

Select a GPU with `CUDA_VISIBLE_DEVICES`. Training saves the lowest validation argmax pixel FDE as `best.pt`, and reports per-sequence and overall metrics. Choose a new output path in YAML for each training run. Letterbox applies the same transformation to the image, box and target. Visualizations show the box (green), current foot (cyan), original GT foot (yellow), argmax (red), and heatmap (red overlay). Points outside the image may not be visible.

## E2-C 2 FPS ablation

`configs/e2c_2fps.yaml` selects training start frames approximately 2 FPS from the same JSONL records, using FPS inferred from the annotated frame gap and time delta. All people at a selected frame are kept. Validation still uses all 11,999 original validation records; no annotation files are changed. All model and optimization settings are identical to `e2c.yaml`, except the separate output directory and training sample rate. Since epochs contain fewer training examples, 50 epochs also means fewer optimizer updates; comparisons must account for this difference.

```bash
$PY run.py train --config configs/e2c_2fps.yaml
```

## E2-C with pose

`configs/e2c_p1.yaml` and `configs/e2c_p2.yaml` share the same frozen
RTMPose-M cache and two-layer pose MLP. P1 appends pose and appearance as
separate tokens; P2 adds gated projected pose to appearance before the scene
transformer. Both tokens use the current foot-point position. DINOv3 remains
shared and frozen. The RTMPose-M model is the rtmlib balanced COCO 17-joint
model (192×256); the annotation box is passed directly to the top-down model.
Its predicted image coordinates are normalized relative to the original box;
coordinates can lie outside [0, 1], and each joint includes its confidence.

Install `rtmlib==0.0.16` and an ONNX Runtime backend in `flypose` for cache
generation. The default CPU inference downloads the official model on first
use. Each cache is resumable and keyed by sequence, frame, person, image path,
image size, and box. A missing or mismatched record fails instead of silently
substituting a pose. Cache files and model weights are generated artifacts.
Generate both splits once, then reuse them for both variants:

```bash
PY=/root/workplace/miniconda3/envs/flypose/bin/python
$PY -m pip install rtmlib==0.0.16
$PY cache_pose.py --config configs/e2c_p1.yaml --split train
$PY cache_pose.py --config configs/e2c_p1.yaml --split val
$PY run.py train --config configs/e2c_p1.yaml
$PY run.py train --config configs/e2c_p2.yaml
$PY run.py eval --config configs/e2c_p2.yaml --checkpoint outputs/annotations-e2c-p2/best.pt
$PY run.py eval --config configs/e2c_p2.yaml --checkpoint outputs/annotations-e2c-p2/best.pt --shuffle-pose
```

Shuffling uses a fixed seed across the validation dataset, preserving each
pose's joints and confidence together; compare against ordinary evaluation on
the same checkpoint. Both configurations stop after eight epochs without
improvement in overall validation argmax FDE, matching the existing checkpoint
selection criterion. Metrics include movement thresholds, zero-motion FDE,
improvement over zero-motion, and sequence breakdowns. The supplied split
shares scenes between training and validation; configure separate sequence
annotations and caches to measure unseen-scene generalization. Generating a
cache and training full experiments are separate steps and may take time.

## Historical PRW supervised calibration (not valid for the ID-free protocol)

**Do not use this historical experiment for the current ID-free PRW protocol:**
it trains and selects a scalar using PRW identity correspondences. The previously
reported 93.151% result came from this supervised experiment. Use the official
full-gallery section below instead.

This experiment combines the official CUCPS detector and ReID embeddings with
E2-C's future-position heatmap. The query is the annotated person box in the
previous image; candidates are CUCPS detections in the next available PRW image
from the same camera segment. Candidate scores are CUCPS's detection-weighted
appearance score plus a learned nonnegative scalar times mean max-normalized E2-C
heatmap probability around the candidate foot. No future ground-truth box or ID
is used to compute either score. The query box is ground truth (an oracle): this
is not yet a fully automatic video tracker or person-search system.

**Supervision:** The official PRW CUCPS checkpoint was learned without person ID
labels, but the scalar reranker here is **supervised**, using matching PRW person
IDs for train/validation. The E2-C checkpoint was trained using same-person
trajectory associations on PersonPath22. Therefore the complete pipeline should
not be described as ID-free weakly supervised person search.

The official CUCPS repository and the authors' PRW `args.json` and
`checkpoint_epoch18.pth` are expected under `external/` as in
`configs/temporal_prw.yaml`. CUCPS's official PRW pretrained checkpoint and
instructions are available from https://github.com/VIP-Lab-UNIST/CUCPS_official.
These external dependencies, generated datasets, and weights are not committed.

From this directory, prepare the dataset once and run the stages in order:

```bash
PY=/root/workplace/miniconda3/envs/flypose/bin/python
$PY prepare_prw_temporal.py --config configs/temporal_prw.yaml
$PY temporal_search.py extract --config configs/temporal_prw.yaml --split train
$PY temporal_search.py extract --config configs/temporal_prw.yaml --split val
$PY temporal_search.py fit --config configs/temporal_prw.yaml
$PY temporal_search.py extract --config configs/temporal_prw.yaml --split test
$PY temporal_search.py evaluate --config configs/temporal_prw.yaml --split test
```

Set `CUDA_VISIBLE_DEVICES` to the selected available GPU before extraction.
The YAML is the source of settings and paths; use a fresh `output_root` for a
new run. Extraction writes one candidate record per query and refuses to
overwrite existing output; interrupted output must be moved aside before a
fresh extraction. `fit` selects an epoch on validation Rank-1 and MRR, with
weight zero as an eligible baseline. Evaluation reports pre/post Rank-1, MRR,
and candidate recall including queries for which the detector missed the true
person. It also separates source frame gaps. These are next-frame matching
metrics, not CUCPS's standard full-gallery mAP or Rank-1.

Pairs use consecutive available PRW frames within the same camera segment and
both endpoints in the same split. The official PRW train/test split is retained;
PRW train frames are split chronologically within each camera segment into 80%
train and 20% validation, with cross-boundary pairs excluded. Scenes and people
can appear in both training and validation, and the pretrained CUCPS model has
seen the PRW train set, including the reranker's validation frames. The official
PRW test split is used only for final evaluation. Unknown IDs cannot be positive
pairs. Instances whose person is absent in the next frame are not included, so
this evaluates the conditional case only. Consecutive available PRW frames have
variable original frame-number gaps and unknown elapsed time; the E2-C model was
trained near one second. Compare performance by gap rather than interpreting
all predictions as a fixed-time horizon. GT candidate mode is an oracle ablation
(`cucps.candidate_mode: ground_truth`); it uses all valid PRW boxes as candidates
and sets their detection scores to 1, so its results are not detector results.

## PRW official full-gallery temporal experiment

The next-frame Rank-1 experiment above does **not** reproduce official Person
Search. For the full-gallery experiment, `official_gallery_search.py` uses the
PRW official query list, all official test gallery images, and the CUCPS regular
gallery protocol. For each official query, it adds the E2-C position score only
to detections in the immediately next **available PRW frame** from the same
camera and segment, when that frame is in the official test gallery. Other
candidate scores are untouched. The next-frame mapping uses filenames only;
future identity annotations do not decide whether to apply the score. Query
boxes are official ground-truth query boxes.

The inspected CUCPS checkout is pinned at
`d6f1cf2d727300f655bbb871e6c45f5eaf4e45fb`. `compare` calls the original
CUCPS PRW evaluator, independently evaluates the
same unmodified scores, and stops if baseline mAP/Rank-1/Rank-5/Rank-10 do not
match. It then reports reranked full-gallery metrics and frame-gap groups. The
regular CUCPS script uses `det_thresh=0.9`, `ignore_cam_id=True` (same-camera
images are included), and `remove_unlabel=True`. The last option uses PRW GT
to remove detections that do not match a labeled person, so these results
reproduce that GT-assisted benchmark protocol; they are not deployment metrics.

The model score is `CUCPS score + weight * location score`. The nonnegative
`fusion.weight` in `configs/official_gallery_prw.yaml` is specified **before**
evaluation and is not trained or tuned with PRW identities, future boxes, or
future locations. The example value `0.05` is illustrative, not an optimized or
validated choice. The legacy `temporal_search.py fit` and its ranker outputs
use PRW ID supervision and must not be used for this protocol. CUCPS uses its
authors' pretrained PRW checkpoint; E2-C was trained separately on PersonPath22
using person trajectory associations. PRW annotations are used by the official
GT-assisted CUCPS evaluation protocol, not to fit future-location prediction or
fusion weight. Actual time between PRW images is unknown.

```bash
PY=/root/workplace/miniconda3/envs/flypose/bin/python
# From One_Second_Later. Choose the fixed fusion.weight before evaluating PRW.
# Replace GPU_NUMBER with an available GPU index or UUID.
CUDA_VISIBLE_DEVICES=GPU_NUMBER $PY -u official_gallery_search.py run --config configs/official_gallery_prw.yaml

# Alternatively run the two stages separately:
CUDA_VISIBLE_DEVICES=GPU_NUMBER $PY -u official_gallery_search.py extract --config configs/official_gallery_prw.yaml
$PY -u official_gallery_search.py compare --config configs/official_gallery_prw.yaml
```

`extract` uses a GPU selected via `CUDA_VISIBLE_DEVICES`; comparison uses the
CPU. All paths and settings are in the YAML; choose a fresh `output_root` for
each run. Extraction writes `official_features.pkl` only after completing all
images and queries; it does not resume a partially finished extraction. This
file is a Python pickle and must only be loaded from a trusted local run.
`run` reuses a completed feature cache if present; partial extraction cannot
be resumed. `compare` writes original, baseline, reranked, per-query and gap-specific
results to `official_gallery_metrics.json`. Baseline parity with the original
official function is checked on every comparison, rather than assumed.

## CUCPS temporal pseudo-ID clustering experiment

`cucps_temporal_experiment.py` trains the official CUCPS PRW model twice from
the same seed and settings. The baseline uses the original HNM/HPM behavior.
The temporal run adds a fixed E2-C location score to the similarity used by
HNM, before its per-image winner selection and cycle-consistency check. The
score is added only when the candidate belongs to the next available training
image in the same PRW camera segment; all other CUCPS image-pair similarities
are unchanged. HPM continues to use CUCPS's co-appearance calculation.

E2-C priors are computed from the previous frame image and its annotated person
box, and scored at each candidate box's bottom center in the next frame. Prior
construction reads boxes and frame names only; it does not read PRW identity
labels. CUCPS continues to build its own pseudo IDs from its per-instance
memory and HNM/HPM. The motion checkpoint is the existing PersonPath22-trained
E2-C model. The fixed `temporal_prior.weight: 0.05` is not fitted on PRW.
Consecutive available PRW images can have different frame-number gaps, so the
time horizon is not constant. The current PRW train list has a median gap of 50
frame numbers, a 90th-percentile gap of 375, and a maximum gap of 3,900; the
runner records the actual gap summary for each run. The experiment follows the
requested next-available-frame rule even for large gaps, so E2-C's prior may be
less reliable for those pairs.

Training uses the official `train-prw.sh` settings in the YAML. Each run gets a
new timestamped directory under `../PRW_cucps_temporal/runs/{baseline,temporal}`
and prints its checkpoint path. Temporal heatmaps are cached outside this
repository in `../PRW_cucps_temporal/cache/`. The cache contains no identity
labels. Both checkpoints can be evaluated with CUCPS's official regular PRW
full-gallery evaluator (`det_thresh=0.9`, `ignore_cam_id=True`,
`remove_unlabel=True`), then compared with the `compare` action. As with the
official CUCPS protocol, `remove_unlabel=True` uses PRW test annotations during
evaluation; this is benchmark evaluation, not training supervision.

Run the baseline first, then the temporal variant on the same selected GPU.
Run these commands directly in a terminal; training logs remain visible there.

```bash
cd /root/workplace/assignment/principle_of_deeplearning/One_Second_Later
PY=/root/workplace/miniconda3/envs/flypose/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -u cucps_temporal_experiment.py train --config configs/cucps_temporal_prw.yaml --variant baseline
```

After the baseline finishes, start the temporal run:

```bash
CUDA_VISIBLE_DEVICES=0 $PY -u cucps_temporal_experiment.py train --config configs/cucps_temporal_prw.yaml --variant temporal
```

Copy each printed run directory into the evaluation commands below, replacing
`RUN_DIR` with its absolute path. Evaluate both final checkpoints and compare
the resulting official metrics:

```bash
$PY cucps_temporal_experiment.py test --config configs/cucps_temporal_prw.yaml --checkpoint RUN_DIR/checkpoint_epoch_last.pth
$PY cucps_temporal_experiment.py compare \
  --baseline-checkpoint BASELINE_RUN_DIR/checkpoint_epoch_last.pth \
  --temporal-checkpoint TEMPORAL_RUN_DIR/checkpoint_epoch_last.pth \
  --output ../PRW_cucps_temporal/comparison.json
```

The comparison reports mAP and Rank-1 changes in percentage points. Do not
compare this result to the earlier post-search reranking experiment as though
the clustering stage were unchanged: this experiment trains CUCPS with the
temporal prior inside HNM.

### Temporal graph positive propagation

The separate `graph` variant keeps the original CUCPS HNM/HPM result, then
extends it through an undirected temporal graph before ReID loss. It does not
change inference. Each edge combines the current memory cosine similarity and
the frozen predictor's cached next-frame location score. Edges require mutual
best matches, appearance and combined thresholds, and a top-two margin in
both directions. The cache rejects pairs beyond `max_frame_gap` and predictions
whose softmax heatmap peak is below `min_heatmap_peak`. These are frame-number
and concentration filters, not calibrated time or prediction confidence.

After CUCPS warm-up, the graph is rebuilt on every clustering call using current
memory features. CUCPS mining is also run for temporal neighbors of the batch;
maximum products of edge confidences and `alpha` expand both ends of each
positive relation up to `hops` per side. Original batch positives are retained;
new candidates must meet path and appearance thresholds and cannot occupy a
scene already represented in the original or newly accepted positives. The
implementation uses sparse adjacency lists; it does not form an N-by-N graph.
The extra mining and graph construction add training cost. Thresholds in the
example YAML are illustrative and are not validated performance claims.

```bash
PY=/root/workplace/miniconda3/envs/flypose/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -u cucps_temporal_experiment.py train --config configs/cucps_temporal_graph_prw.yaml --variant graph
```

Choose a GPU and thresholds before running, and use a separate cache when
changing predictor, heatmap, or graph-cache settings. The cache contains box
locations and frame metadata, not PRW identity labels. The predictor was trained
on PersonPath22 trajectories, so it adds external trajectory supervision.
PRW frame-number gaps are not a verified time interval; check the predictor's
one-second horizon against the data before interpreting results. Compare with a
same-seed baseline and measure cross-camera positive precision/recall and
negative contamination separately; the current evaluator does not compute
those diagnostic metrics or an oracle graph.
