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
