# One Second Later

Single-image pedestrian future localization with a frozen DINOv3 encoder. E0 predicts the current foot point; E2-A adds spatially conditioned convolutional decoding; E2-B adds two self-attention layers. Model inputs are only the current image and current bounding box. Future and intermediate GT are used solely for targets, metrics, and visualization categories.

## Setup on this machine

The commands below use the existing `flypose` environment, MOT17 images and the local DINOv3 ViT-B/16 checkpoint. Obtain the official DINOv3 implementation separately (it is excluded from this repository):

```bash
mkdir -p external
git clone https://github.com/facebookresearch/dinov3.git external/dinov3
git -C external/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
/root/workplace/miniconda3/envs/flypose/bin/pip install --target external/python-deps wandb
```

The checkpoint path, dataset root, splits, batch size, output directory and logging are in `configs/e2a.yaml` and `configs/e2b.yaml`. W&B defaults to offline mode because no API key was available. Authenticate with W&B (`WANDB_API_KEY` or `wandb login`) and set `wandb.mode: online` for live logging, or run `wandb sync` on the generated offline run after authentication. The supplied inference API example is unrelated to experiment-metric tracking; this implementation uses the W&B Python SDK and does not call a language model. Keep credentials out of configuration and Git.

```bash
PY=/root/workplace/miniconda3/envs/flypose/bin/python
$PY -m unittest discover -s tests -v
$PY run.py e0 --config configs/e2a.yaml
$PY run.py overfit --config configs/e2a.yaml --steps 100
$PY run.py overfit --config configs/e2b.yaml --steps 100
$PY run.py train --config configs/e2a.yaml
$PY run.py train --config configs/e2b.yaml
$PY run.py eval --config configs/e2b.yaml --checkpoint outputs/e2b/best.pt
$PY run.py visualize --config configs/e2b.yaml --checkpoint outputs/e2b/best.pt
```

Select a GPU with `CUDA_VISIBLE_DEVICES`. Training saves the lowest validation argmax pixel FDE as `best.pt` and prints overall and per-sequence metrics. Run E2-A and E2-B with the same seed and settings and compare the moving-subset metrics. For leave-one-sequence-out, change `data.train` and `data.val` in a new YAML; they must be disjoint. Fixed and moving MOT17 cameras are prohibited from being combined in one split.

## Data and coordinates

The existing `MOT17_prediction/create_annotations.py` and its JSONL files use MOT17 GT, marked pedestrian class 1 and positive boxes at endpoints. This adapter reads the same original `seqinfo.ini` and `gt/gt.txt` to reconstruct those pairs including the original time-split boundary pairs, then adds configurable visibility, requires every intermediate GT frame, samples approximately 2 start frames per second, and excludes future feet outside the image. It does not read the JSONL files directly. The default sequence split is 02+04 train and 09 validation, not the original within-sequence time split. No frame from the validation scene is used for training. `min_visibility` is applied throughout the track; its default 0.0 retains all nonnegative visibility values. Change it in YAML when needed.

Letterbox uses the same scale and offsets for RGB, boxes and points. Heatmap cells represent their centers; NLL uses the cell containing the continuous transformed GT (`floor` and clamp). FDE is measured in original image pixels; height-normalized FDE and movement use the current original bounding-box height. Reported zero-motion improvements are baseline minus model FDE in pixels. All thresholds are evaluated inclusively. The visualizations show bbox (green), current foot (cyan), GT foot (yellow), predicted argmax (red), and heatmap (red overlay). Turning uses the angle between current-to-midpoint and midpoint-to-future GT segments; it is only an approximate inspection label, never model input. Categories without examples do not produce an image.

For another dataset, implement a PyTorch Dataset that returns the fields consumed by `run.py` and `osl.evaluate`; for another backbone implement `channels` and a spatial feature `forward`. The DINO backbone remains frozen and uses patch tokens only. Checkpoints and experiment outputs are intentionally not tracked.
