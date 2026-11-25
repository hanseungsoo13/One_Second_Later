# One Second Later

PersonPath22에서 사람의 다음 위치를 히트맵으로 예측하는 **OSL(One Second Later)** 모델을 학습하고, 이 모델의 위치 점수를 PRW의 CUCPS pseudo-ID 학습에 사용하는 방법입니다. 아래 명령은 모두 `One_Second_Later` 디렉터리에서 실행합니다.

OSL은 현재 이미지, 사람 bbox, 사람 주변 crop과 RTMPose 관절을 입력받습니다. Pose 특징을 사람 외형 특징에 gated fusion으로 결합하고 다음 위치 히트맵을 출력합니다. PRW에서는 OSL을 고정한 채 CUCPS를 학습합니다. 기본 OSL 모델 이름은 `full_bbox_pose2`입니다. 이전 이름도 호환용으로 사용할 수 있습니다.

## 저장소 구조

- `src/osl/`, `scripts/osl/`, `configs/osl/`: 위치 예측 데이터 준비, 캐시, 모델 학습과 평가.
- `src/cucps/`, `scripts/cucps/`, `configs/cucps/`: CUCPS 학습, 공식 PRW 평가와 결과 비교.
- `src/integration/`: OSL 위치 점수를 CUCPS 학습의 prior로 변환·결합.
- `experiments/`, `configs/experiments/`: PRW 위치 예측 보조 분석과 검색 시점 재정렬 실험.
- `tests/`: 모델, 데이터, 캐시 및 기존·새 실행 경로의 회귀 테스트.

경로는 모두 저장소 루트를 작업 디렉터리로 실행하는 기준입니다. 이전 최상위 실행 경로 대신 위 폴더의 명령을 사용하세요.

## 0. 설치 및 데이터 준비

아래는 Linux와 NVIDIA GPU를 사용하는 예시입니다. Git, Conda, NVIDIA 드라이버와 충분한 저장 공간이 필요합니다. 코드는 Git으로 받지만 **데이터, 전처리 결과, 모델 가중치와 학습 결과는 별도**로 준비해야 합니다.

### 프로젝트와 가상환경

```bash
git clone https://github.com/hanseungsoo13/One_Second_Later.git
cd One_Second_Later

conda create -n osl python=3.11 -y
conda activate osl
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt onnxruntime
```

PyTorch 설치 예시는 CUDA 12.4용입니다. 환경에 맞지 않으면 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/previous-versions/)에서 맞는 torch/torchvision 조합을 선택하세요. 기존 실험 환경은 Python 3.11, torch 2.6.0+cu124, torchvision 0.21.0+cu124였습니다. 위 명령으로 만든 새 환경 전체를 이번에 검증한 것은 아닙니다.

```bash
python -c "import torch, torchvision, onnxruntime; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); print('ONNX providers:', onnxruntime.get_available_providers())"
```

`CUDA: True`인지 확인하세요. `onnxruntime`은 CPU용이므로 pose 추정이 느릴 수 있습니다. GPU용 ONNX Runtime은 드라이버 및 CUDA 호환성을 확인해 [공식 안내](https://onnxruntime.ai/docs/install/)에 따라 별도로 설치하세요. `CUDA_VISIBLE_DEVICES=2`는 물리 GPU 2번을 선택하는 예시이므로 실제 사용 가능한 번호로 바꿉니다.

### DINOv3와 CUCPS

```bash
mkdir -p external
git clone https://github.com/facebookresearch/dinov3.git external/dinov3
git -C external/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
git clone https://github.com/VIP-Lab-UNIST/CUCPS_official.git external/CUCPS_official
git -C external/CUCPS_official checkout d6f1cf2d727300f655bbb871e6c45f5eaf4e45fb
```

[DINOv3 공식 안내](https://github.com/facebookresearch/dinov3)에 따라 **ViT-B/16 LVD-1689M** 가중치 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`를 받아 사용합니다. 가중치는 라이선스 동의와 접근 승인이 필요할 수 있습니다. 받은 경로를 PersonPath22 YAML의 `model.weights`와 PRW YAML의 `motion.weights`에 각각 지정하세요. 현재 YAML의 `/root/.cache/...`는 기존 서버의 경로입니다.

OSL용 [RTMPose](https://github.com/Tau-J/rtmlib) 모델은 `scripts/osl/cache_pose.py`가 처음 실행될 때 다운로드합니다. CUCPS 학습은 torchvision의 사전학습 ResNet-50 백본을 사용하므로 처음 실행할 때 해당 가중치 다운로드가 필요할 수 있습니다.

### PersonPath22 다운로드와 전처리

1. [PersonPath22 공식 저장소](https://github.com/amazon-science/tracking-dataset)의 다운로드 안내를 따라 영상과 annotation을 받습니다. 공식 다운로드 스크립트는 AWS CLI와 `ffmpeg`/`ffprobe`가 필요하며, 일부 외부 데이터는 별도 안내가 있습니다.
2. 저장소에 포함된 전처리 스크립트와 68개 ground-view 영상 ID 목록을 사용합니다. 각 영상의 앞 80%는 학습, 뒤 20%는 검증에 사용하며 경계를 넘는 쌍은 제외합니다. 같은 장면을 두 분할이 공유합니다.
3. 다운로드한 원본 영상과 amodal annotation 위치, 아직 존재하지 않는 출력 경로를 지정합니다. 기존 디렉터리는 덮어쓰지 않습니다.

```bash
python scripts/osl/build_personpath22.py \
  --video-dir /path/to/raw_data \
  --annotation-dir /path/to/anno_amodal_2022 \
  --output /path/to/personpath22_processed
```

목록은 `data/personpath22_ground_view_ids.txt`입니다. 수동 선택은 영상별 미리보기 한 프레임에 기반하므로 영상 전체의 시점이 일정함을 보장하지 않습니다. 생성 JSONL의 이미지 경로는 절대경로이므로 결과를 옮기면 경로도 수정해야 합니다. YAML의 `data.root`를 출력 위치로 설정하세요. 결과 구조는 다음과 같습니다.

```text
../PersonPath22_ground_view/
  uid_vid_00001/
    train.jsonl
    val.jsonl
    images/
  uid_vid_00002/...
```

각 JSONL에는 `sequence`, `person_id`, `frame`, `image_path`, `image_size`, `bbox_xywh`, `target_bottom_center_xy` 등이 필요합니다. `image_path`는 실행 환경에서 읽을 수 있어야 합니다. 기존 데이터는 amodal bbox를 사용했고, 같은 사람의 약 1초 후 annotation이 있는 쌍을 선택했습니다. 자세한 필드 처리는 `src/osl/data.py`를 참고하세요.

### PRW 다운로드와 체크포인트

[PRW 공식 페이지](https://zheng-lab-anu.github.io/Project/project_prw.html)의 v2016.04.20 데이터셋을 내려받아 압축을 풉니다. `frames/`, `annotations/`, `frame_train.mat`, `frame_test.mat`, `query_info.txt`가 같은 PRW 루트 아래에 있어야 합니다. 이 경로를 PRW 학습 YAML의 `prw.root`에 지정하세요.

PRW OSL 학습에는 사용할 OSL 모델의 체크포인트가 필요합니다. pose 모델은 `motion.pose_checkpoint`에 지정하고, pose가 없는 모델은 `motion.checkpoint`에 지정합니다. 기존 temporal-prior 캐시는 메타데이터 형식이 바뀌었으므로 새 캐시 경로를 지정해 다시 생성하세요.

## 1. PersonPath22 OSL 위치 예측 학습

`configs/osl/personpath22_pose2.yaml`은 한 개 시퀀스를 사용하는 공유용 예제입니다. 데이터·가중치 경로와 시퀀스 목록을 환경에 맞게 수정하고, 새 실험마다 `training.output`을 바꾸세요. 기존 68개 시퀀스 설정은 `configs/personpath22_osl.yaml`에 보존했습니다.

시퀀스별 train/val 분할을 사용하므로 두 분할이 같은 장면을 공유합니다. Pose 캐시를 생성한 후 OSL을 학습합니다. `CUDA_VISIBLE_DEVICES`는 사용할 GPU 번호로 바꾸세요.

```bash
python scripts/osl/cache_pose.py --config configs/osl/personpath22_pose2.yaml --split train
python scripts/osl/cache_pose.py --config configs/osl/personpath22_pose2.yaml --split val
CUDA_VISIBLE_DEVICES=2 python scripts/osl/train.py --config configs/osl/personpath22_pose2.yaml
```

캐시는 재실행 시 기존 레코드를 재사용합니다. 학습은 검증 FDE가 가장 낮은 모델을 `outputs/example/osl/best.pt`에 저장합니다. 기존 출력 디렉터리에 학습 로그가 있으면 덮어쓰지 않으므로 설정에서 새 경로를 지정하세요.

## 2. PRW OSL 기반 CUCPS 학습

`configs/cucps/prw_pose2.yaml`은 공유용 예제입니다. `motion.pose_checkpoint`에 위 학습의 `best.pt`를 지정하고 별도의 prior 캐시와 출력 경로를 설정하세요. 데이터·체크포인트 경로를 확인하세요. 다른 OSL 체크포인트를 사용할 때는 `motion.pose_checkpoint`와 `temporal_prior.cache`를 함께 변경하세요.

이 설정에서 OSL 가중치는 `motion.pose_checkpoint`에서 불러옵니다. 새 체크포인트를 사용하면 별도의 prior 캐시 경로를 지정하세요.

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/cucps/train.py \
  --config configs/cucps/prw_pose2.yaml --variant full_bbox_pose2
```

설정상 CUCPS를 25에폭 학습하며, timestamp가 붙은 실행 디렉터리에 체크포인트가 저장됩니다. 시작 시 PRW 학습 이미지의 bbox로 OSL 히트맵을 만들고 다음에 존재하는 같은 카메라·장면 이미지의 후보 bbox마다 위치 점수를 캐시합니다. 이 점수는 CUCPS 학습 중 HNM/HPM의 pseudo-ID 후보 선택에 사용됩니다. PRW의 사람 ID 정답으로 OSL을 학습하거나 위치 점수를 맞추지 않습니다. PRW 이미지의 프레임 번호 간격은 일정하지 않습니다.

OSL은 **CUCPS 학습 중에만** 보조 단서로 사용됩니다. 저장된 CUCPS 체크포인트로 검색할 때 히트맵 점수를 별도로 더하지는 않습니다.

동일한 CUCPS 설정에서 `--variant baseline`으로 OSL prior를 사용하지 않는 기준 모델도 학습할 수 있습니다. 학습된 체크포인트는 다음 명령으로 공식 PRW 검색 평가를 실행합니다.

```bash
python scripts/cucps/evaluate.py --config configs/cucps/prw_pose2.yaml \
  --checkpoint /path/to/cucps/checkpoint.pth
```

기준 모델과 OSL을 사용한 모델을 각각 평가한 뒤 아래 결과 비교 명령을 사용합니다.

## 기존 체크포인트에 대한 주의

이전 PRW 실험에서 사용한 `outputs/pose-runs/20261123-0228/p2/best.pt`에는 학습 데이터가 `../MOT17_prediction`으로 기록돼 있습니다. 따라서 이 체크포인트를 PersonPath22에서 학습한 OSL로 간주하면 안 됩니다. `configs/cucps_temporal_prw_pose_p2.yaml`은 기본적으로 이 기존 체크포인트를 가리킵니다. PersonPath22에서 학습한 OSL을 쓰려면 위처럼 `motion.pose_checkpoint`를 변경해야 합니다.

## 3. OSL + 장면 분할 학습

`configs/osl/personpath22_pose_seg.yaml`은 OSL에 SegFormer-B2의 150개 장면 분할 확률 특징을 추가합니다. 1번 단계의 같은 PersonPath22 데이터와 pose 캐시를 사용하며, SegFormer는 학습 전 이미지별 semantic 캐시를 생성할 때만 사용합니다. 첫 실행에는 Hugging Face의 모델 가중치 다운로드가 필요합니다.

```bash
CUDA_VISIBLE_DEVICES=8 python scripts/osl/cache_semantic.py --config configs/osl/personpath22_pose_seg.yaml --split train
CUDA_VISIBLE_DEVICES=8 python scripts/osl/cache_semantic.py --config configs/osl/personpath22_pose_seg.yaml --split val
CUDA_VISIBLE_DEVICES=8 python scripts/osl/train.py --config configs/osl/personpath22_pose_seg.yaml
```

pose 및 semantic train/val 캐시가 모두 준비돼 있어야 학습할 수 있습니다. Semantic 캐시는 이미지별로 저장되며 재실행 시 기존 항목을 재사용합니다. 예제 출력은 `outputs/example/osl_semantic`에 저장됩니다. 데이터와 가중치 경로 및 GPU 번호는 환경에 맞게 변경하세요.

코드의 입력 형식과 학습·검색 경로는 [코드와 데이터 흐름](docs/architecture.md)을 참고하세요.

## 모델과 실험 이름

| 모델 이름 | 이전 이름 | 구성 |
| --- | --- | --- |
| `full_bbox_a` | `e2a` | 장면과 bbox |
| `full_bbox_b` | `e2b` | A + Transformer |
| `full_bbox_c` | `e2c` | B + 사람 crop |
| `full_bbox_pose1` | `e2c_p1` | C + 별도 pose 토큰 |
| `full_bbox_pose2` | `e2c_p2` | C + gated pose fusion |
| `full_bbox_pose_seg` | `e2c_p2_s` | pose2 + SegFormer 장면 분할 |

OSL YAML의 `model.variant`와 CUCPS 학습의 `--variant`에 위 이름을 사용합니다. 기존 `e2*` 이름은 OSL 설정에서, `temporal`, `pose_p1`, `pose_p2`는 CUCPS에서 이전 실행과의 호환성을 위해 유지합니다. CUCPS `baseline`은 OSL 위치 점수를 사용하지 않는 기준 학습이며 `graph`는 별도의 temporal graph 방식입니다.

CUCPS에서 pose 모델은 `motion.pose_checkpoint`, 나머지는 `motion.checkpoint`에 해당 모델의 체크포인트를 지정합니다. `full_bbox_pose_seg`는 추가로 다음 설정이 필요하며 prior 생성 시 이미지마다 SegFormer를 실행합니다.

```yaml
motion:
  semantic:
    model: nvidia/segformer-b2-finetuned-ade-512-512
    revision: de01bae28967510f9ddd496c60a969357195400c
    device: cuda
```

공식 평가를 마친 체크포인트는 새 이름을 지정해 비교할 수 있습니다. 기존 `pose_p2`로 기록된 체크포인트도 `full_bbox_pose2`의 이전 이름으로 인식합니다.

```bash
python scripts/cucps/compare.py \
  --baseline-checkpoint /path/to/baseline/checkpoint.pth \
  --model-checkpoint full_bbox_pose2=/path/to/model/checkpoint.pth \
  --output /path/to/comparison.json
```

기존 설정의 체크포인트, 데이터, 캐시 및 출력 경로는 예전 실행 환경의 값입니다. 실행 전에 YAML에서 모두 확인하고 새 실험에는 새 출력과 캐시 경로를 지정하세요. prior 캐시는 모델과 설정이 다르면 재사용하지 않아야 합니다.

## 테스트

프로젝트 의존성이 설치된 Python 환경에서 다음 명령을 실행합니다.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

전체 학습과 실제 데이터 기반 재현은 데이터와 모델 가중치 및 GPU가 별도로 필요합니다.

## 설정 파일 안내

공유용 시작점은 `configs/osl/personpath22_pose2.yaml`과 `configs/cucps/prw_pose2.yaml`입니다. 예제의 `/path/to/...` 값과 사용할 시퀀스 목록을 실제 환경에 맞게 수정하세요. 예제 PersonPath22 설정은 시퀀스 하나만 선택하므로 기존 68개 시퀀스 실험의 재현 설정이 아닙니다. 각 실험은 별도의 출력 디렉터리와 prior 캐시를 사용하세요.

`configs/` 최상위의 파일은 이전 실험 명령과 재현 이력을 위해 보관합니다. 일부에 서버 절대경로와 과거 결과 경로가 남아 있으므로 실행 전에 확인하세요. `configs/examples/`는 이전 공유용 예제 경로의 호환성을 위해 남겨두었습니다.
