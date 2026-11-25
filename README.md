# One Second Later

PersonPath22에서 사람의 다음 위치를 히트맵으로 예측하는 **OSL(One Second Later)** 모델을 학습하고, 이 모델의 위치 점수를 PRW의 CUCPS pseudo-ID 학습에 사용하는 방법입니다. 아래 명령은 모두 `One_Second_Later` 디렉터리에서 실행합니다.

OSL은 현재 이미지, 사람 bbox, 사람 주변 crop과 RTMPose 관절을 입력받습니다. Pose 특징을 사람 외형 특징에 gated fusion으로 결합하고 다음 위치 히트맵을 출력합니다. PRW에서는 OSL을 고정한 채 CUCPS를 학습합니다. 기존 코드와 설정에서 OSL은 `e2c_p2`, `pose_p2`라는 내부 식별자를 사용합니다.

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

OSL용 [RTMPose](https://github.com/Tau-J/rtmlib) 모델은 `cache_pose.py`가 처음 실행될 때 다운로드합니다. CUCPS 학습은 torchvision의 사전학습 ResNet-50 백본을 사용하므로 처음 실행할 때 해당 가중치 다운로드가 필요할 수 있습니다.

### PersonPath22 다운로드와 전처리

1. [PersonPath22 공식 저장소](https://github.com/amazon-science/tracking-dataset)의 다운로드 안내를 따라 영상과 annotation을 받습니다. 공식 다운로드 스크립트는 AWS CLI와 `ffmpeg`/`ffprobe`가 필요하며, 일부 외부 데이터는 별도 안내가 있습니다.
2. 학습에는 원본 영상 자체가 아니라, ground-view 영상을 선택해 **같은 사람의 현재 프레임과 약 1초 후 프레임을 연결한 JSONL 및 추출 이미지**가 필요합니다. 기존 실험은 68개 영상을 수동 선택했고, 각 영상의 시간순 80:20 train/validation 분할을 사용했습니다.
3. 현재 저장소에는 이 ground-view 영상 목록과 전처리 스크립트가 **포함돼 있지 않습니다**. 기존 작업 환경에서는 저장소 상위 폴더의 `PersonPath22_analysis/ground_view_candidates.csv`와 `PersonPath22_ground_view/build_dataset.py`를 사용했습니다. 새 환경에서는 이 두 파일을 별도로 확보하고 CSV의 영상·annotation 경로를 실제 다운로드 위치로 수정한 뒤 실행하거나, 아래 형식의 학습 데이터를 직접 준비해야 합니다. 원본 다운로드만으로는 다음 학습 명령을 실행할 수 없습니다.

기존 전처리 자료가 있는 경우 상위 폴더에서 `python PersonPath22_ground_view/build_dataset.py`로 생성할 수 있습니다. 이 스크립트는 기존 생성 파일을 덮어쓰므로 별도 위치에서 실행하세요. 결과 구조는 다음과 같습니다.

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

PRW OSL 학습 설정은 OSL 체크포인트 외에도 `motion.checkpoint`의 **기존 E2-C 체크포인트 파일**을 캐시 메타데이터 확인에 요구합니다. 이 파일은 저장소에 포함되지 않으므로 기존 실행에서 생성한 파일을 확보하고 YAML에 경로를 지정해야 합니다. 현재 코드만으로는 새로 학습한 OSL만으로 PRW 단계를 시작할 수 없습니다.

## 1. PersonPath22 OSL 위치 예측 학습

`configs/personpath22_osl.yaml`은 PersonPath22 ground-view 68개 시퀀스와 독립된 pose 캐시 및 출력 경로를 사용합니다. 데이터 위치와 DINOv3 가중치 경로를 환경에 맞게 확인하세요. 새 실험마다 `training.output`을 바꾸세요.

현재 환경에는 68개 시퀀스가 있습니다. 위 설정은 기존 시퀀스별 train/val 분할을 그대로 사용하므로 두 분할이 같은 장면을 공유합니다. Pose 캐시를 생성한 후 OSL을 학습합니다. `CUDA_VISIBLE_DEVICES`는 사용할 GPU 번호로 바꾸세요.

```bash
python cache_pose.py --config configs/personpath22_osl.yaml --split train
python cache_pose.py --config configs/personpath22_osl.yaml --split val
CUDA_VISIBLE_DEVICES=2 python run.py train --config configs/personpath22_osl.yaml
```

캐시는 재실행 시 기존 레코드를 재사용합니다. 학습은 검증 FDE가 가장 낮은 모델을 `outputs/personpath22-osl/run-01/best.pt`에 저장합니다. 기존 출력 디렉터리에 학습 로그가 있으면 덮어쓰지 않으므로 설정에서 새 경로를 지정하세요.

## 2. PRW OSL 기반 CUCPS 학습

`configs/cucps_temporal_prw_osl.yaml`은 위 학습의 `best.pt`를 `motion.pose_checkpoint`로 사용하며, 별도의 prior 캐시와 출력 경로를 지정합니다. 데이터·체크포인트 경로를 확인하세요. 다른 OSL 체크포인트를 사용할 때는 `motion.pose_checkpoint`와 `temporal_prior.cache`를 함께 변경하세요.

이 설정의 `motion.checkpoint`도 캐시 메타데이터 생성에 사용되므로 기존 E2-C 체크포인트가 해당 경로에 있어야 합니다. 경로가 다르면 YAML에서 수정하세요. OSL 가중치는 `motion.pose_checkpoint`에서 불러옵니다.

```bash
CUDA_VISIBLE_DEVICES=2 python cucps_temporal_experiment.py train \
  --config configs/cucps_temporal_prw_osl.yaml --variant pose_p2
```

설정상 CUCPS를 25에폭 학습하며, timestamp가 붙은 실행 디렉터리에 체크포인트가 저장됩니다. 시작 시 PRW 학습 이미지의 bbox로 OSL 히트맵을 만들고 다음에 존재하는 같은 카메라·장면 이미지의 후보 bbox마다 위치 점수를 캐시합니다. 이 점수는 CUCPS 학습 중 HNM/HPM의 pseudo-ID 후보 선택에 사용됩니다. PRW의 사람 ID 정답으로 OSL을 학습하거나 위치 점수를 맞추지 않습니다. PRW 이미지의 프레임 번호 간격은 일정하지 않습니다.

OSL은 **CUCPS 학습 중에만** 보조 단서로 사용됩니다. 저장된 CUCPS 체크포인트로 검색할 때 히트맵 점수를 별도로 더하지는 않습니다.

## 기존 체크포인트에 대한 주의

이전 PRW 실험에서 사용한 `outputs/pose-runs/20261123-0228/p2/best.pt`에는 학습 데이터가 `../MOT17_prediction`으로 기록돼 있습니다. 따라서 이 체크포인트를 PersonPath22에서 학습한 OSL로 간주하면 안 됩니다. `configs/cucps_temporal_prw_pose_p2.yaml`은 기본적으로 이 기존 체크포인트를 가리킵니다. PersonPath22에서 학습한 OSL을 쓰려면 위처럼 `motion.pose_checkpoint`를 변경해야 합니다.

## 3. OSL + 장면 분할 학습

`configs/personpath22_osl_s.yaml`은 OSL에 SegFormer-B2의 150개 장면 분할 확률 특징을 추가합니다. 1번 단계의 같은 PersonPath22 데이터와 pose 캐시를 사용하며, SegFormer는 학습 전 이미지별 semantic 캐시를 생성할 때만 사용합니다. 첫 실행에는 Hugging Face의 모델 가중치 다운로드가 필요합니다.

```bash
CUDA_VISIBLE_DEVICES=8 python cache_semantic.py --config configs/personpath22_osl_s.yaml --split train
CUDA_VISIBLE_DEVICES=8 python cache_semantic.py --config configs/personpath22_osl_s.yaml --split val
CUDA_VISIBLE_DEVICES=8 python run.py train --config configs/personpath22_osl_s.yaml
```

pose 및 semantic train/val 캐시가 모두 준비돼 있어야 학습할 수 있습니다. Semantic 캐시는 이미지별로 저장되며 재실행 시 기존 항목을 재사용합니다. 출력은 `outputs/personpath22-osl-s/run-01`에 저장됩니다. 데이터와 가중치 경로 및 GPU 번호는 환경에 맞게 변경하세요.
