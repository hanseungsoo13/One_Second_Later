# 코드와 데이터 흐름

## OSL: 위치 예측

`scripts/osl/build_personpath22.py`는 선택한 PersonPath22 영상과 amodal annotation에서 동일 인물의 현재·약 1초 후 위치 쌍과 이미지를 생성한다. 시퀀스별 시간순 80:20 분할을 사용하므로 검증 결과는 새로운 장면에 대한 일반화 성능을 직접 나타내지는 않는다.

`scripts/osl/cache_pose.py`는 bbox별 RTMPose 관절을 SQLite에 기록한다. semantic 모델에는 `scripts/osl/cache_semantic.py`의 이미지별 SegFormer 확률 캐시도 필요하다. `scripts/osl/train.py`와 `scripts/osl/evaluate.py`는 각각 OSL을 학습·평가하며, 구현은 `src/osl/`에 있다. 모델별 입력은 `src/osl/variants.py`, 설정 검증은 `src/osl/config.py`에서 관리한다.

설정의 이미지 크기는 `[height, width]`, 원본 annotation의 `image_size`는 `[width, height]`, `bbox_xywh`는 원본 이미지 픽셀 좌표다. 네트워크 bbox는 letterbox 변환을 적용한 `[x1, y1, x2, y2]`다. 타깃은 미래 bbox의 아래쪽 중심점이다.

## 연결: OSL 위치 prior

`src/integration/`은 고정된 OSL로 PRW의 같은 카메라·장면에 있는 다음 프레임 후보 bbox에 위치 점수를 부여하고 CUCPS 학습에 사용할 sparse prior를 만든다. `prior.py`가 생성과 캐시 검증을, `motion_inputs.py`가 입력 준비를, `temporal.py`가 prior 및 graph 계산을 담당한다. prior는 CUCPS 학습 시작 시 필요하면 자동으로 생성한다.

캐시 메타데이터 버전 2는 데이터 구성, 모델 변형, 실제 사용한 체크포인트와 입력 설정을 기록한다. 이전 형식과 일치하지 않으면 오류를 낸다. 기존 파일을 보존하고 새 경로를 지정해 생성한다. pose 모델은 `motion.pose_checkpoint`, pose가 없는 모델은 `motion.checkpoint`를 사용한다. segmentation 모델은 `motion.semantic`의 모델과 revision이 필요하다.

## CUCPS: person search 학습과 공식 평가

`scripts/cucps/train.py`, `evaluate.py`, `compare.py`는 각각 학습, 공식 PRW 평가, baseline 대비 결과 비교를 실행한다. 구현은 `src/cucps/`에 있으며 `compat.py`는 고정된 외부 CUCPS 버전과 현재 환경의 차이를 처리한다.

`baseline`은 OSL을 사용하지 않는 CUCPS 기준 모델이고 `full_bbox_*`는 위치 prior를 만드는 OSL 모델을 선택한다. `graph`는 기존 `full_bbox_c`를 사용하는 별도 prior 활용 방식이다. PRW 정답 사람 ID는 prior 생성이나 pseudo-ID 학습에 사용하지 않는다.

## 별도 분석 실험

`experiments/prw_motion/`은 PRW 쌍 생성 및 OSL 위치 예측 지표를 분석한다. `experiments/temporal_reranking/`은 검색 시점에 위치 점수를 추가하는 별도 실험이다. 이 재정렬은 CUCPS 학습 중 prior를 쓰는 핵심 경로와 다르다. 각 공유용 설정은 `configs/experiments/`에 있다.

`scripts/build_personpath22.py`도 기존 경로의 호환 진입점이다. 외부 데이터·가중치·체크포인트·결과물은 저장소에 포함하지 않는다. 구체적인 준비와 명령은 [README](../README.md)를 참고한다.
