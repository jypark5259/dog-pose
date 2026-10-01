# dog-behavior-analysis

반려견의 행동(손 주기/누움/앉음/걷거나 뜀)을 이미지에서 탐지하는 YOLOv8 기반 프로젝트입니다.

## 클래스

| 클래스 | 설명 |
|---|---|
| `footup` | 손(앞발) 주기 |
| `lying` | 누움 |
| `sit` | 앉음 |
| `walkrun` | 걷거나 뜀 |

## 데이터

[AI Hub 반려동물 행동 영상 데이터](https://www.aihub.or.kr/)를 사용합니다. 원본 영상에서 프레임을 추출한 이미지와, 프레임별 bounding box·keypoint 라벨(JSON)이 클래스 폴더별로 제공됩니다.

원본 데이터가 크기 때문에(클래스당 영상 수천 개, 프레임 수만 장) 파이프라인 검증용으로 아래 기준으로 소규모 서브셋을 뽑아 사용합니다.

- **클래스당 400장** (train 320 / val 80, 80:20 분할)
- **영상 단위로 먼저 train/val 분할** 후, 각 영상에서 프레임 1장만 무작위 추출 — 같은 영상의 프레임이 train/val 양쪽에 섞여 생기는 데이터 유출(leakage) 방지
- 시드 고정(42)으로 재현 가능

## 프로젝트 구조

```
dog-pose-project/
  RAW/                    # 원본 데이터 (git 미포함)
    images/<class>/<video_id>.mp4/frame_*.jpg
    labels/<class>/<video_id>.mp4.json
  dataset/                # 학습용 서브셋 (이미지/라벨은 git 미포함, 아래 두 개만 포함)
    manifest.csv           # 추출된 이미지 ↔ 원본 영상 매핑 + 견종/행동 설명
    data.yaml               # YOLO 학습용 데이터셋 설정
  scripts/
    build_dataset.py         # RAW에서 소규모 데이터셋 추출 (이미지 복사 + manifest 생성)
    convert_to_yolo.py       # JSON bounding_box → YOLO txt 라벨 변환
  assets/                  # README용 결과 이미지
  requirements.txt
```

## 환경 설정

```bash
pip install -r requirements.txt
```

- PyTorch는 CUDA 12.8(cu128) 빌드로 고정되어 있습니다 — RTX 50 시리즈(Blackwell, sm_120) GPU 지원을 위해 필요합니다. 다른 GPU를 쓴다면 본인 환경에 맞는 CUDA 빌드로 바꿔도 됩니다.

## 데이터 준비

```bash
python scripts/build_dataset.py     # RAW → dataset/images, dataset/manifest.csv
python scripts/convert_to_yolo.py   # manifest.csv 기준으로 dataset/labels/*.txt 생성
```

## 학습

wandb로 실험을 기록합니다.

```bash
wandb login                 # 최초 1회
yolo settings wandb=True    # 최초 1회

yolo detect train model=yolov8n.pt data=dataset/data.yaml \
  epochs=100 imgsz=640 batch=16 patience=20 \
  project=dog-pose-behavior name=yolov8n-baseline
```

## Baseline 결과 (yolov8n, 클래스당 400장)

99 epoch에서 조기 종료 (best epoch 79, `patience=20`).

| metric | 값 (best epoch 기준) |
|---|---|
| precision | 0.669 |
| recall | 0.628 |
| mAP50 | 0.679 |
| mAP50-95 | 0.542 |

![baseline results](assets/baseline-results.png)

### Confusion Matrix

![baseline confusion matrix](assets/baseline-confusion-matrix.png)

클래스별 정확도: `footup` 0.67, `lying` 0.65, `sit` 0.61, `walkrun` 0.52. 눈에 띄는 혼동은 `walkrun`→`sit` 26%, `footup`→`sit` 24% — `sit`이 다른 클래스의 애매한 프레임을 흡수하는 경향이 있습니다 (걷다가 멈칫하는 순간, 손 주며 앉아있는 프레임 등이 단순 `sit`으로 오분류).

다음 개선 방향 후보:

1. `walkrun`/`footup` 클래스 데이터 추가 확보 (현재 각 400장만 사용, RAW에는 더 많음)
2. `sit`과 헷갈리는 오분류 샘플 직접 확인 후, 애매한 프레임 제거·재선별
3. 더 큰 모델(`yolov8s.pt`)로 재학습
