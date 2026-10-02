"""
RAW 영상 프레임 데이터에서 클래스당 400장(train 320 / val 80)을 뽑아
dataset/images/train, dataset/images/val 폴더로 복사한다.

- 영상 단위로 먼저 train/val을 나눈 뒤, 각 영상에서 프레임 1장만 무작위로 뽑는다.
  (같은 영상의 프레임이 train/val 양쪽에 섞이는 데이터 유출을 방지)
- footup 외 클래스는 완전 무작위 대신 영상 중간 구간(프레임 순서 기준 40~60%
  지점)에서만 뽑는다. 영상 시작/끝부분은 해당 행동이 아직 시작 전이거나 끝난
  뒤일 가능성이 높아 라벨과 실제 자세가 안 맞는 프레임(라벨 노이즈)이 섞이기
  쉽기 때문.
- footup은 JSON의 keypoints를 이용해 "손(앞발)이 가장 높이 들린" 프레임을
  고른다. footup 영상들에서 프레임별 keypoint y좌표 변동폭을 분석해보니
  10~15번 keypoint가 가장 크게 움직였고(앞다리/발로 추정), 이 keypoint들이
  bbox 내에서 가장 위쪽에 있는 프레임을 "손을 든 순간"으로 간주한다.
- 원본 RAW 파일은 건드리지 않고 복사만 한다.
"""

import csv
import json
import random
import re
import shutil
from pathlib import Path

SEED = 42
IMAGES_PER_CLASS = 800
TRAIN_RATIO = 0.8
MID_RANGE = (0.4, 0.6)  # 프레임 순서 기준 중간 구간 비율 (footup 외 클래스)
EDGE_TRIM = (0.1, 0.9)  # footup: 양끝 10%를 제외한 구간에서만 후보 선정
PAW_KEYPOINTS = ["10", "11", "12", "13", "14", "15"]  # 변동폭 분석으로 추정한 앞다리/발 keypoint

FRAME_NUM_RE = re.compile(r"frame_(\d+)_timestamp")

ROOT = Path(__file__).resolve().parent.parent
RAW_IMAGES = ROOT / "RAW" / "images"
RAW_LABELS = ROOT / "RAW" / "labels"
DATASET_DIR = ROOT / "dataset"

CLASSES = ["footup", "lying", "sit", "walkrun"]


def frame_num(p: Path) -> int:
    m = FRAME_NUM_RE.match(p.name)
    return int(m.group(1)) if m else 0


def pick_mid_frame(rng: random.Random, frames: list[Path]) -> Path:
    frames_sorted = sorted(frames, key=frame_num)
    n = len(frames_sorted)
    lo = int(n * MID_RANGE[0])
    hi = max(lo + 1, int(n * MID_RANGE[1]) + 1)
    candidates = frames_sorted[lo:hi] or frames_sorted
    return rng.choice(candidates)


def pick_footup_frame(rng: random.Random, class_name: str, video_dir: Path, frames: list[Path]) -> Path:
    label_path = RAW_LABELS / class_name / f"{video_dir.name}.json"
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return pick_mid_frame(rng, frames)

    anns_by_frame = {a["frame_number"]: a for a in data.get("annotations", [])}
    files_by_num = {frame_num(f): f for f in frames}
    avail = sorted(files_by_num.keys())
    n = len(avail)
    lo = int(n * EDGE_TRIM[0])
    hi = max(lo + 1, int(n * EDGE_TRIM[1]))
    candidates = avail[lo:hi] or avail

    scored = []
    for fn in candidates:
        ann = anns_by_frame.get(fn)
        bbox = ann.get("bounding_box") if ann else None
        if not bbox:
            continue
        kps = ann.get("keypoints", {})
        ys = [kps[k]["y"] for k in PAW_KEYPOINTS if kps.get(k)]
        if not ys:
            continue
        rel_y = (sum(ys) / len(ys) - bbox["y"]) / bbox["height"]  # 작을수록 bbox 위쪽(=손이 높이 들림)
        scored.append((rel_y, files_by_num[fn]))

    if not scored:
        return pick_mid_frame(rng, frames)
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def load_label_meta(class_name: str, video_dir_name: str) -> dict:
    label_path = RAW_LABELS / class_name / f"{video_dir_name}.json"
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        return {
            "action": meta.get("action", ""),
            "breed": meta.get("animal", {}).get("breed", ""),
        }
    except (FileNotFoundError, json.JSONDecodeError):
        return {"action": "", "breed": ""}


def main():
    rng = random.Random(SEED)

    for split in ("train", "val"):
        for class_name in CLASSES:
            (DATASET_DIR / "images" / split / class_name).mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for class_name in CLASSES:
        class_img_dir = RAW_IMAGES / class_name
        video_dirs = sorted(p for p in class_img_dir.iterdir() if p.is_dir())
        rng.shuffle(video_dirs)

        n_needed = IMAGES_PER_CLASS
        if len(video_dirs) < n_needed:
            print(f"[경고] {class_name}: 영상 {len(video_dirs)}개뿐, {n_needed}개 필요 -> 있는 만큼만 사용")
            n_needed = len(video_dirs)

        selected = video_dirs[:n_needed]
        n_train = round(n_needed * TRAIN_RATIO)

        splits = [("train", selected[:n_train]), ("val", selected[n_train:])]

        for split_name, videos in splits:
            for video_dir in videos:
                frames = list(video_dir.glob("*.jpg"))
                if not frames:
                    print(f"[경고] 프레임 없음, 스킵: {video_dir}")
                    continue
                if class_name == "footup":
                    frame = pick_footup_frame(rng, class_name, video_dir, frames)
                else:
                    frame = pick_mid_frame(rng, frames)

                video_id = video_dir.name  # e.g. "20201022_dog-lying-000021.mp4"
                dest_name = f"{video_id}__{frame.name}"
                dest_path = DATASET_DIR / "images" / split_name / class_name / dest_name
                shutil.copy2(frame, dest_path)

                meta = load_label_meta(class_name, video_id)
                manifest_rows.append({
                    "split": split_name,
                    "class": class_name,
                    "filename": dest_name,
                    "source_video": video_id,
                    "source_frame": frame.name,
                    "breed": meta["breed"],
                    "action": meta["action"],
                })

        print(f"{class_name}: train={len(splits[0][1])}, val={len(splits[1][1])}")

    manifest_path = DATASET_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "class", "filename", "source_video", "source_frame", "breed", "action"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n총 {len(manifest_rows)}장 복사 완료. manifest: {manifest_path}")


if __name__ == "__main__":
    main()
