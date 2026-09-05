"""
RAW 영상 프레임 데이터에서 클래스당 400장(train 320 / val 80)을 뽑아
dataset/images/train, dataset/images/val 폴더로 복사한다.

- 영상 단위로 먼저 train/val을 나눈 뒤, 각 영상에서 프레임 1장만 무작위로 뽑는다.
  (같은 영상의 프레임이 train/val 양쪽에 섞이는 데이터 유출을 방지)
- 원본 RAW 파일은 건드리지 않고 복사만 한다.
"""

import csv
import json
import random
import shutil
from pathlib import Path

SEED = 42
IMAGES_PER_CLASS = 400
TRAIN_RATIO = 0.8

ROOT = Path(__file__).resolve().parent.parent
RAW_IMAGES = ROOT / "RAW" / "images"
RAW_LABELS = ROOT / "RAW" / "labels"
DATASET_DIR = ROOT / "dataset"

CLASSES = ["footup", "lying", "sit", "walkrun"]


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
                frame = rng.choice(frames)

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
