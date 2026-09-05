"""
dataset/manifest.csv를 기준으로 RAW/labels/<class>/<video_id>.json의
bounding_box를 YOLO 포맷(.txt)으로 변환해 dataset/labels/<split>/<class>/에 저장한다.

- class 인덱스: footup=0, lying=1, sit=2, walkrun=3 (build_dataset.py와 동일 순서)
- 좌표 정규화는 metadata를 신뢰하지 않고, 실제 이미지 파일을 열어 실측한 크기 기준으로 계산한다.
"""

import csv
import json
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
RAW_LABELS = ROOT / "RAW" / "labels"
MANIFEST = DATASET_DIR / "manifest.csv"

CLASSES = ["footup", "lying", "sit", "walkrun"]
FRAME_NUM_RE = re.compile(r"frame_(\d+)_timestamp")

_json_cache: dict[str, dict] = {}


def load_video_json(class_name: str, video_id: str) -> dict | None:
    key = f"{class_name}/{video_id}"
    if key not in _json_cache:
        path = RAW_LABELS / class_name / f"{video_id}.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                _json_cache[key] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _json_cache[key] = None
    return _json_cache[key]


def main():
    converted = 0
    skipped = []

    with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        split = row["split"]
        class_name = row["class"]
        filename = row["filename"]
        source_video = row["source_video"]
        source_frame = row["source_frame"]

        m = FRAME_NUM_RE.match(source_frame)
        if not m:
            skipped.append((filename, "frame_number 파싱 실패"))
            continue
        frame_number = int(m.group(1))

        data = load_video_json(class_name, source_video)
        if data is None:
            skipped.append((filename, f"json 없음/파싱실패: {source_video}"))
            continue

        ann = next((a for a in data["annotations"] if a["frame_number"] == frame_number), None)
        if ann is None:
            skipped.append((filename, f"frame_number {frame_number} annotation 없음"))
            continue

        bbox = ann.get("bounding_box")
        if not bbox:
            skipped.append((filename, "bounding_box 없음"))
            continue

        img_path = DATASET_DIR / "images" / split / class_name / filename
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except FileNotFoundError:
            skipped.append((filename, f"이미지 파일 없음: {img_path}"))
            continue

        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h

        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)

        class_idx = CLASSES.index(class_name)

        label_path = DATASET_DIR / "labels" / split / class_name / (Path(filename).stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        converted += 1

    print(f"변환 완료: {converted}개")
    print(f"스킵: {len(skipped)}개")
    for fname, reason in skipped[:20]:
        print(f"  - {fname}: {reason}")


if __name__ == "__main__":
    main()
