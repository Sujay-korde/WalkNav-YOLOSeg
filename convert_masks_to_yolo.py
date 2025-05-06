import os
import cv2
import numpy as np
from pathlib import Path

# Paths
image_dir = Path("dataset/images/train")
mask_dir = Path("dataset/masks/train")
label_dir = Path("dataset/labels/train")
label_dir.mkdir(parents=True, exist_ok=True)

def mask_to_yolo_seg(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []

    for cnt in contours:
        if len(cnt) >= 6:  # YOLO requires at least 6 points
            cnt = cnt.squeeze(1)
            segment = []
            for x, y in cnt:
                segment.append(x / mask.shape[1])
                segment.append(y / mask.shape[0])
            segments.append(segment)

    return segments

for image_file in image_dir.glob("*.png"):
    # Use same name for mask as image
    mask_file = mask_dir / image_file.name
    label_file = label_dir / (image_file.stem + ".txt")

    if not mask_file.exists():
        print(f"Mask not found for {image_file.name}")
        continue

    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    segments = mask_to_yolo_seg(bin_mask)

    with open(label_file, "w") as f:
        for segment in segments:
            segment_str = " ".join([f"{pt:.6f}" for pt in segment])
            f.write(f"0 {segment_str}\n")  # '0' is the class ID for walkable path
