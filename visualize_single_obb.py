import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np
import random

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_labels(label_path, w, h):
    boxes = []
    if not Path(label_path).exists():
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                cls = int(float(parts[0]))
                nums = list(map(float, parts[1:9]))
            except Exception:
                continue
            pts = [(int(nums[i] * w), int(nums[i + 1] * h)) for i in range(0, 8, 2)]
            boxes.append((cls, pts))
    return boxes

def draw_obb(img, boxes, names):
    h, w = img.shape[:2]
    rng = random.Random(0)
    palette = {}
    for cls, pts in boxes:
        if cls not in palette:
            palette[cls] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        color = palette[cls]
        cnt = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [cnt], isClosed=True, color=color, thickness=2)
        label = names[cls] if 0 <= cls < len(names) else str(cls)
        x, y = pts[0]
        cv2.rectangle(img, (x, max(0, y - 22)), (min(w, x + 120), y), color, -1)
        cv2.putText(img, label, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img

def build_names(data):
    names = data.get("names")
    if isinstance(names, dict):
        items = sorted(((int(k), v) for k, v in names.items()), key=lambda x: x[0])
        names = [v for _, v in items]
    return names if isinstance(names, list) else []

def visualize_single(image_path, label_path, yaml_path, output_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    h, w = img.shape[:2]
    
    if yaml_path:
        data = load_yaml(yaml_path)
        names = build_names(data)
    else:
        names = []
    
    boxes = parse_labels(label_path, w, h)
    img = draw_obb(img, boxes, names)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Saved visualization to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--label", type=str, required=True, help="Path to label file")
    parser.add_argument("--yaml", type=str, default=None, help="Path to yaml file for class names")
    parser.add_argument("--output", type=str, default="visualized.jpg", help="Path to output image")
    args = parser.parse_args()
    visualize_single(args.image, args.label, args.yaml, args.output)

if __name__ == "__main__":
    main()
