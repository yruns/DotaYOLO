import argparse
import os
import random
from pathlib import Path
import yaml
import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def list_images(dir_path):
    d = Path(dir_path)
    return [x for x in d.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXTS]

def get_label_path(image_path, images_root, labels_root):
    rel = Path(image_path).relative_to(images_root)
    return Path(labels_root) / rel.with_suffix(".txt")

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

def make_collage(images, rows, cols, tile):
    h, w = tile
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, im in enumerate(images):
        if idx >= rows * cols:
            break
        r = idx // cols
        c = idx % cols
        tile_img = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile_img
    return canvas

def resolve_paths(data, split):
    root = Path(data.get("path") or Path(data.get("yaml_file", "")).parent)
    images_rel = data[split]
    labels_rel = images_rel.replace("images", "labels")
    return (root / images_rel, root / labels_rel)

def build_names(data):
    names = data.get("names")
    if isinstance(names, dict):
        items = sorted(((int(k), v) for k, v in names.items()), key=lambda x: x[0])
        names = [v for _, v in items]
    return names if isinstance(names, list) else []

def visualize(yaml_path, out_dir, grid_cols=3, grid_rows=2, per_collage=6, collages=3, tile_w=512, tile_h=512):
    data = load_yaml(yaml_path)
    names = build_names(data)
    ds_name = Path(yaml_path).stem
    base_out = Path(out_dir) / ds_name
    ensure_dir(base_out)
    for split in ["train", "val"]:
        imgs_dir, labels_dir = resolve_paths(data, split)
        imgs = list_images(imgs_dir)
        if not imgs:
            continue
        k = min(len(imgs), per_collage * collages)
        sample = random.sample(imgs, k) if len(imgs) >= k else imgs
        for ci in range(collages):
            start = ci * per_collage
            end = start + per_collage
            batch = sample[start:end]
            if not batch:
                break
            tiles = []
            for ip in batch:
                im = cv2.imread(str(ip))
                if im is None:
                    continue
                h, w = im.shape[:2]
                lp = get_label_path(ip, imgs_dir, labels_dir)
                boxes = parse_labels(lp, w, h)
                im = draw_obb(im, boxes, names)
                tiles.append(im)
            if not tiles:
                continue
            collage = make_collage(tiles, grid_rows, grid_cols, (tile_h, tile_w))
            out_path = base_out / f"{split}_collage_{ci+1}.jpg"
            cv2.imwrite(str(out_path), collage)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, required=True)
    parser.add_argument("--out", type=str, default="visualizations")
    args = parser.parse_args()
    visualize(args.yaml, args.out)

if __name__ == "__main__":
    main()
