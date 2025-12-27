#!/usr/bin/env python3
"""
TXT(cx, cy, w, h, angle) → YOLO-OBB 转换器

输入:
- TXT 每行: `class cx cy w h angle`
- 坐标可为归一化(0–1)或像素值；若为像素值需提供 `--images` 用于归一化

示例输入(归一化):
- 0 0.500000 0.500000 0.200000 0.100000 0.785398

示例输出:
- 0 0.435355 0.435355 0.564645 0.435355 0.564645 0.564645 0.435355 0.564645

示例命令:
- python -m convert.txt_cxwha_to_yolo_obb --labels /data/labels_cxwha --images /data/images --output /data/labels_obb --angle-unit rad
"""
import argparse
import math
from pathlib import Path
from typing import List, Optional

from .common import clamp01, norm_auto, to_obb_from_cxwh_angle, write_label, find_image_file


def convert_dir(labels_dir: Path, images_dir: Optional[Path], out_dir: Path, normalize: str, angle_unit: str) -> None:
    files = [labels_dir] if labels_dir.is_file() and labels_dir.suffix.lower() == ".txt" else list(labels_dir.glob("*.txt"))
    for lp in files:
        base = lp.stem
        lines_out: List[str] = []
        iw = ih = None
        if normalize != "no" and images_dir is not None:
            ip = find_image_file(images_dir, base)
            if ip:
                try:
                    from PIL import Image
                    with Image.open(ip) as im:
                        iw, ih = im.size
                except Exception:
                    iw = ih = None
        with open(lp, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().split()
                if len(s) < 6:
                    continue
                try:
                    cls = int(float(s[0]))
                    cx, cy, w, h, ang = float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5])
                except Exception:
                    continue
                ang_rad = ang
                if angle_unit == "deg":
                    ang_rad = math.radians(ang)
                elif angle_unit == "rad":
                    ang_rad = ang
                else:
                    if abs(ang) > 6.3:
                        ang_rad = math.radians(ang)
                is_norm = normalize == "yes" or (normalize == "auto" and norm_auto([cx, cy, w, h]))
                if is_norm:
                    pts = to_obb_from_cxwh_angle(cx, cy, w, h, ang_rad)
                    coords: List[float] = []
                    for x, y in pts:
                        coords.extend([clamp01(x), clamp01(y)])
                    lines_out.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in coords))
                else:
                    if iw is None or ih is None:
                        continue
                    pts = to_obb_from_cxwh_angle(cx, cy, w, h, ang_rad)
                    coords: List[float] = []
                    for x, y in pts:
                        coords.extend([clamp01(x / iw), clamp01(y / ih)])
                    lines_out.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in coords))
        write_label(out_dir / (base + ".txt"), lines_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=str, required=True)
    p.add_argument("--images", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--normalize", type=str, default="auto", choices=["auto", "yes", "no"])
    p.add_argument("--angle-unit", type=str, default="auto", choices=["auto", "deg", "rad"])
    args = p.parse_args()
    labels_dir = Path(args.labels)
    images_dir = Path(args.images) if args.images else None
    out_dir = Path(args.output) if args.output else (labels_dir.parent / "labels_obb")
    out_dir.mkdir(parents=True, exist_ok=True)
    convert_dir(labels_dir, images_dir, out_dir, args.normalize, args.angle_unit)
    print(str(out_dir))


if __name__ == "__main__":
    main()
