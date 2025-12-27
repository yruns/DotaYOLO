#!/usr/bin/env python3
"""
COCO → YOLO-OBB 转换器

输入:
- COCO JSON (支持 `segmentation` 四点/多边形，或 `rbox/obb/rbbox=[cx,cy,w,h,angle]`)

示例输入:
{
  "images": [{"id": 1, "file_name": "0001.jpg", "width": 1024, "height": 768}],
  "categories": [{"id": 1, "name": "ship"}],
  "annotations": [
    {"image_id": 1, "category_id": 1, "segmentation": [[100,200, 180,190, 190,250, 110,260]]}
  ]
}

输出:
- 生成 `labels_obb/0001.txt`，每行: `class_id x1 y1 x2 y2 x3 y3 x4 y4`

示例输出:
- 0 0.097656 0.260417 0.175781 0.247396 0.185547 0.325521 0.107422 0.338542

示例命令:
- python -m convert.coco_to_yolo_obb --input /data/annotations.json --output /data/labels_obb --normalize auto --angle-unit auto
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from convert.common import clamp01, norm_auto, to_obb_from_cxwh_angle, write_label, min_area_rect


def convert(coco_json: Path, out_dir: Path, normalize: str, angle_unit: str) -> None:
    data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    imgs = {int(x.get("id")): {"file": x.get("file_name"), "w": int(x.get("width") or 0), "h": int(x.get("height") or 0)} for x in data.get("images", [])}
    cats = data.get("categories", [])
    cat_ids = [int(c.get("id")) for c in cats]
    cat_id_to_idx: Dict[int, int] = {cid: i for i, cid in enumerate(cat_ids)}
    by_img: Dict[int, List[dict]] = {}
    for ann in data.get("annotations", []):
        iid = int(ann.get("image_id"))
        by_img.setdefault(iid, []).append(ann)
    for iid, anns in by_img.items():
        info = imgs.get(iid)
        if not info:
            continue
        iw = info["w"]
        ih = info["h"]
        if iw <= 0 or ih <= 0:
            continue
        base = Path(info.get("file") or str(iid)).stem
        lines: List[str] = []
        for ann in anns:
            cid = int(ann.get("category_id"))
            idx = cat_id_to_idx.get(cid, 0)
            used = False
            rbox = ann.get("rbox") or ann.get("obb") or ann.get("rbbox")
            if isinstance(rbox, (list, tuple)) and len(rbox) >= 5:
                cx, cy, w, h, ang = float(rbox[0]), float(rbox[1]), float(rbox[2]), float(rbox[3]), float(rbox[4])
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
                    cx *= iw
                    cy *= ih
                    w *= iw
                    h *= ih
                pts = to_obb_from_cxwh_angle(cx, cy, w, h, ang_rad)
                coords: List[float] = []
                for x, y in pts:
                    coords.extend([clamp01(x / iw), clamp01(y / ih)])
                lines.append(str(idx) + " " + " ".join(f"{v:.6f}" for v in coords))
                used = True
            seg = ann.get("segmentation")
            if not used and isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
                poly = seg[0]
                if isinstance(poly, list) and len(poly) >= 8:
                    pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
                    if len(pts) == 4:
                        coords: List[float] = []
                        for x, y in pts:
                            coords.extend([clamp01(x / iw), clamp01(y / ih)])
                        lines.append(str(idx) + " " + " ".join(f"{v:.6f}" for v in coords))
                        used = True
                    else:
                        obb = min_area_rect(pts)
                        coords: List[float] = []
                        for x, y in obb:
                            coords.extend([clamp01(x / iw), clamp01(y / ih)])
                        lines.append(str(idx) + " " + " ".join(f"{v:.6f}" for v in coords))
                        used = True
            if not used:
                bbox = ann.get("bbox")
                if isinstance(bbox, list) and len(bbox) >= 4:
                    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    is_norm = normalize == "yes" or (normalize == "auto" and norm_auto([x, y, w, h]))
                    if is_norm:
                        x *= iw
                        y *= ih
                        w *= iw
                        h *= ih
                    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                    coords: List[float] = []
                    for xx, yy in pts:
                        coords.extend([clamp01(xx / iw), clamp01(yy / ih)])
                    lines.append(str(idx) + " " + " ".join(f"{v:.6f}" for v in coords))
        write_label(out_dir / (base + ".txt"), lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--normalize", type=str, default="auto", choices=["auto", "yes", "no"])
    p.add_argument("--angle-unit", type=str, default="auto", choices=["auto", "deg", "rad"])
    args = p.parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output) if args.output else (in_path.parent / "labels_obb")
    out_dir.mkdir(parents=True, exist_ok=True)
    coco_json = in_path
    if not coco_json.is_file():
        files = list(in_path.glob("*.json"))
        if not files:
            print("no json")
            return
        coco_json = files[0]
    convert(coco_json, out_dir, args.normalize, args.angle_unit)
    print(str(out_dir))


if __name__ == "__main__":
    main()
