#!/usr/bin/env python3
"""
VOC XML → YOLO-OBB 转换器

输入:
- VOC XML (支持 roLabelImg 的 `robndbox` 以及常规 `bndbox`)
- XML 关键字段示例:
  <annotation>
    <size><width>1024</width><height>768</height></size>
    <object>
      <name>ship</name>
      <robndbox>
        <cx>512</cx><cy>384</cy><w>200</w><h>80</h><angle>30</angle>
      </robndbox>
    </object>
  </annotation>

输出:
- 每图一份 `.txt`，每行: `class_id x1 y1 x2 y2 x3 y3 x4 y4`，坐标归一化到 0–1

示例输出:
- 0 0.420000 0.500000 0.580000 0.400000 0.620000 0.600000 0.460000 0.700000

示例命令:
- python -m convert.voc_to_yolo_obb --input /data/ann_xml --output /data/labels_obb --normalize auto --angle-unit auto
"""
import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from convert.common import clamp01, norm_auto, to_obb_from_cxwh_angle, write_label


def convert_file(xml_path: Path, out_dir: Path, normalize: str, angle_unit: str, name_to_idx: Optional[Dict[str, int]]) -> None:
    root = ET.parse(str(xml_path)).getroot()
    fn = root.find("filename")
    base = xml_path.stem if fn is None else Path(fn.text or "").stem
    size = root.find("size")
    if size is None:
        return
    iw = int(size.findtext("width", default="0"))
    ih = int(size.findtext("height", default="0"))
    if iw <= 0 or ih <= 0:
        return
    lines: List[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if not name:
            continue
        idx = 0
        if name_to_idx is not None:
            idx = int(name_to_idx.get(name, 0))
        bnd = obj.find("robndbox")
        if bnd is not None:
            cx = float(bnd.findtext("cx", default="0"))
            cy = float(bnd.findtext("cy", default="0"))
            w = float(bnd.findtext("w", default="0"))
            h = float(bnd.findtext("h", default="0"))
            ang = float(bnd.findtext("angle", default="0"))
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
            continue
        hb = obj.find("bndbox")
        if hb is not None:
            xmin = float(hb.findtext("xmin", default="0"))
            ymin = float(hb.findtext("ymin", default="0"))
            xmax = float(hb.findtext("xmax", default="0"))
            ymax = float(hb.findtext("ymax", default="0"))
            is_norm = normalize == "yes" or (normalize == "auto" and norm_auto([xmin, ymin, xmax, ymax]))
            if is_norm:
                xmin *= iw
                ymin *= ih
                xmax *= iw
                ymax *= ih
            pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
            coords: List[float] = []
            for x, y in pts:
                coords.extend([clamp01(x / iw), clamp01(y / ih)])
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
    out_dir = Path(args.output) if args.output else (in_path / "labels_obb")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [in_path] if in_path.is_file() and in_path.suffix.lower() == ".xml" else list(in_path.glob("*.xml"))
    names: List[str] = []
    for xp in files:
        try:
            root = ET.parse(str(xp)).getroot()
            for obj in root.findall("object"):
                nn = obj.findtext("name")
                if nn:
                    names.append(nn)
        except Exception:
            pass
    uniq: List[str] = []
    s = set()
    for n in names:
        if n not in s:
            s.add(n)
            uniq.append(n)
    name_to_idx = {n: i for i, n in enumerate(uniq)}
    for xp in files:
        convert_file(xp, out_dir, args.normalize, args.angle_unit, name_to_idx)
    print(str(out_dir))


if __name__ == "__main__":
    main()
