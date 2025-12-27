import math
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import cv2
except Exception:
    cv2 = None


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def norm_auto(vals: List[float]) -> bool:
    return max(vals) <= 1.5


def to_obb_from_cxwh_angle(cx: float, cy: float, w: float, h: float, angle_rad: float) -> List[Tuple[float, float]]:
    hw = w / 2.0
    hh = h / 2.0
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    pts = []
    for dx, dy in corners:
        x = cx + dx * ca - dy * sa
        y = cy + dx * sa + dy * ca
        pts.append((x, y))
    return pts


def write_label(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            f.write("")


def find_image_file(images_dir: Optional[Path], base: str) -> Optional[Path]:
    if images_dir is None:
        return None
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    for e in exts:
        p = images_dir / f"{base}{e}"
        if p.exists():
            return p
    return None


def min_area_rect(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if cv2 is None:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    try:
        import numpy as np
        arr = np.array(points, dtype=np.float32)
    except Exception:
        arr = None
    if arr is None or arr.ndim != 2 or arr.shape[1] != 2:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    rect = cv2.minAreaRect(arr)
    box = cv2.boxPoints(rect)
    return [(float(box[i, 0]), float(box[i, 1])) for i in range(4)]

