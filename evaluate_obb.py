import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


Point = Tuple[float, float]


@dataclass(frozen=True)
class Det:
    image_id: str
    cls: int
    pts: Tuple[Point, Point, Point, Point]
    conf: float


def _as_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _as_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _order_points_ccw(pts: Sequence[Point]) -> Tuple[Point, Point, Point, Point]:
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] != 4:
        raise ValueError("OBB must have 4 points")
    center = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - center[1], arr[:, 0] - center[0])
    idx = np.argsort(angles)
    arr = arr[idx]
    if _polygon_area(arr) < 0:
        arr = arr[::-1]
    return tuple((float(x), float(y)) for x, y in arr)  # type: ignore[return-value]


def _polygon_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_area_abs(pts: Sequence[Point]) -> float:
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 3:
        return 0.0
    return abs(_polygon_area(arr))


def _clip_polygon(subject: Sequence[Point], clipper: Sequence[Point]) -> List[Point]:
    def inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
        return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= 0.0

    def intersection(s: np.ndarray, e: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dc = a - b
        dp = s - e
        n1 = a[0] * b[1] - a[1] * b[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        if abs(float(denom)) < 1e-12:
            return e
        x = (n1 * dp[0] - n2 * dc[0]) / denom
        y = (n1 * dp[1] - n2 * dc[1]) / denom
        return np.array([x, y], dtype=np.float64)

    output = [np.array(p, dtype=np.float64) for p in subject]
    if not output:
        return []
    cp = [np.array(p, dtype=np.float64) for p in clipper]
    for i in range(len(cp)):
        input_list = output
        output = []
        a = cp[i]
        b = cp[(i + 1) % len(cp)]
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if inside(e, a, b):
                if not inside(s, a, b):
                    output.append(intersection(s, e, a, b))
                output.append(e)
            elif inside(s, a, b):
                output.append(intersection(s, e, a, b))
            s = e
    return [(float(p[0]), float(p[1])) for p in output]


def _intersection_area(poly1: Sequence[Point], poly2: Sequence[Point]) -> float:
    if cv2 is not None:
        p1 = np.asarray(poly1, dtype=np.float32).reshape(-1, 1, 2)
        p2 = np.asarray(poly2, dtype=np.float32).reshape(-1, 1, 2)
        try:
            area, _ = cv2.intersectConvexConvex(p1, p2)  # type: ignore[attr-defined]
            return float(area)
        except Exception:
            pass
    inter = _clip_polygon(poly1, poly2)
    return _polygon_area_abs(inter)


def obb_iou(pts1: Sequence[Point], pts2: Sequence[Point]) -> float:
    a1 = _polygon_area_abs(pts1)
    a2 = _polygon_area_abs(pts2)
    if a1 <= 0.0 or a2 <= 0.0:
        return 0.0
    inter = _intersection_area(pts1, pts2)
    if inter <= 0.0:
        return 0.0
    union = a1 + a2 - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _parse_label_line(line: str) -> Optional[Tuple[int, Tuple[Point, Point, Point, Point], float]]:
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    cls = _as_int(parts[0], default=-1)
    coords = [_as_float(x, default=float("nan")) for x in parts[1:9]]
    if any(math.isnan(v) for v in coords):
        return None
    pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
    conf = 1.0
    if len(parts) >= 10:
        conf = _as_float(parts[9], default=1.0)
    pts = _order_points_ccw(pts)
    return cls, pts, conf


def load_labels_dir(label_dir: Path, min_conf: float = 0.0) -> Dict[str, List[Det]]:
    out: Dict[str, List[Det]] = {}
    if not label_dir.exists():
        raise FileNotFoundError(str(label_dir))
    for p in sorted(label_dir.glob("*.txt")):
        image_id = p.stem
        dets: List[Det] = []
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        for line in text.splitlines():
            parsed = _parse_label_line(line)
            if parsed is None:
                continue
            cls, pts, conf = parsed
            if conf < min_conf:
                continue
            dets.append(Det(image_id=image_id, cls=cls, pts=pts, conf=conf))
        out[image_id] = dets
    return out


def _suffix_max(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    m = -np.inf
    for i in range(x.size - 1, -1, -1):
        v = float(x[i])
        if v > m:
            m = v
        out[i] = m
    return out


def ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    if rec.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    mpre = _suffix_max(mpre)
    recall_levels = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(mrec, recall_levels, side="left")
    idx = np.clip(idx, 0, mpre.size - 1)
    return float(np.mean(mpre[idx]))


def match_detections(
    preds: List[Det],
    gts_by_image: Dict[str, List[Det]],
    iou_thr: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    preds_sorted = sorted(preds, key=lambda d: d.conf, reverse=True)
    gt_pts: Dict[str, List[Tuple[Point, Point, Point, Point]]] = {}
    gt_matched: Dict[str, np.ndarray] = {}
    n_gt = 0
    for image_id, gts in gts_by_image.items():
        pts_list = [g.pts for g in gts]
        gt_pts[image_id] = pts_list
        gt_matched[image_id] = np.zeros(len(pts_list), dtype=bool)
        n_gt += len(pts_list)

    tp = np.zeros(len(preds_sorted), dtype=np.float64)
    fp = np.zeros(len(preds_sorted), dtype=np.float64)
    for i, d in enumerate(preds_sorted):
        pts_list = gt_pts.get(d.image_id)
        if not pts_list:
            fp[i] = 1.0
            continue
        matched = gt_matched[d.image_id]
        best_iou = 0.0
        best_j = -1
        for j, gt_pts_j in enumerate(pts_list):
            if matched[j]:
                continue
            iou = obb_iou(d.pts, gt_pts_j)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thr and best_j >= 0:
            tp[i] = 1.0
            matched[best_j] = True
        else:
            fp[i] = 1.0
    return tp, fp, n_gt


def pr_curve(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> Tuple[np.ndarray, np.ndarray]:
    if tp.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    rec = tp_c / max(n_gt, 1)
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-12)
    return rec, prec


def best_f1(rec: np.ndarray, prec: np.ndarray, confs: np.ndarray) -> Tuple[float, float, float, float]:
    if rec.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    f1 = 2.0 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    return float(prec[i]), float(rec[i]), float(f1[i]), float(confs[i])


def evaluate(
    gt_dir: Path,
    pred_dir: Path,
    iou_thrs: Sequence[float],
    min_conf: float,
    per_class: bool,
) -> Dict[str, object]:
    gts_all = load_labels_dir(gt_dir, min_conf=0.0)
    preds_all = load_labels_dir(pred_dir, min_conf=min_conf)
    image_ids = sorted(set(gts_all.keys()) | set(preds_all.keys()))

    gts_by_cls: Dict[int, Dict[str, List[Det]]] = {}
    preds_by_cls: Dict[int, List[Det]] = {}
    micro_gts_by_image: Dict[str, List[Det]] = {}
    micro_preds: List[Det] = []

    for image_id in image_ids:
        gts = gts_all.get(image_id, [])
        preds = preds_all.get(image_id, [])
        micro_gts_by_image[image_id] = gts
        micro_preds.extend(preds)
        for g in gts:
            gts_by_cls.setdefault(g.cls, {}).setdefault(image_id, []).append(g)
        for d in preds:
            preds_by_cls.setdefault(d.cls, []).append(d)

    classes = sorted(set(gts_by_cls.keys()) | set(preds_by_cls.keys()))

    ap50_by_cls: Dict[int, float] = {}
    ap_by_cls: Dict[int, float] = {}
    n_gt_by_cls: Dict[int, int] = {}
    n_pred_by_cls: Dict[int, int] = {}

    for cls in classes:
        cls_gts = gts_by_cls.get(cls, {})
        cls_preds = preds_by_cls.get(cls, [])
        n_gt = sum(len(v) for v in cls_gts.values())
        n_gt_by_cls[cls] = n_gt
        n_pred_by_cls[cls] = len(cls_preds)
        if n_gt == 0:
            ap50_by_cls[cls] = 0.0
            ap_by_cls[cls] = 0.0
            continue
        aps = []
        for t in iou_thrs:
            tp, fp, n_gt_t = match_detections(cls_preds, cls_gts, iou_thr=float(t))
            rec, prec = pr_curve(tp, fp, n_gt_t)
            aps.append(ap_from_pr(rec, prec))
        ap50_by_cls[cls] = float(aps[0])
        ap_by_cls[cls] = float(np.mean(aps))

    valid_classes = [c for c in classes if n_gt_by_cls.get(c, 0) > 0]
    map50 = float(np.mean([ap50_by_cls[c] for c in valid_classes])) if valid_classes else 0.0
    map5095 = float(np.mean([ap_by_cls[c] for c in valid_classes])) if valid_classes else 0.0

    tp, fp, n_gt = match_detections(micro_preds, micro_gts_by_image, iou_thr=float(iou_thrs[0]))
    rec, prec = pr_curve(tp, fp, n_gt)
    confs = np.array([d.conf for d in sorted(micro_preds, key=lambda d: d.conf, reverse=True)], dtype=np.float64)
    p_best, r_best, f1_best, conf_best = best_f1(rec, prec, confs)

    out: Dict[str, object] = {
        "num_images": len(image_ids),
        "num_classes": len(classes),
        "precision": p_best,
        "recall": r_best,
        "f1": f1_best,
        "conf_best": conf_best,
        "map50": map50,
        "map5095": map5095,
        "valid_classes": len(valid_classes),
    }
    if per_class:
        out["per_class"] = {
            "ap50": ap50_by_cls,
            "ap": ap_by_cls,
            "n_gt": n_gt_by_cls,
            "n_pred": n_pred_by_cls,
        }
    return out


def _parse_iou_thrs(spec: str) -> List[float]:
    s = spec.strip()
    if not s:
        return [0.5]
    if "," in s:
        vals = []
        for part in s.split(","):
            v = _as_float(part.strip(), default=float("nan"))
            if not math.isnan(v):
                vals.append(v)
        return vals if vals else [0.5]
    if ":" in s:
        a, b, c = (p.strip() for p in s.split(":", maxsplit=2))
        start = _as_float(a, default=0.5)
        stop = _as_float(b, default=0.95)
        step = _as_float(c, default=0.05)
        if step <= 0:
            return [start]
        vals = []
        v = start
        while v <= stop + 1e-9:
            vals.append(round(v, 10))
            v += step
        return vals
    v = _as_float(s, default=0.5)
    return [v]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--iou", type=str, default="0.5:0.95:0.05")
    parser.add_argument("--min-conf", type=float, default=0.001)
    parser.add_argument("--per-class", action="store_true")
    args = parser.parse_args()

    metrics = evaluate(
        gt_dir=Path(args.gt),
        pred_dir=Path(args.pred),
        iou_thrs=_parse_iou_thrs(args.iou),
        min_conf=float(args.min_conf),
        per_class=bool(args.per_class),
    )

    print(f"images: {metrics['num_images']}")
    print(f"classes: {metrics['num_classes']} (valid: {metrics['valid_classes']})")
    print(
        "precision: {:.6f}  recall: {:.6f}  f1: {:.6f}  conf*: {:.4f}".format(
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["conf_best"],
        )
    )
    print("mAP50: {:.6f}  mAP50-95: {:.6f}".format(metrics["map50"], metrics["map5095"]))

    if "per_class" in metrics:
        pc = metrics["per_class"]
        ap50 = pc["ap50"]
        ap = pc["ap"]
        n_gt = pc["n_gt"]
        n_pred = pc["n_pred"]
        keys = sorted(ap50.keys())
        print("per-class:")
        for k in keys:
            print(
                f"  {k}: n_gt={n_gt.get(k, 0)} n_pred={n_pred.get(k, 0)} "
                f"AP50={ap50.get(k, 0.0):.6f} AP={ap.get(k, 0.0):.6f}"
            )


if __name__ == "__main__":
    main()

