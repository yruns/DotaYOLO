import argparse
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import json
import cv2
from tqdm import tqdm

def parse_obb_line(line: str) -> Tuple[int, np.ndarray, float]:
    parts = line.strip().split()
    if len(parts) == 10:
        class_id = int(parts[0])
        coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32)
        confidence = float(parts[9])
    elif len(parts) == 9:
        class_id = int(parts[0])
        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        confidence = 1.0
    else:
        raise ValueError(f"Invalid OBB line format: {line}")
    return class_id, coords, confidence

def load_labels(label_dir: str, desc: str = "Loading labels") -> Dict[str, List[Tuple[int, np.ndarray, float]]]:
    labels = {}
    label_path = Path(label_dir)
    txt_files = list(label_path.glob('*.txt'))
    
    for txt_file in tqdm(txt_files, desc=desc, unit="file"):
        image_id = txt_file.stem
        boxes = []
        with open(txt_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        class_id, coords, conf = parse_obb_line(line)
                        boxes.append((class_id, coords, conf))
                    except ValueError as e:
                        print(f"Warning: {e}")
        labels[image_id] = boxes
    return labels

def polygon_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    p1 = poly1.reshape(-1, 2).astype(np.float32)
    p2 = poly2.reshape(-1, 2).astype(np.float32)
    
    try:
        inter_area = cv2.intersectConvexConvex(p1, p2)[0]
        area1 = cv2.contourArea(p1)
        area2 = cv2.contourArea(p2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    except:
        return 0.0

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_class(gt_boxes: List[Tuple[str, np.ndarray]], 
                   pred_boxes: List[Tuple[str, np.ndarray, float]],
                   iou_threshold: float = 0.5) -> Tuple[float, float, float, int, int, int]:
    num_gt = len(gt_boxes)
    if num_gt == 0:
        return 0.0, 0.0, 0.0, 0, 0, 0
    
    pred_sorted = sorted(pred_boxes, key=lambda x: -x[2])
    
    tp_list = []
    fp_list = []
    matched_gt = set()
    
    for pred_id, pred_box, conf in pred_sorted:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, (gt_id, gt_box) in enumerate(gt_boxes):
            if gt_idx in matched_gt or gt_id != pred_id:
                continue
            iou = polygon_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt.add(best_gt_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)
    
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    ap = compute_ap(recalls, precisions)
    
    tp = int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0
    fp = int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0
    fn = num_gt - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1, tp, fp, fn, ap

def compute_map(gt_labels: Dict[str, List[Tuple[int, np.ndarray, float]]],
                pred_labels: Dict[str, List[Tuple[int, np.ndarray, float]]],
                iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                num_classes: int = 6) -> Dict:
    ap_per_class_per_iou = {iou_th: [] for iou_th in iou_thresholds}
    class_stats = {}
    
    for class_id in tqdm(range(num_classes), desc="Evaluating classes", unit="class"):
        gt_boxes = []
        pred_boxes = []
        
        for image_id in gt_labels:
            gt_boxes_class = [(image_id, box) for cls, box, conf in gt_labels[image_id] if cls == class_id]
            pred_boxes_class = [(image_id, box, conf) for cls, box, conf in pred_labels.get(image_id, []) if cls == class_id]
            
            gt_boxes.extend(gt_boxes_class)
            pred_boxes.extend(pred_boxes_class)
        
        class_metrics = {}
        
        for iou_threshold in iou_thresholds:
            precision, recall, f1, tp, fp, fn, ap = evaluate_class(gt_boxes, pred_boxes, iou_threshold)
            
            if iou_threshold == 0.5:
                class_metrics['iou_50'] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                }
            
            ap_per_class_per_iou[iou_threshold].append(ap)
        
        class_stats[class_id] = class_metrics
    
    map_50 = np.mean(ap_per_class_per_iou[0.5]) if ap_per_class_per_iou[0.5] else 0.0
    map_50_95 = np.mean([np.mean(aps) if aps else 0.0 for aps in ap_per_class_per_iou.values()])
    
    overall_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'tp': 0,
        'fp': 0,
        'fn': 0
    }
    
    for class_id in class_stats:
        if 'iou_50' in class_stats[class_id]:
            m = class_stats[class_id]['iou_50']
            overall_metrics['precision'] += m['precision']
            overall_metrics['recall'] += m['recall']
            overall_metrics['f1'] += m['f1']
            overall_metrics['tp'] += m['tp']
            overall_metrics['fp'] += m['fp']
            overall_metrics['fn'] += m['fn']
    
    num_valid_classes = len([c for c in class_stats if 'iou_50' in class_stats[c]])
    if num_valid_classes > 0:
        overall_metrics['precision'] /= num_valid_classes
        overall_metrics['recall'] /= num_valid_classes
        overall_metrics['f1'] /= num_valid_classes
    
    return {
        'class_stats': class_stats,
        'overall': overall_metrics,
        'mAP@0.5': float(map_50),
        'mAP@0.5:0.95': float(map_50_95),
        'ap_per_iou': {str(k): float(np.mean(v) if v else 0.0) for k, v in ap_per_class_per_iou.items()}
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO OBB predictions')
    parser.add_argument('--gt', type=str, required=True, help='Ground truth labels directory')
    parser.add_argument('--pred', type=str, required=True, help='Predicted labels directory')
    parser.add_argument('--num-classes', type=int, default=6, help='Number of classes')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output JSON file')
    args = parser.parse_args()
    
    print("Loading ground truth labels...")
    gt_labels = load_labels(args.gt, desc="Loading ground truth")
    print(f"Loaded {len(gt_labels)} ground truth files")
    
    print("Loading predicted labels...")
    pred_labels = load_labels(args.pred, desc="Loading predictions")
    print(f"Loaded {len(pred_labels)} prediction files")
    
    print("\nComputing metrics...")
    results = compute_map(gt_labels, pred_labels, num_classes=args.num_classes)
    
    class_names = ['ship', 'bridge', 'car', 'tank', 'aircraft', 'harbor']
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nOverall Metrics (IoU=0.5):")
    print(f"  Precision: {results['overall']['precision']:.4f}")
    print(f"  Recall:    {results['overall']['recall']:.4f}")
    print(f"  F1-Score:  {results['overall']['f1']:.4f}")
    print(f"  TP: {results['overall']['tp']}, FP: {results['overall']['fp']}, FN: {results['overall']['fn']}")
    
    print(f"\nmAP Metrics:")
    print(f"  mAP@0.5:     {results['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}")
    
    print(f"\nPer-Class Metrics (IoU=0.5):")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-"*72)
    for class_id in range(args.num_classes):
        if class_id in results['class_stats'] and 'iou_50' in results['class_stats'][class_id]:
            stats = results['class_stats'][class_id]['iou_50']
            name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            print(f"{name:<12} {stats['precision']:<12.4f} {stats['recall']:<12.4f} {stats['f1']:<12.4f} {stats['tp']:<6} {stats['fp']:<6} {stats['fn']:<6}")
    
    print(f"\nAP at different IoU thresholds:")
    print(f"{'IoU':<10} {'AP':<12}")
    print("-"*22)
    for iou_th in sorted([float(k) for k in results['ap_per_iou'].keys()]):
        print(f"{iou_th:<10.2f} {results['ap_per_iou'][str(iou_th)]:<12.4f}")
    
    output_results = {
        'overall_metrics': results['overall'],
        'map_metrics': {
            'mAP@0.5': results['mAP@0.5'],
            'mAP@0.5:0.95': results['mAP@0.5:0.95']
        },
        'class_metrics': results['class_stats'],
        'ap_per_iou': results['ap_per_iou'],
        'class_names': class_names
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main()
