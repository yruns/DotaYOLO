import argparse
import sys
import csv
import os
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import cv2

def gather_images(root):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    p = Path(root)
    files = [str(x) for x in p.rglob('*') if x.is_file() and x.suffix.lower() in exts]
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    # 输出固定到 predictions/<timestamp>/ 下，包含 images/、labels/、log.csv
    args = parser.parse_args()

    imgs = gather_images(args.source)
    if not imgs:
        print(f'No images found under {args.source}')
        sys.exit(1)

    model = YOLO(args.pt)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dest_root = Path('predictions') / timestamp
    dest_images = dest_root / 'images'
    dest_labels = dest_root / 'labels'
    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_labels, exist_ok=True)

    log_path = dest_root / 'log.csv'
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image','width','height','num_dets','counts','preprocess_ms','inference_ms','postprocess_ms','saved_image','saved_label'])
        writer.writeheader()
        for im in imgs[:10]:
            print(f"Predicting {im}")
            results = model.predict(source=im, conf=args.conf, batch=1, verbose=False)
            r = results[0]
            sp = getattr(r, 'speed', {}) or {}
            h, w = r.orig_shape

            # 统计类别数量
            cls = None
            if hasattr(r, 'obb') and r.obb is not None and hasattr(r.obb, 'cls') and r.obb.cls is not None:
                cls = r.obb.cls
            elif hasattr(r, 'boxes') and r.boxes is not None and hasattr(r.boxes, 'cls') and r.boxes.cls is not None:
                cls = r.boxes.cls
            num_dets = int(cls.shape[0]) if cls is not None else 0
            counts_map = {}
            if cls is not None:
                for i in cls.int().cpu().tolist():
                    name = model.names.get(int(i), str(int(i))) if isinstance(model.names, dict) else str(int(i))
                    counts_map[name] = counts_map.get(name, 0) + 1
            counts = '; '.join([f"{k}:{v}" for k, v in counts_map.items()])

            saved_image = dest_images / Path(im).name
            try:
                plotted = r.plot()
                cv2.imwrite(str(saved_image), plotted)
            except Exception:
                saved_image = None

            saved_label = dest_labels / (Path(im).stem + '.txt')
            try:
                if hasattr(r, 'obb') and r.obb is not None:
                    polys = getattr(r.obb, 'xyxyxyxy', None)
                    cls_tensor = getattr(r.obb, 'cls', None)
                    if polys is not None and cls_tensor is not None:
                        arr = polys.cpu().numpy()
                        cls_list = cls_tensor.int().cpu().tolist()
                        with open(saved_label, 'w') as lf:
                            for jj, ci in enumerate(cls_list):
                                coords = []
                                p = arr[jj]
                                for k in range(4):
                                    x = p[k][0] / w
                                    y = p[k][1] / h
                                    coords.extend([x, y])
                                lf.write(str(int(ci)) + ' ' + ' '.join(f"{v:.6f}" for v in coords) + '\n')
                else:
                    saved_label = None
            except Exception:
                saved_label = None

            writer.writerow({
                'image': im,
                'width': w,
                'height': h,
                'num_dets': num_dets,
                'counts': counts,
                'preprocess_ms': sp.get('preprocess'),
                'inference_ms': sp.get('inference'),
                'postprocess_ms': sp.get('postprocess'),
                'saved_image': str(saved_image) if saved_image else '',
                'saved_label': str(saved_label) if saved_label else '',
            })
    print(f"Images saved to {dest_images}")
    print(f"Labels saved to {dest_labels}")
    print(f"Log saved to {log_path}")

if __name__ == '__main__':
    main()
