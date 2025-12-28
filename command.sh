python predict_obb.py --pt runs/rsar/yolo11n-obb_20251227_032355/weights/best.pt --source /home/tiger/codebase/DotaYOLO/datasets/RSAR_YOLO_OBB/test/images

python evaluate_obb.py \
    --gt datasets/RSAR_YOLO_OBB/val/labels \
    --pred datasets/RSAR_YOLO_OBB/val/labels \
    --num-classes 6 \
    --output evaluation_results.json