import os
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='n')
    parser.add_argument('--data', type=str, default='datasets/DroneVehicle_IR_YOLO_OBB/dronevehicle_ir.yaml')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    m = args.model.strip().lower()
    short = {'n': 'yolo11n-obb', 'm': 'yolo11m-obb', 'l': 'yolo11l-obb', 'x': 'yolo11x-obb'}
    base = short.get(m, m)
    model_name = base if base.endswith('.pt') else base + '.pt'
    data_yaml = args.data

    if not Path(data_yaml).exists():
        print(f"Error: dataset config not found: {data_yaml}")
        return

    SETTINGS.update({'wandb': True})
    model = YOLO(model_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{Path(model_name).stem}_{timestamp}"
    model.train(
        data=data_yaml,
        batch=args.batchsize,
        imgsz=args.imgsz,
        device=args.device,
        project='runs/dronevehicle',
        name=exp_name,
        exist_ok=True,
    )

if __name__ == '__main__':
    main()
