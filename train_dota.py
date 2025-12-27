import os
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['yolo11n-obb', 'yolo11x-obb'], default='yolo11n-obb')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    model_name = args.model if args.model.endswith('.pt') else args.model + '.pt'
    data_yaml = 'datasets/DOTAv1-split/dota.yaml'

    if not Path(data_yaml).exists():
        print(f"Error: dataset config not found: {data_yaml}")
        return

    SETTINGS.update({'wandb': True})
    model = YOLO(model_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.model}_{timestamp}"
    model.train(
        data=data_yaml,
        batch=args.batchsize,
        imgsz=args.imgsz,
        device=args.device,
        project='runs/dota',
        name=exp_name,
        exist_ok=True,
    )

if __name__ == '__main__':
    main()
