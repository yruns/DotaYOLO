"""
YOLO11 Swin Transformer Stage OBB 训练脚本
"""

import argparse
from datetime import datetime
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11 Swin Stage OBB model")
    parser.add_argument("--model", type=str, default="models/yolo11_swin_stage_obb.yaml", help="模型配置文件")
    parser.add_argument("--data", type=str, default="datasets/DOTAv1-split/dota.yaml", help="数据集配置文件")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=8, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--device", type=str, default="0", help="设备 (0, 1, 2... 或 cpu)")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 生成带时间戳的实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"swin_stage_obb_{timestamp}"
    
    # 构建模型
    model = YOLO(args.model, task="obb")
    
    # 开始训练
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume,
        project="runs/dota",
        name=exp_name,
    )
