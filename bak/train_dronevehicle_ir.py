#!/usr/bin/env python3
"""
DroneVehicle红外图像 YOLO OBB 训练脚本

数据集: DroneVehicle IR (红外图像)
任务: 旋转目标检测 (OBB)
类别: car, truck, bus, van, freight_car
"""

import argparse
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO OBB model on DroneVehicle IR dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置训练
  python train_dronevehicle_ir.py
  
  # 使用yolo11x-obb预训练模型
  python train_dronevehicle_ir.py --model yolo11x-obb.pt --batch 8
  
  # 指定GPU和训练轮数
  python train_dronevehicle_ir.py --device 0 --epochs 200
  
  # 恢复训练
  python train_dronevehicle_ir.py --resume runs/dronevehicle_ir/xxx/weights/last.pt
        """
    )
    
    # 模型配置
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolo11n-obb.pt",
        help="模型配置文件或预训练权重 (默认: yolo11n-obb.pt)"
    )
    
    # 数据集配置
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="datasets/DroneVehicle_IR_YOLO_OBB/dronevehicle_ir.yaml",
        help="数据集配置文件"
    )
    
    # 训练参数
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="训练轮数 (默认: 100)"
    )
    
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=16,
        help="批次大小 (默认: 16)"
    )
    
    parser.add_argument(
        "--imgsz", "-i",
        type=int,
        default=640,
        help="图像尺寸 (默认: 640)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="设备 (0, 1, 2... 或 cpu) (默认: 0)"
    )
    
    # 优化器参数
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
        help="优化器 (默认: AdamW)"
    )
    
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="初始学习率 (默认: 0.01)"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="权重衰减 (默认: 0.0005)"
    )
    
    # 数据增强
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="启用数据增强 (默认: True)"
    )
    
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic增强概率 (默认: 1.0)"
    )
    
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.0,
        help="Mixup增强概率 (默认: 0.0)"
    )
    
    # 其他参数
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的checkpoint路径"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="数据加载工作进程数 (默认: 8)"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="runs/dronevehicle_ir",
        help="保存结果的项目目录"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="实验名称 (默认: 自动生成带时间戳的名称)"
    )
    
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="使用预训练权重 (默认: True)"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="早停耐心值 (默认: 50)"
    )
    
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="每N个epoch保存一次模型 (默认: 10)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查数据集配置文件
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 数据集配置文件不存在: {data_path}")
        print("请先运行转换脚本生成数据集")
        return
    
    # 生成实验名称
    if args.name:
        exp_name = args.name
    else:
        model_name = Path(args.model).stem.replace("-", "_").replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{model_name}_{timestamp}"
    
    print("=" * 60)
    print("DroneVehicle IR 红外图像 OBB 训练")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"数据集: {args.data}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch}")
    print(f"图像尺寸: {args.imgsz}")
    print(f"设备: {args.device}")
    print(f"优化器: {args.optimizer}")
    print(f"初始学习率: {args.lr0}")
    print(f"实验名称: {exp_name}")
    print("=" * 60)
    
    # 构建模型
    if args.resume:
        print(f"\n从检查点恢复训练: {args.resume}")
        model = YOLO(args.resume, task="obb")
    else:
        model = YOLO(args.model, task="obb")
    
    # 开始训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        augment=args.augment,
        mosaic=args.mosaic,
        mixup=args.mixup,
        workers=args.workers,
        project=args.project,
        name=exp_name,
        pretrained=args.pretrained,
        patience=args.patience,
        save_period=args.save_period,
        resume=args.resume is not None,
        exist_ok=True,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"结果保存在: {args.project}/{exp_name}")
    print(f"最佳模型: {args.project}/{exp_name}/weights/best.pt")
    
    # 验证最佳模型
    print("\n正在验证最佳模型...")
    best_model = YOLO(f"{args.project}/{exp_name}/weights/best.pt")
    metrics = best_model.val(data=args.data)
    
    print("\n验证结果:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()

