#!/usr/bin/env python3
"""
YOLO11-Swin-MultiScale-OBB 训练脚本
使用 Swin Transformer 多尺度特征提取，解决特征衰减问题
"""

import sys
import argparse

# 使用本地 ultralytics 代码
sys.path.insert(0, '/home/shyue/codebase/datov1/ultralytics')

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Train YOLO11-Swin-MultiScale-OBB')
    parser.add_argument('--model', type=str, default='models/yolo11n_swin_multiscale_obb.yaml')
    parser.add_argument('--data', type=str, default='datasets/DOTAv1-split-sub/dota_sub.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--lr0', type=float, default=0.0001)
    parser.add_argument('--imgsz', type=int, default=1024)
    args = parser.parse_args()
    
    print("=" * 80)
    print("YOLO11-Swin-MultiScale-OBB 训练")
    print("使用 Swin Transformer 多尺度特征，解决特征衰减问题")
    print("=" * 80)
    print()
    print(f"模型: {args.model}")
    print(f"数据: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Learning Rate: {args.lr0}")
    print()
    
    # 加载模型
    print("加载模型...")
    model = YOLO(args.model)
    print("✅ 模型加载成功")
    
    # 参数统计
    total = sum(p.numel() for p in model.model.parameters())
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"参数: {total:,} (可训练: {trainable:,})")
    print()
    
    # 开始训练
    print("开始训练...")
    print("=" * 80)
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        
        # 优化器设置
        optimizer='AdamW',
        lr0=args.lr0,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # 损失权重
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # 数据增强
        degrees=0.0,  # OBB 不使用旋转增强
        mosaic=1.0,
        mixup=0.15,
        
        # 其他
        cos_lr=True,
        amp=True,
        patience=50,
        
        # 保存设置
        name='yolo11_swin_multiscale_obb',
        project='runs/obb',
        exist_ok=True,
        save_period=10,
    )
    
    print()
    print("=" * 80)
    print("训练完成!")
    print("=" * 80)
    
    if hasattr(results, 'metrics') and hasattr(results.metrics, 'box'):
        print(f"最终指标:")
        print(f"  mAP50: {results.metrics.box.map50:.4f}")
        print(f"  mAP50-95: {results.metrics.box.map:.4f}")


if __name__ == "__main__":
    main()

