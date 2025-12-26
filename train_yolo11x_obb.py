"""
YOLO11x-OBB Training Script for DOTA Dataset
使用YOLO11最大版本进行训练，追求最佳精度
"""
import os
import torch
from ultralytics import YOLO

def train_yolo11x_obb(
    model_name='yolo11x-obb.pt',
    data_config='datasets/DOTAv1-split/dota.yaml',
    epochs=300,
    batch_size=8,  # X模型使用较小batch
    imgsz=1024,
    device='0,1',  # 使用双GPU
    project='runs/dota',
    name='yolo11x-obb',
    optimizer='SGD',
    lr0=0.008,  # YOLO11推荐稍低的学习率
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    patience=100,  # X模型需要更多耐心
    workers=8,
    save_period=10,
    **kwargs
):
    """
    训练YOLO11x-OBB模型
    
    参数说明:
    - model_name: 预训练模型名称 (yolo11x-obb.pt是最大版本)
    - data_config: 数据集配置文件路径
    - epochs: 训练轮数
    - batch_size: 批次大小
    - imgsz: 输入图像尺寸
    - device: GPU设备ID
    - optimizer: 优化器类型
    - lr0: 初始学习率 (YOLO11推荐稍低)
    - patience: 早停耐心值 (大模型需要更多轮次)
    """
    
    print("=" * 80)
    print("YOLO11x-OBB Training on DOTA Dataset")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Data: {data_config}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {imgsz}")
    print(f"Device: {device}")
    print(f"Optimizer: {optimizer}")
    print(f"Initial LR: {lr0}")
    print("=" * 80)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Warning: No GPU available, using CPU")
    
    # 检查数据集配置文件
    if not os.path.exists(data_config):
        print(f"Error: Data config file not found: {data_config}")
        return None
    
    # 加载模型
    print(f"\nLoading model: {model_name}")
    try:
        model = YOLO(model_name)
        print(f"Model loaded successfully: {model.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # 训练配置 - YOLO11优化参数
    train_args = {
        # 基础配置
        'data': data_config,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': False,
        'pretrained': True,
        'optimizer': optimizer,
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        
        # 学习率和优化器参数 (YOLO11优化)
        'lr0': lr0,
        'lrf': lrf,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'warmup_momentum': warmup_momentum,
        'warmup_bias_lr': warmup_bias_lr,
        
        # 损失函数权重 (YOLO11推荐)
        'box': box,
        'cls': cls,
        'dfl': dfl,
        
        # 数据增强参数 (针对DOTA优化)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,  # 旋转目标检测不需要额外旋转
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.2,  # 添加Mixup增强
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        
        # 其他参数
        'save': True,
        'save_period': save_period,
        'cache': False,  # 大数据集建议关闭cache
        'workers': workers,
        'patience': patience,
        'iou': 0.7,
        'max_det': 300,
    }
    
    # 更新用户自定义参数
    train_args.update(kwargs)
    
    # 开始训练
    print("\nStarting training...")
    print("=" * 80)
    
    try:
        results = model.train(**train_args)
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        return results
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # YOLO11x训练配置 - 追求最佳精度
    config = {
        'model_name': 'yolo11n-obb.pt',  # 最大版本，最高精度
        'data_config': '/home/shyue/codebase/datov1/datasets/DOTAv1-split-sub/dota_sub.yaml',
        'epochs': 300,  # DOTA数据集推荐300轮
        'batch_size': 8,  # X模型显存需求大，使用小batch
        'imgsz': 1024,  # DOTA标准尺寸
        'device': '0',  # 使用双GPU
        'project': 'runs/dota',
        'name': 'yolo11x-obb-swin-style-exp1',
        'optimizer': 'AdamW',  # AdamW适合Transformer架构
        'lr0': 0.001,  # 较低学习率适合AdamW
        'lrf': 0.01,  # 最终学习率=lr0*lrf
        'momentum': 0.937,
        'weight_decay': 0.05,  # 更高权重衰减
        'warmup_epochs': 3.0,
        'patience': 100,  # 大模型需要更多耐心
        'workers': 8,  # 数据加载线程
        'save_period': 10,  # 每10轮保存一次
    }
    
    # 开始训练
    results = train_yolo11x_obb(**config)
    
    if results:
        print("\n训练完成！查看结果:")
        print(f"  - 日志目录: runs/dota/{config['name']}")
        print(f"  - 最佳模型: runs/dota/{config['name']}/weights/best.pt")
        print(f"  - 最后模型: runs/dota/{config['name']}/weights/last.pt")
        print(f"  - 训练曲线: runs/dota/{config['name']}/results.png")