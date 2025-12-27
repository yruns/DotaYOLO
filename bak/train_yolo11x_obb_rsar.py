"""
YOLO11x-OBB Training Script for RSAR Dataset
使用YOLO11最大版本训练RSAR旋转目标检测数据集
"""
import os
import torch
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def train_yolo11x_obb_rsar(
    model_name='yolo11x-obb.pt',
    data_config='datasets/RSAR_YOLO_OBB/rsar.yaml',
    epochs=300,
    batch_size=8,
    imgsz=1024,
    device='0,1',
    project='runs/rsar',
    name=None,
    lr0=0.001,
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
    - lr0: 初始学习率
    - patience: 早停耐心值
    """
    
    print("=" * 80)
    print("YOLO11x-OBB Training on RSAR Dataset")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Data: {data_config}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {imgsz}")
    print(f"Device: {device}")
    print(f"Initial LR: {lr0}")
    print("=" * 80)
    
    SETTINGS.update({'wandb': True})
    try:
        import wandb  # noqa: F401
        print("已启用 Weights & Biases 日志")
    except Exception:
        print("未检测到 wandb 包，已跳过 W&B 日志。请运行: pip install wandb")

    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Warning: No GPU available, using CPU")
    
    if not os.path.exists(data_config):
        print(f"Error: Data config file not found: {data_config}")
        return None
    
    print(f"\nLoading model: {model_name}")
    try:
        model = YOLO(model_name)
        print(f"Model loaded successfully: {model.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    train_args = {
        'data': data_config,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'device': device,
        'project': project,
        'name': name,
        'pretrained': True,
        # 'verbose': True,
        'seed': 0,
    }
    
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
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    config = {
        'model_name': 'yolo11x-obb.pt',
        'data_config': 'datasets/RSAR_YOLO_OBB/rsar.yaml',
        'epochs': 100,
        'batch_size': 32,
        'imgsz': 512,
        'device': '0',
        'project': 'runs/rsar',
        'name': 'yolo11x-obb_' + timestamp,
    }
    
    results = train_yolo11x_obb_rsar(**config)
    
    if results:
        print("\n训练完成！查看结果:")
        print(f"  - 日志目录: runs/rsar/{config['name']}")
        print(f"  - 最佳模型: runs/rsar/{config['name']}/weights/best.pt")
        print(f"  - 最后模型: runs/rsar/{config['name']}/weights/last.pt")
        print(f"  - 训练曲线: runs/rsar/{config['name']}/results.png")
