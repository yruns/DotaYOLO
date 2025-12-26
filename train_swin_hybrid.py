"""
YOLO11 + Swin-T OBB 训练脚本
使用 TorchVision 官方 Swin-T，无需自定义模块

使用方法: python train_swin_hybrid.py
"""

from pathlib import Path
from ultralytics import YOLO

# ============== 配置 ==============
PROJECT_ROOT = Path(__file__).parent

# 模型配置 - TorchVision Swin-T backbone
MODEL_YAML = PROJECT_ROOT / "models" / "yolo11_swin_obb.yaml"

# 数据集配置
DATA_YAML = "dota8.yaml"

# ============== 训练参数 ==============
if __name__ == "__main__":
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"模型配置: {MODEL_YAML}")
    print(f"数据集配置: {DATA_YAML}")
    
    # 验证配置文件存在
    if not MODEL_YAML.exists():
        raise FileNotFoundError(f"模型配置文件不存在: {MODEL_YAML}")
    
    # 创建模型
    model = YOLO(str(MODEL_YAML), task="obb")
    
    # 打印模型信息
    print("\n模型结构:")
    for i, layer in enumerate(model.model.model):
        layer_type = type(layer).__name__
        print(f"  Layer {i:2d}: {layer_type}")
    
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 开始训练
    print("\n开始训练...")
    results = model.train(
        data=DATA_YAML,
        
        # === 设备 ===
        device="cpu",
        
        # === 训练参数 ===
        epochs=2,
        batch=1,
        imgsz=640,
        
        # === 保存设置 ===
        project=str(PROJECT_ROOT / "runs" / "obb"),
        name="yolo11_swin",
        save=True,
        
        # === 学习率 ===
        lr0=0.01,
        lrf=0.01,
        
        # === 数据增强 ===
        degrees=0,
        mosaic=0.0,
        
        # === 其他 ===
        workers=0,
        verbose=True,
        plots=True,
    )
    
    print("\n训练完成！")
    print(f"结果保存在: {results.save_dir}")
