"""
YOLO11 OBB CPU 训练脚本
使用方法: python train_yolo11_obb_cpu.py
"""

from ultralytics import YOLO

# ============== 配置 ==============
# 模型配置 (n/s/m/l/x 选择一个)
# 注意：文件名中的 n/s/m/l/x 决定模型大小
MODEL_YAML = "/home/shyue/codebase/datov1/models/yolo11n-obb.yaml"  # nano 版本，最小
# MODEL_YAML = "/home/shyue/codebase/datov1/models/yolo11s-obb.yaml"  # small
# MODEL_YAML = "/home/shyue/codebase/datov1/models/yolo11m-obb.yaml"  # medium  
# MODEL_YAML = "/home/shyue/codebase/datov1/models/yolo11l-obb.yaml"  # large
# MODEL_YAML = "/home/shyue/codebase/datov1/models/yolo11x-obb.yaml"  # extra-large

# 数据集配置
DATA_YAML = "DOTAv1.yaml"  # 或者你自己的数据集 yaml 路径

# ============== 训练参数 ==============
if __name__ == "__main__":
    # 从 yaml 创建模型（从头训练）
    model = YOLO(MODEL_YAML, task="obb")
    
    # 或者从预训练权重开始（推荐，收敛更快）
    # model = YOLO("yolo11n-obb.pt")
    
    # 开始训练
    results = model.train(
        data=DATA_YAML,
        
        # === 设备 ===
        device="cpu",           # 使用 CPU 训练
        
        # === 训练参数 ===
        epochs=100,             # 训练轮数
        batch=2,                # CPU 训练建议小 batch
        imgsz=640,              # 图像尺寸 (CPU 建议用小尺寸)
        
        # === 保存设置 ===
        project="runs/obb",
        name="yolo11_obb_cpu",
        save=True,
        save_period=10,         # 每10轮保存一次
        
        # === 学习率 ===
        lr0=0.01,               # 初始学习率
        lrf=0.01,               # 最终学习率比例
        
        # === 数据增强 (OBB 特有) ===
        degrees=180,            # 旋转角度范围
        flipud=0.5,             # 上下翻转概率
        fliplr=0.5,             # 左右翻转概率
        mosaic=1.0,             # 马赛克增强
        
        # === 其他 ===
        workers=0,              # CPU 训练建议 workers=0
        verbose=True,
        plots=True,
    )
    
    print("训练完成！")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")

