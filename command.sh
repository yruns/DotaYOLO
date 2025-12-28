#!/bin/bash

python evaluate_obb.py \
  --gt datasets/RSAR_YOLO_OBB/val/labels \
  --pred /home/tiger/codebase/DotaYOLO/runs/obb/predict4/labels \
  --num-classes 6

python predict_obb.py \
    --pt runs/rsar/yolo11x-obb_20251227_132037/weights/best.pt \
    --source data_samples/RSAR_YOLO_OBB/train/images 

yolo obb val model=runs/rsar/yolo11x-obb_20251227_132037/weights/best.pt data=datasets/RSAR_YOLO_OBB/rsar.yaml

yolo obb val model=runs/rsar/yolo11x-obb_20251227_132037/weights/best.pt data=/home/tiger/codebase/DotaYOLO/data_samples/RSAR_YOLO_OBB/rsar.yaml

yolo obb predict \
  model=runs/rsar/yolo11x-obb_20251227_132037/weights/best.pt \
  source=/home/tiger/codebase/DotaYOLO/data_samples/RSAR_YOLO_OBB/val/images \
  save=True \
  save_txt=True \
  save_conf=True

yolo obb predict \
  model=runs/rsar/yolo11x-obb_20251227_132037/weights/best.pt \
  source=/home/tiger/codebase/DotaYOLO/datasets/RSAR_YOLO_OBB/val/images \
  save=True \
  save_txt=True \
  save_conf=True


# YOLO CLI 命令集合
# 语法: yolo TASK MODE ARGS
# TASK: detect, segment, classify, pose, obb
# MODE: train, val, predict, export, track, benchmark
# ARGS: 各种参数如 model=, data=, epochs=, device= 等

# ==================== 训练命令 ====================
# 训练 OBB 模型
# yolo obb train data=datasets/RSAR_YOLO_OBB/rsar.yaml model=yolo11n-obb.pt epochs=100 imgsz=1024 device=0

# 恢复训练（使用 last.pt）
# yolo obb train resume model=runs/rsar/yolo11n-obb_20251227_032355/weights/last.pt device=0

# 恢复训练（使用 CPU）
# yolo obb train resume model=/home/tiger/codebase/DotaYOLO/runs/dronevehicle/yolo11x-obb_20251227_163846/weights/last.pt device=cpu

# 训练 OBB 模型（使用预训练权重）
# yolo obb train data=datasets/RSAR_YOLO_OBB/rsar.yaml model=yolo11x-obb.pt epochs=300 imgsz=1024 batch=8 device=0,1

# 训练检测模型
# yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640

# 训练分割模型
# yolo segment train data=coco8-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640

# 训练分类模型
# yolo classify train data=imagenet8 model=yolo11n-cls.pt epochs=100 imgsz=224

# 训练姿态估计模型
# yolo pose train data=coco8-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640

# ==================== 验证命令 ====================
# 验证 OBB 模型
# yolo obb val model=runs/rsar/yolo11x-obb_20251227_132037/weights/best.pt data=datasets/RSAR_YOLO_OBB/rsar.yaml imgsz=512 batch=16

# 验证检测模型
# yolo detect val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640

# 验证自定义模型
# yolo detect val model=path/to/best.pt

# ==================== 预测命令 ====================
# OBB 模型预测（单张图片）
# yolo obb predict model=runs/rsar/yolo11n-obb_20251227_032355/weights/best.pt source=/home/tiger/codebase/DotaYOLO/datasets/RSAR_YOLO_OBB/test/images

# OBB 模型预测（文件夹）
# yolo obb predict model=runs/rsar/yolo11n-obb_20251227_032355/weights/best.pt source=datasets/RSAR_YOLO_OBB/test/images save=True

# 检测模型预测（图片）
# yolo detect predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'

# 检测模型预测（文件夹）
# yolo detect predict model=yolo11n.pt source=path/to/images save=True

# 检测模型预测（视频）
# yolo detect predict model=yolo11n.pt source=path/to/video.mp4

# 检测模型预测（摄像头）
# yolo detect predict model=yolo11n.pt source=0 show=True

# ==================== 导出命令 ====================
# 导出 OBB 模型为 ONNX 格式
# yolo obb export model=runs/rsar/yolo11n-obb_20251227_032355/weights/best.pt format=onnx

# 导出 OBB 模型为 TensorRT 引擎格式
# yolo obb export model=runs/rsar/yolo11n-obb_20251227_032355/weights/best.pt format=engine device=0

# 导出 OBB 模型为 ONNX 格式（动态尺寸）
# yolo obb export model=runs/rsar/yolo11n-obb_20251227_032355/weights/best.pt format=onnx dynamic=True

# 导出检测模型为 ONNX 格式
# yolo detect export model=yolo11n.pt format=onnx imgsz=640

# 导出分类模型为 ONNX 格式
# yolo classify export model=yolo11n-cls.pt format=onnx imgsz=224,128

# ==================== 跟踪命令 ====================
# OBB 模型跟踪（视频）
# yolo obb track model=runs/rsar/yolo11n-obb_20251227_032355/weights/best.pt source=video.mp4 show=True

# 检测模型跟踪（视频）
# yolo detect track model=yolo11n.pt source='https://youtu.be/LNwODJXcvt4' show=True

# 检测模型跟踪（使用 ByteTrack）
# yolo detect track model=yolo11n.pt source='https://youtu.be/LNwODJXcvt4' show=True tracker=bytetrack.yaml

# 检测模型跟踪（摄像头）
# yolo detect track model=yolo11n.pt source=0 show=True

# ==================== 基准测试命令 ====================
# OBB 模型基准测试
# yolo obb benchmark model=runs/rsar/yolo11n-obb_20251227_032355/weights/best.pt data=datasets/RSAR_YOLO_OBB/rsar.yaml

# 检测模型基准测试
# yolo detect benchmark model=yolo11n.pt data=coco8.yaml

# ==================== 特殊命令 ====================
# 查看 YOLO 帮助信息
# yolo help

# 查看 YOLO 版本
# yolo version

# 查看 YOLO 设置
# yolo settings

# 运行 YOLO 检查
# yolo checks

# 查看 YOLO 配置
# yolo cfg

# ==================== Python 脚本命令 ====================
# 使用 Python 脚本进行预测
python predict_obb.py --pt runs/rsar/yolo11n-obb_20251227_032355/weights/best.pt --source /home/tiger/codebase/DotaYOLO/datasets/RSAR_YOLO_OBB/test/images

# 使用 Python 脚本进行评估
python evaluate_obb.py \
    --gt datasets/RSAR_YOLO_OBB/val/labels \
    --pred datasets/RSAR_YOLO_OBB/val/labels \
    --num-classes 6 \
    --output evaluation_results.json