#!/usr/bin/env python3
"""
创建 DOTA 数据集的子集用于快速测试
从完整的 DOTAv1-split 数据集中提取指定数量的样本
"""
import os
import glob
import shutil
from pathlib import Path

# 配置
source_path = "datasets/DOTAv1-split"
target_path = "datasets/DOTAv1-split-sub"

nums_train = 3000  # 训练集样本数
nums_val = 1000    # 验证集样本数

print("=" * 80)
print("创建 DOTA 数据集子集")
print("=" * 80)
print(f"源数据集: {source_path}")
print(f"目标数据集: {target_path}")
print(f"训练样本数: {nums_train}")
print(f"验证样本数: {nums_val}")
print("=" * 80)

# 创建目标目录
os.makedirs(target_path, exist_ok=True)
os.makedirs(os.path.join(target_path, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(target_path, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(target_path, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(target_path, "labels", "val"), exist_ok=True)

print("\n处理训练集...")
train_files = sorted(glob.glob(os.path.join(source_path, "images", "train", "*.jpg")))
print(f"找到 {len(train_files)} 个训练图像")

copied_train = 0
for idx, img_file in enumerate(train_files):
    if idx >= nums_train:
        break
    
    # 获取文件名（不含路径）
    img_basename = os.path.basename(img_file)
    label_file = img_file.replace("images", "labels").replace(".jpg", ".txt")
    
    # 检查标签文件是否存在
    if not os.path.exists(label_file):
        print(f"  警告: 标签文件不存在，跳过: {label_file}")
        continue
    
    # 复制文件（保持原始文件名）
    target_img = os.path.join(target_path, "images", "train", img_basename)
    target_label = os.path.join(target_path, "labels", "train", img_basename.replace(".jpg", ".txt"))
    
    shutil.copy(img_file, target_img)
    shutil.copy(label_file, target_label)
    copied_train += 1
    
    if (idx + 1) % 500 == 0:
        print(f"  已复制 {idx + 1}/{nums_train} 个训练样本")

print(f"✓ 训练集完成: 成功复制 {copied_train} 个样本")

print("\n处理验证集...")
val_files = sorted(glob.glob(os.path.join(source_path, "images", "val", "*.jpg")))
print(f"找到 {len(val_files)} 个验证图像")

copied_val = 0
for idx, img_file in enumerate(val_files):
    if idx >= nums_val:
        break
    
    # 获取文件名（不含路径）
    img_basename = os.path.basename(img_file)
    label_file = img_file.replace("images", "labels").replace(".jpg", ".txt")
    
    # 检查标签文件是否存在
    if not os.path.exists(label_file):
        print(f"  警告: 标签文件不存在，跳过: {label_file}")
        continue
    
    # 复制文件（保持原始文件名）
    target_img = os.path.join(target_path, "images", "val", img_basename)
    target_label = os.path.join(target_path, "labels", "val", img_basename.replace(".jpg", ".txt"))
    
    shutil.copy(img_file, target_img)
    shutil.copy(label_file, target_label)
    copied_val += 1
    
    if (idx + 1) % 200 == 0:
        print(f"  已复制 {idx + 1}/{nums_val} 个验证样本")

print(f"✓ 验证集完成: 成功复制 {copied_val} 个样本")

# 创建对应的 yaml 配置文件
print("\n创建数据集配置文件...")
yaml_content = f"""# DOTA Dataset Configuration (Subset for fast testing)

# Dataset paths
path: {os.path.abspath(target_path)}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes (DOTA v1.0)
names:
  0: plane
  1: ship
  2: storage-tank
  3: baseball-diamond
  4: tennis-court
  5: basketball-court
  6: ground-track-field
  7: harbor
  8: bridge
  9: large-vehicle
  10: small-vehicle
  11: helicopter
  12: roundabout
  13: soccer-ball-field
  14: swimming-pool

# Number of classes
nc: 15

# Subset info
subset: true
train_samples: {copied_train}
val_samples: {copied_val}
"""

yaml_file = os.path.join(target_path, "dota_sub.yaml")
with open(yaml_file, 'w') as f:
    f.write(yaml_content)

print(f"✓ 配置文件已创建: {yaml_file}")

print("\n" + "=" * 80)
print("数据集子集创建完成！")
print("=" * 80)
print(f"训练样本: {copied_train}")
print(f"验证样本: {copied_val}")
print(f"总样本数: {copied_train + copied_val}")
print(f"\n使用方法:")
print(f"  python train_yolo11_swin_obb.py \\")
print(f"      --data {yaml_file} \\")
print(f"      --epochs 10 \\")
print(f"      --batch 4")
print("=" * 80)