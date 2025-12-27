"""
DOTA数据集可视化脚本
用于验证数据集的正确性，显示OBB标注框
"""
import cv2
import numpy as np
import os
import glob
import random
from pathlib import Path

def read_dota_label(label_path):
    """读取DOTA格式的标签文件（归一化的OBB坐标）"""
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 9:  # 类别 + 8个坐标值
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:9]]
                annotations.append({
                    'class_id': class_id,
                    'coords': coords
                })
    return annotations

def denormalize_coords(coords, img_width, img_height):
    """将归一化的坐标转换为像素坐标"""
    points = []
    for i in range(0, len(coords), 2):
        x = int(coords[i] * img_width)
        y = int(coords[i+1] * img_height)
        points.append([x, y])
    return np.array(points, dtype=np.int32)

def draw_obb(image, annotations, class_names=None):
    """在图像上绘制OBB标注框"""
    img_height, img_width = image.shape[:2]
    
    # 为不同类别生成不同颜色
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (255, 0, 128), (0, 255, 128), (128, 255, 0)
    ]
    
    for ann in annotations:
        class_id = ann['class_id']
        coords = ann['coords']
        
        # 反归一化坐标
        points = denormalize_coords(coords, img_width, img_height)
        
        # 选择颜色
        color = colors[class_id % len(colors)]
        
        # 绘制多边形
        cv2.polylines(image, [points], True, color, 2)
        
        # 绘制类别标签
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
        
        label = str(class_id) if class_names is None else class_names[class_id]
        cv2.putText(image, label, (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def visualize_samples(data_root, split='train', num_samples=10, output_dir='visualizations'):
    """可视化数据集样本"""
    print(f"正在可视化 {split} 集的样本...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图片路径
    img_dir = os.path.join(data_root, 'images', split)
    label_dir = os.path.join(data_root, 'labels', split)
    
    if not os.path.exists(img_dir):
        print(f"错误：找不到图片目录 {img_dir}")
        return
    
    # 获取所有图片
    img_paths = glob.glob(os.path.join(img_dir, '*.jpg')) + \
                glob.glob(os.path.join(img_dir, '*.png'))
    
    if len(img_paths) == 0:
        print(f"错误：{img_dir} 中没有图片")
        return
    
    print(f"找到 {len(img_paths)} 张图片")
    
    # DOTA类别名称
    class_names = [
        'plane', 'ship', 'storage-tank', 'baseball-diamond',
        'tennis-court', 'basketball-court', 'ground-track-field',
        'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
        'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool'
    ]
    
    # 随机选择样本
    sample_paths = random.sample(img_paths, min(num_samples, len(img_paths)))
    
    # 统计信息
    total_objects = 0
    class_counts = {}
    
    for idx, img_path in enumerate(sample_paths):
        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告：无法读取图片 {img_path}")
            continue
        
        # 读取标签
        img_name = Path(img_path).stem
        label_path = os.path.join(label_dir, f"{img_name}.txt")
        annotations = read_dota_label(label_path)
        
        # 统计
        total_objects += len(annotations)
        for ann in annotations:
            class_id = ann['class_id']
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        # 绘制标注
        vis_image = draw_obb(image.copy(), annotations, class_names)
        
        # 添加信息文本
        info_text = f"Image: {img_name} | Objects: {len(annotations)} | Size: {image.shape[1]}x{image.shape[0]}"
        cv2.putText(vis_image, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存结果
        output_path = os.path.join(output_dir, f"{split}_{idx:03d}_{img_name}.jpg")
        cv2.imwrite(output_path, vis_image)
        print(f"  [{idx+1}/{len(sample_paths)}] 已保存: {output_path}")
    
    # 打印统计信息
    print(f"\n=== {split.upper()} 集统计 ===")
    print(f"总图片数: {len(img_paths)}")
    print(f"总目标数: {total_objects}")
    print(f"平均每张图片目标数: {total_objects/len(sample_paths):.2f}")
    print("\n类别分布:")
    for class_id in sorted(class_counts.keys()):
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        print(f"  {class_name} (id={class_id}): {class_counts[class_id]}")
    print()

if __name__ == "__main__":
    # 数据集路径
    data_root = "/home/shyue/codebase/datov1/datasets/DOTAv1-split"
    
    # 可视化训练集
    print("=" * 60)
    print("DOTA数据集可视化")
    print("=" * 60)
    visualize_samples(data_root, split='train', num_samples=15)
    
    # 可视化验证集
    visualize_samples(data_root, split='val', num_samples=10)
    
    print("=" * 60)
    print("可视化完成！请查看 visualizations/ 目录")
    print("=" * 60)

