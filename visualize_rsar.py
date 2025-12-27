"""
可视化RSAR数据集
检查图像和旋转边界框标注
"""
import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from collections import defaultdict

CLASS_NAMES = ['ship', 'bridge', 'car', 'tank', 'aircraft', 'harbor']
CLASS_COLORS = [
    (255, 0, 0),      # ship - 红色
    (0, 255, 0),      # bridge - 绿色
    (0, 0, 255),      # car - 蓝色
    (255, 255, 0),    # tank - 黄色
    (255, 0, 255),    # aircraft - 紫色
    (0, 255, 255),    # harbor - 青色
]

def load_annotations(label_path):
    """加载YOLO OBB格式的标注"""
    annotations = []
    if not label_path.exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 9:
                class_idx = int(parts[0])
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[1:9])
                annotations.append({
                    'class_idx': class_idx,
                    'class_name': CLASS_NAMES[class_idx],
                    'points': [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                })
    
    return annotations

def denormalize_points(points, img_width, img_height):
    """将归一化的坐标转换为像素坐标"""
    return [(x * img_width, y * img_height) for x, y in points]

def visualize_single_image(img_path, label_path, ax=None):
    """可视化单张图像及其标注"""
    img = Image.open(img_path)
    img_width, img_height = img.size
    
    annotations = load_annotations(label_path)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        show = True
    else:
        show = False
    
    ax.imshow(img)
    ax.set_title(f"{img_path.name}\n{len(annotations)} objects", fontsize=10)
    ax.axis('off')
    
    for ann in annotations:
        points = denormalize_points(ann['points'], img_width, img_height)
        color = CLASS_COLORS[ann['class_idx']]
        
        polygon = Polygon(points, 
                      closed=True, 
                      fill=False, 
                      edgecolor=[c/255.0 for c in color], 
                      linewidth=2,
                      alpha=0.8)
        ax.add_patch(polygon)
        
        cx = sum(p[0] for p in points) / 4
        cy = sum(p[1] for p in points) / 4
        ax.text(cx, cy, ann['class_name'], 
                color='white', 
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor=[c/255.0 for c in color], 
                         alpha=0.7))
    
    if show:
        plt.tight_layout()
        plt.show()

def visualize_dataset_grid(data_root, split='train', num_samples=16, seed=42):
    """网格可视化数据集样本"""
    random.seed(seed)
    np.random.seed(seed)
    
    data_root = Path(data_root)
    images_dir = data_root / split / 'images'
    labels_dir = data_root / split / 'labels'
    
    if not images_dir.exists():
        print(f"错误: {images_dir} 不存在")
        return
    
    image_files = list(images_dir.glob('*.*'))
    print(f"找到 {len(image_files)} 个图像文件")
    
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    class_counts = defaultdict(int)
    
    for idx, img_path in enumerate(selected_files):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        visualize_single_image(img_path, label_path, ax=ax)
        
        annotations = load_annotations(label_path)
        for ann in annotations:
            class_counts[ann['class_name']] += 1
    
    for idx in range(len(selected_files), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f"RSAR Dataset - {split} split (random {num_samples} samples)", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = f"rsar_{split}_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存到: {output_path}")
    
    print(f"\n类别统计 (可视化样本):")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {class_counts[class_name]}")
    
    plt.show()

def analyze_class_distribution(data_root, split='train'):
    """分析类别分布"""
    data_root = Path(data_root)
    labels_dir = data_root / split / 'labels'
    
    if not labels_dir.exists():
        print(f"错误: {labels_dir} 不存在")
        return
    
    class_counts = defaultdict(int)
    total_objects = 0
    
    label_files = list(labels_dir.glob('*.txt'))
    print(f"\n分析 {split} 集类别分布...")
    print(f"找到 {len(label_files)} 个标注文件")
    
    for label_path in label_files:
        annotations = load_annotations(label_path)
        for ann in annotations:
            class_counts[ann['class_name']] += 1
            total_objects += 1
    
    print(f"\n{split} 集类别分布:")
    print("-" * 50)
    for class_name in CLASS_NAMES:
        count = class_counts[class_name]
        percentage = count / total_objects * 100 if total_objects > 0 else 0
        print(f"  {class_name:12s}: {count:6d} ({percentage:5.2f}%)")
    print("-" * 50)
    print(f"  {'总计':12s}: {total_objects:6d}")
    
    return class_counts

def visualize_class_distribution(data_root, splits=['train', 'val', 'test']):
    """可视化类别分布"""
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5))
    if len(splits) == 1:
        axes = [axes]
    
    all_counts = {}
    
    for idx, split in enumerate(splits):
        data_root = Path(data_root)
        labels_dir = data_root / split / 'labels'
        
        class_counts = defaultdict(int)
        total_objects = 0
        
        if labels_dir.exists():
            label_files = list(labels_dir.glob('*.txt'))
            for label_path in label_files:
                annotations = load_annotations(label_path)
                for ann in annotations:
                    class_counts[ann['class_name']] += 1
                    total_objects += 1
        
        all_counts[split] = class_counts
        
        counts = [class_counts[name] for name in CLASS_NAMES]
        colors = [[c/255.0 for c in color] for color in CLASS_COLORS]
        
        bars = axes[idx].bar(CLASS_NAMES, counts, color=colors, alpha=0.8, edgecolor='black')
        axes[idx].set_title(f'{split} (total: {total_objects})', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45, labelsize=9)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('RSAR Dataset Class Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = "rsar_class_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n类别分布图已保存到: {output_path}")
    
    plt.show()

def check_annotation_quality(data_root, split='train', num_samples=100):
    """检查标注质量"""
    random.seed(42)
    data_root = Path(data_root)
    images_dir = data_root / split / 'images'
    labels_dir = data_root / split / 'labels'
    
    print(f"\n检查 {split} 集标注质量...")
    
    issues = {
        'no_label': [],
        'empty_label': [],
        'invalid_coords': [],
        'out_of_bounds': []
    }
    
    image_files = list(images_dir.glob('*.*'))
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    for img_path in sample_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            issues['no_label'].append(img_path.name)
            continue
        
        annotations = load_annotations(label_path)
        
        if not annotations:
            issues['empty_label'].append(img_path.name)
            continue
        
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        for ann in annotations:
            points = denormalize_points(ann['points'], img_width, img_height)
            
            for x, y in points:
                if x < 0 or x > img_width or y < 0 or y > img_height:
                    issues['out_of_bounds'].append(f"{img_path.name} - {ann['class_name']}")
                    break
    
    print(f"\n检查了 {len(sample_files)} 个样本")
    print(f"发现的问题:")
    print(f"  无标注文件: {len(issues['no_label'])}")
    print(f"  空标注文件: {len(issues['empty_label'])}")
    print(f"  坐标超出边界: {len(issues['out_of_bounds'])}")
    
    if issues['no_label']:
        print(f"\n无标注文件示例 (最多5个):")
        for name in issues['no_label'][:5]:
            print(f"  - {name}")
    
    return issues

if __name__ == "__main__":
    data_root = '/home/tiger/codebase/DotaYOLO/datasets/RSAR_YOLO_OBB'
    
    print("=" * 80)
    print("RSAR数据集可视化检查")
    print("=" * 80)
    
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*80}")
        print(f"分析 {split} 集")
        print(f"{'='*80}")
        
        analyze_class_distribution(data_root, split)
    
    print(f"\n{'='*80}")
    print("可视化类别分布")
    print(f"{'='*80}")
    visualize_class_distribution(data_root, splits=['train', 'val', 'test'])
    
    print(f"\n{'='*80}")
    print("可视化训练集样本")
    print(f"{'='*80}")
    visualize_dataset_grid(data_root, split='train', num_samples=16, seed=42)
    
    print(f"\n{'='*80}")
    print("检查标注质量")
    print(f"{'='*80}")
    check_annotation_quality(data_root, split='train', num_samples=100)
    
    print("\n" + "=" * 80)
    print("可视化检查完成！")
    print("=" * 80)
