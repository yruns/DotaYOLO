#!/usr/bin/env python3
"""
RSAR数据集转YOLO OBB格式转换脚本

RSAR原始标注格式:
    x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
    - 四个角点的像素坐标（8个值）
    - 类别名称
    - 难度标志（0=简单，1=困难）

YOLO OBB格式:
    class_index x1 y1 x2 y2 x3 y3 x4 y4
    - class_index: 类别索引（从0开始）
    - 四个角点的归一化坐标（0-1之间）
    - 坐标相对于图像宽高进行归一化

类别映射:
    0: ship (船舶)
    1: bridge (桥梁)
    2: car (汽车)
    3: tank (坦克)
    4: aircraft (飞机)
    5: harbor (港口)
"""

import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm
import shutil


# 类别映射
CLASS_NAMES = ['ship', 'bridge', 'car', 'tank', 'aircraft', 'harbor']
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def get_image_size(image_path: Path) -> tuple:
    """获取图像尺寸"""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def find_image_file(images_dir: Path, base_name: str) -> Path | None:
    """查找对应的图像文件（支持多种格式）"""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    for ext in extensions:
        image_path = images_dir / f"{base_name}{ext}"
        if image_path.exists():
            return image_path
    return None


def convert_annotation(
    ann_path: Path,
    images_dir: Path,
    output_labels_dir: Path,
    output_images_dir: Path | None = None,
    copy_images: bool = False
) -> dict:
    """
    转换单个标注文件
    
    返回: 统计信息字典
    """
    base_name = ann_path.stem
    image_path = find_image_file(images_dir, base_name)
    
    stats = {
        'success': False,
        'objects': 0,
        'skipped_objects': 0,
        'error': None
    }
    
    if image_path is None:
        stats['error'] = f"找不到图像文件: {base_name}"
        return stats
    
    try:
        img_width, img_height = get_image_size(image_path)
    except Exception as e:
        stats['error'] = f"无法读取图像 {image_path}: {e}"
        return stats
    
    # 读取标注文件
    yolo_lines = []
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 10:
                stats['skipped_objects'] += 1
                continue
            
            try:
                # 解析8个坐标点
                x1, y1 = float(parts[0]), float(parts[1])
                x2, y2 = float(parts[2]), float(parts[3])
                x3, y3 = float(parts[4]), float(parts[5])
                x4, y4 = float(parts[6]), float(parts[7])
                class_name = parts[8]
                # difficulty = int(parts[9])  # 目前不使用
                
                # 检查类别是否有效
                if class_name not in CLASS_TO_INDEX:
                    stats['skipped_objects'] += 1
                    continue
                
                class_idx = CLASS_TO_INDEX[class_name]
                
                # 归一化坐标
                x1_norm = x1 / img_width
                y1_norm = y1 / img_height
                x2_norm = x2 / img_width
                y2_norm = y2 / img_height
                x3_norm = x3 / img_width
                y3_norm = y3 / img_height
                x4_norm = x4 / img_width
                y4_norm = y4 / img_height
                
                # 裁剪到[0, 1]范围（处理可能超出边界的标注）
                x1_norm = max(0.0, min(1.0, x1_norm))
                y1_norm = max(0.0, min(1.0, y1_norm))
                x2_norm = max(0.0, min(1.0, x2_norm))
                y2_norm = max(0.0, min(1.0, y2_norm))
                x3_norm = max(0.0, min(1.0, x3_norm))
                y3_norm = max(0.0, min(1.0, y3_norm))
                x4_norm = max(0.0, min(1.0, x4_norm))
                y4_norm = max(0.0, min(1.0, y4_norm))
                
                # 生成YOLO OBB格式的行
                yolo_line = f"{class_idx} {x1_norm:.6f} {y1_norm:.6f} {x2_norm:.6f} {y2_norm:.6f} {x3_norm:.6f} {y3_norm:.6f} {x4_norm:.6f} {y4_norm:.6f}"
                yolo_lines.append(yolo_line)
                stats['objects'] += 1
                
            except (ValueError, IndexError) as e:
                stats['skipped_objects'] += 1
                continue
    
    # 写入YOLO格式标注
    output_label_path = output_labels_dir / f"{base_name}.txt"
    with open(output_label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_lines))
        if yolo_lines:
            f.write('\n')
    
    # 复制或链接图像（可选）
    if copy_images and output_images_dir:
        output_image_path = output_images_dir / image_path.name
        if not output_image_path.exists():
            shutil.copy2(image_path, output_image_path)
    
    stats['success'] = True
    return stats


def convert_split(
    split_name: str,
    input_dir: Path,
    output_dir: Path,
    copy_images: bool = False,
    num_workers: int = 8
):
    """转换一个数据集分割（train/val/test）"""
    ann_dir = input_dir / split_name / 'annfiles'
    images_dir = input_dir / split_name / 'images'
    
    if not ann_dir.exists():
        print(f"警告: {ann_dir} 不存在，跳过")
        return
    
    # 创建输出目录
    output_labels_dir = output_dir / split_name / 'labels'
    output_images_dir = output_dir / split_name / 'images'
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    if copy_images:
        output_images_dir.mkdir(parents=True, exist_ok=True)
    else:
        # 创建符号链接到原始图像目录
        if not output_images_dir.exists():
            output_images_dir.symlink_to(images_dir.resolve())
    
    # 获取所有标注文件
    ann_files = list(ann_dir.glob('*.txt'))
    
    print(f"\n转换 {split_name} 集: {len(ann_files)} 个文件")
    
    total_objects = 0
    total_skipped = 0
    errors = []
    
    # 使用多进程加速
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                convert_annotation,
                ann_path,
                images_dir,
                output_labels_dir,
                output_images_dir if copy_images else None,
                copy_images
            ): ann_path for ann_path in ann_files
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"处理 {split_name}"):
            ann_path = futures[future]
            try:
                stats = future.result()
                total_objects += stats['objects']
                total_skipped += stats['skipped_objects']
                if stats['error']:
                    errors.append(stats['error'])
            except Exception as e:
                errors.append(f"处理 {ann_path} 时出错: {e}")
    
    print(f"  - 成功转换目标数: {total_objects}")
    print(f"  - 跳过的目标数: {total_skipped}")
    if errors:
        print(f"  - 错误数: {len(errors)}")
        for err in errors[:5]:
            print(f"    {err}")
        if len(errors) > 5:
            print(f"    ... 还有 {len(errors) - 5} 个错误")


def create_yaml_config(output_dir: Path):
    """创建YOLO数据集配置文件"""
    yaml_content = f"""# RSAR Dataset for YOLO OBB
# Converted from RSAR multi-class rotated SAR object detection dataset

path: {output_dir.resolve()}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: ship
  1: bridge
  2: car
  3: tank
  4: aircraft
  5: harbor

# Dataset statistics:
# - train: 78,837 images
# - val: 8,467 images  
# - test: 8,538 images
# - Total instances: ~150,000
"""
    
    yaml_path = output_dir / 'rsar.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n已创建配置文件: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description='将RSAR数据集转换为YOLO OBB格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本转换（图像使用符号链接）
  python convert_rsar_to_yolo_obb.py --input /data1/ysh/Datasets/RSAR --output /data1/ysh/Datasets/RSAR_YOLO_OBB
  
  # 复制图像（占用更多空间，但更独立）
  python convert_rsar_to_yolo_obb.py --input /data1/ysh/Datasets/RSAR --output /data1/ysh/Datasets/RSAR_YOLO_OBB --copy-images
  
  # 只转换特定分割
  python convert_rsar_to_yolo_obb.py --input /data1/ysh/Datasets/RSAR --output /data1/ysh/Datasets/RSAR_YOLO_OBB --splits train val
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='RSAR数据集根目录路径'
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str,
        required=True,
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--splits', '-s',
        nargs='+',
        default=['train', 'val', 'test'],
        help='要转换的数据集分割 (默认: train val test)'
    )
    
    parser.add_argument(
        '--copy-images',
        action='store_true',
        help='复制图像而非创建符号链接'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=8,
        help='并行处理的工作进程数 (默认: 8)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    print("=" * 60)
    print("RSAR → YOLO OBB 格式转换")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"图像处理: {'复制' if args.copy_images else '符号链接'}")
    print(f"工作进程: {args.workers}")
    print(f"分割集: {', '.join(args.splits)}")
    print()
    print("类别映射:")
    for name, idx in CLASS_TO_INDEX.items():
        print(f"  {idx}: {name}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换各分割
    for split in args.splits:
        convert_split(
            split,
            input_dir,
            output_dir,
            copy_images=args.copy_images,
            num_workers=args.workers
        )
    
    # 创建YAML配置文件
    create_yaml_config(output_dir)
    
    print("\n" + "=" * 60)
    print("转换完成!")
    print("=" * 60)
    print(f"\n使用方法:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('yolov8n-obb.pt')")
    print(f"  model.train(data='{output_dir}/rsar.yaml', epochs=100)")


if __name__ == '__main__':
    main()