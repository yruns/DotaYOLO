#!/usr/bin/env python3
"""
DroneVehicle数据集转YOLO OBB格式转换脚本

DroneVehicle原始标注格式 (XML):
    <annotation>
      <size>
        <width>840</width>
        <height>712</height>
      </size>
      <object>
        <name>car</name>
        <polygon>
          <x1>254</x1><y1>259</y1>
          <x2>256</x2><y2>321</y2>
          <x3>279</x3><y3>318</y3>
          <x4>275</x4><y4>258</y4>
        </polygon>
      </object>
    </annotation>

YOLO OBB格式:
    class_index x1 y1 x2 y2 x3 y3 x4 y4
    - class_index: 类别索引（从0开始）
    - 四个角点的归一化坐标（0-1之间）

类别映射:
    0: car (汽车)
    1: truck (卡车)
    2: bus (公交车)
    3: van (面包车)
    4: freight_car (货车)

数据集结构:
    DroneVehicle/
    ├── train/
    │   ├── trainimg/      # RGB可见光图像
    │   ├── trainimgr/     # 红外图像
    │   ├── trainlabel/    # RGB图像标注
    │   └── trainlabelr/   # 红外图像标注
    └── val/
        ├── valimg/
        ├── valimgr/
        ├── vallabel/
        └── vallabelr/
"""

import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm
import shutil


# 类别映射
CLASS_NAMES = ['car', 'truck', 'bus', 'van', 'freight_car']
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# 处理类别名称中的变体和拼写错误
CLASS_ALIASES = {
    'car': 'car',
    'truck': 'truck',
    'truvk': 'truck',  # 拼写错误
    'bus': 'bus',
    'van': 'van',
    'feright_car': 'freight_car',
    'feright car': 'freight_car',
    'feright': 'freight_car',
}


def normalize_class_name(name: str) -> str | None:
    """将类别名称规范化"""
    name = name.strip().lower()
    if name in CLASS_ALIASES:
        return CLASS_ALIASES[name]
    return None


def get_image_size(image_path: Path) -> tuple:
    """获取图像尺寸"""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def get_image_size_from_xml(xml_root) -> tuple | None:
    """从XML标注中获取图像尺寸"""
    size = xml_root.find('size')
    if size is not None:
        width = size.find('width')
        height = size.find('height')
        if width is not None and height is not None:
            return int(width.text), int(height.text)
    return None


def find_image_file(images_dir: Path, base_name: str) -> Path | None:
    """查找对应的图像文件（支持多种格式）"""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    for ext in extensions:
        image_path = images_dir / f"{base_name}{ext}"
        if image_path.exists():
            return image_path
    return None


def parse_polygon(obj) -> tuple | None:
    """从XML对象中解析polygon坐标"""
    polygon = obj.find('polygon')
    if polygon is None:
        return None
    
    try:
        x1 = float(polygon.find('x1').text)
        y1 = float(polygon.find('y1').text)
        x2 = float(polygon.find('x2').text)
        y2 = float(polygon.find('y2').text)
        x3 = float(polygon.find('x3').text)
        y3 = float(polygon.find('y3').text)
        x4 = float(polygon.find('x4').text)
        y4 = float(polygon.find('y4').text)
        return (x1, y1, x2, y2, x3, y3, x4, y4)
    except (AttributeError, ValueError):
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
        'unknown_classes': set(),
        'error': None
    }
    
    if image_path is None:
        stats['error'] = f"找不到图像文件: {base_name}"
        return stats
    
    # 解析XML
    try:
        tree = ET.parse(ann_path)
        root = tree.getroot()
    except ET.ParseError as e:
        stats['error'] = f"XML解析错误 {ann_path}: {e}"
        return stats
    
    # 获取图像尺寸（优先从图像文件读取，保证准确性）
    try:
        img_width, img_height = get_image_size(image_path)
    except Exception as e:
        # 如果无法读取图像，尝试从XML获取
        size = get_image_size_from_xml(root)
        if size is None:
            stats['error'] = f"无法获取图像尺寸 {image_path}: {e}"
            return stats
        img_width, img_height = size
    
    # 解析所有对象
    yolo_lines = []
    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is None:
            stats['skipped_objects'] += 1
            continue
        
        class_name = name_elem.text
        if class_name is None:
            stats['skipped_objects'] += 1
            continue
        
        # 规范化类别名称
        normalized_name = normalize_class_name(class_name)
        if normalized_name is None:
            stats['unknown_classes'].add(class_name)
            stats['skipped_objects'] += 1
            continue
        
        class_idx = CLASS_TO_INDEX[normalized_name]
        
        # 解析polygon坐标
        coords = parse_polygon(obj)
        if coords is None:
            stats['skipped_objects'] += 1
            continue
        
        x1, y1, x2, y2, x3, y3, x4, y4 = coords
        
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
    image_type: str = 'rgb',  # 'rgb' 或 'ir'
    copy_images: bool = False,
    num_workers: int = 8
):
    """转换一个数据集分割（train/val）"""
    
    # 根据分割名称和图像类型确定目录
    if split_name == 'train':
        if image_type == 'rgb':
            ann_dir = input_dir / 'train' / 'trainlabel'
            images_dir = input_dir / 'train' / 'trainimg'
        else:  # ir
            ann_dir = input_dir / 'train' / 'trainlabelr'
            images_dir = input_dir / 'train' / 'trainimgr'
    else:  # val
        if image_type == 'rgb':
            ann_dir = input_dir / 'val' / 'vallabel'
            images_dir = input_dir / 'val' / 'valimg'
        else:  # ir
            ann_dir = input_dir / 'val' / 'vallabelr'
            images_dir = input_dir / 'val' / 'valimgr'
    
    if not ann_dir.exists():
        print(f"警告: {ann_dir} 不存在，跳过")
        return
    
    if not images_dir.exists():
        print(f"警告: {images_dir} 不存在，跳过")
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
    ann_files = list(ann_dir.glob('*.xml'))
    
    print(f"\n转换 {split_name} 集 ({image_type}): {len(ann_files)} 个文件")
    
    total_objects = 0
    total_skipped = 0
    all_unknown_classes = set()
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
                all_unknown_classes.update(stats.get('unknown_classes', set()))
                if stats['error']:
                    errors.append(stats['error'])
            except Exception as e:
                errors.append(f"处理 {ann_path} 时出错: {e}")
    
    print(f"  - 成功转换目标数: {total_objects}")
    print(f"  - 跳过的目标数: {total_skipped}")
    if all_unknown_classes:
        print(f"  - 未知类别: {all_unknown_classes}")
    if errors:
        print(f"  - 错误数: {len(errors)}")
        for err in errors[:5]:
            print(f"    {err}")
        if len(errors) > 5:
            print(f"    ... 还有 {len(errors) - 5} 个错误")


def create_yaml_config(output_dir: Path, image_type: str):
    """创建YOLO数据集配置文件"""
    suffix = '_ir' if image_type == 'ir' else ''
    yaml_content = f"""# DroneVehicle Dataset for YOLO OBB
# Converted from DroneVehicle drone-based vehicle detection dataset
# Image type: {'Infrared (IR)' if image_type == 'ir' else 'RGB (visible light)'}

path: {output_dir.resolve()}
train: train/images
val: val/images

# Classes
names:
  0: car
  1: truck
  2: bus
  3: van
  4: freight_car

# Dataset info:
# - DroneVehicle is a large-scale drone-based RGB-Infrared cross-modality vehicle detection dataset
# - Contains both RGB and Infrared images with oriented bounding box annotations
# - 5 vehicle categories: car, truck, bus, van, freight_car
"""
    
    yaml_path = output_dir / f'dronevehicle{suffix}.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n已创建配置文件: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description='将DroneVehicle数据集转换为YOLO OBB格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 转换RGB图像（图像使用符号链接）
  python convert_dronevehicle_to_yolo_obb.py --input /home/shyue/Datasets/DroneVehicle --output /home/shyue/Datasets/DroneVehicle_YOLO_OBB
  
  # 转换红外图像
  python convert_dronevehicle_to_yolo_obb.py --input /home/shyue/Datasets/DroneVehicle --output /home/shyue/Datasets/DroneVehicle_IR_YOLO_OBB --image-type ir
  
  # 复制图像（占用更多空间，但更独立）
  python convert_dronevehicle_to_yolo_obb.py --input /home/shyue/Datasets/DroneVehicle --output /home/shyue/Datasets/DroneVehicle_YOLO_OBB --copy-images
  
  # 只转换特定分割
  python convert_dronevehicle_to_yolo_obb.py --input /home/shyue/Datasets/DroneVehicle --output /home/shyue/Datasets/DroneVehicle_YOLO_OBB --splits train
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='DroneVehicle数据集根目录路径'
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
        default=['train', 'val'],
        help='要转换的数据集分割 (默认: train val)'
    )
    
    parser.add_argument(
        '--image-type', '-t',
        choices=['rgb', 'ir'],
        default='rgb',
        help='图像类型: rgb (可见光) 或 ir (红外) (默认: rgb)'
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
    print("DroneVehicle → YOLO OBB 格式转换")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"图像类型: {'红外 (IR)' if args.image_type == 'ir' else 'RGB (可见光)'}")
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
            image_type=args.image_type,
            copy_images=args.copy_images,
            num_workers=args.workers
        )
    
    # 创建YAML配置文件
    create_yaml_config(output_dir, args.image_type)
    
    print("\n" + "=" * 60)
    print("转换完成!")
    print("=" * 60)
    print(f"\n使用方法:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('yolov8n-obb.pt')")
    suffix = '_ir' if args.image_type == 'ir' else ''
    print(f"  model.train(data='{output_dir}/dronevehicle{suffix}.yaml', epochs=100)")


if __name__ == '__main__':
    main()

