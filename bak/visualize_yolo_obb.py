#!/usr/bin/env python3
"""
YOLO OBB格式标注可视化脚本

功能:
    - 可视化YOLO OBB格式的旋转框标注
    - 支持单张图像或批量可视化
    - 支持保存可视化结果或交互式显示
    - 支持自定义颜色和显示选项

YOLO OBB格式:
    class_index x1 y1 x2 y2 x3 y3 x4 y4
    - class_index: 类别索引（从0开始）
    - x1-x4, y1-y4: 四个角点的归一化坐标（0-1之间）
"""

import argparse
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


# 默认颜色调色板 (BGR格式)
DEFAULT_COLORS = [
    (0, 255, 0),      # 绿色
    (0, 0, 255),      # 红色
    (255, 0, 0),      # 蓝色
    (0, 255, 255),    # 黄色
    (255, 0, 255),    # 品红
    (255, 255, 0),    # 青色
    (128, 0, 255),    # 橙色
    (255, 128, 0),    # 天蓝
    (0, 128, 255),    # 橙黄
    (128, 255, 0),    # 黄绿
    (255, 0, 128),    # 玫红
    (0, 255, 128),    # 春绿
]


def generate_colors(num_classes: int) -> list:
    """生成指定数量的颜色"""
    if num_classes <= len(DEFAULT_COLORS):
        return DEFAULT_COLORS[:num_classes]
    
    # 需要更多颜色时随机生成
    colors = list(DEFAULT_COLORS)
    random.seed(42)  # 固定随机种子保证一致性
    for _ in range(num_classes - len(DEFAULT_COLORS)):
        colors.append((
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
    return colors


def load_class_names(yaml_path: Path) -> list:
    """从YAML配置文件加载类别名称"""
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        names = config.get('names', {})
        if isinstance(names, dict):
            # 按索引排序
            max_idx = max(names.keys()) if names else -1
            return [names.get(i, f'class_{i}') for i in range(max_idx + 1)]
        elif isinstance(names, list):
            return names
        return []
    except Exception as e:
        print(f"警告: 无法加载类别名称: {e}")
        return []


def parse_yolo_obb_label(label_path: Path, img_width: int, img_height: int) -> list:
    """
    解析YOLO OBB格式的标注文件
    
    返回: [(class_idx, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]), ...]
    """
    annotations = []
    
    if not label_path.exists():
        return annotations
    
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 9:
                continue
            
            try:
                class_idx = int(parts[0])
                # 归一化坐标转像素坐标
                x1 = float(parts[1]) * img_width
                y1 = float(parts[2]) * img_height
                x2 = float(parts[3]) * img_width
                y2 = float(parts[4]) * img_height
                x3 = float(parts[5]) * img_width
                y3 = float(parts[6]) * img_height
                x4 = float(parts[7]) * img_width
                y4 = float(parts[8]) * img_height
                
                points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                annotations.append((class_idx, points))
            except (ValueError, IndexError):
                continue
    
    return annotations


def draw_obb(
    image: np.ndarray,
    annotations: list,
    class_names: Optional[list] = None,
    colors: Optional[list] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    show_label: bool = True,
    show_conf: bool = False
) -> np.ndarray:
    """
    在图像上绘制OBB标注
    
    参数:
        image: 输入图像
        annotations: 标注列表 [(class_idx, points), ...] 或 [(class_idx, points, conf), ...]
        class_names: 类别名称列表
        colors: 颜色列表
        line_thickness: 线条粗细
        font_scale: 字体大小
        show_label: 是否显示类别标签
        show_conf: 是否显示置信度
    """
    img = image.copy()
    
    # 获取最大类别索引
    max_class = max((ann[0] for ann in annotations), default=0) + 1
    
    if colors is None:
        colors = generate_colors(max_class)
    
    if class_names is None:
        class_names = [f'class_{i}' for i in range(max_class)]
    
    for ann in annotations:
        class_idx = ann[0]
        points = ann[1]
        conf = ann[2] if len(ann) > 2 else None
        
        # 获取颜色
        color = colors[class_idx % len(colors)]
        
        # 绘制旋转框
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=line_thickness)
        
        # 绘制角点
        for pt in points:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color, -1)
        
        # 绘制标签
        if show_label:
            label = class_names[class_idx] if class_idx < len(class_names) else f'class_{class_idx}'
            if show_conf and conf is not None:
                label = f'{label} {conf:.2f}'
            
            # 计算标签位置（取最上方的点）
            top_point = min(points, key=lambda p: p[1])
            label_x = int(top_point[0])
            label_y = int(top_point[1]) - 5
            
            # 获取文本大小
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # 确保标签不超出图像边界
            label_y = max(text_h + 5, label_y)
            label_x = min(img.shape[1] - text_w - 5, max(5, label_x))
            
            # 绘制背景矩形
            cv2.rectangle(
                img,
                (label_x - 2, label_y - text_h - 2),
                (label_x + text_w + 2, label_y + 2),
                color,
                -1
            )
            
            # 绘制文本
            cv2.putText(
                img,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
    
    return img


def find_label_file(image_path: Path, labels_dir: Path) -> Path:
    """查找对应的标注文件"""
    base_name = image_path.stem
    return labels_dir / f"{base_name}.txt"


def visualize_single(
    image_path: Path,
    label_path: Path,
    class_names: Optional[list] = None,
    colors: Optional[list] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    show_label: bool = True
) -> np.ndarray:
    """可视化单张图像"""
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return None
    
    img_height, img_width = img.shape[:2]
    
    # 解析标注
    annotations = parse_yolo_obb_label(label_path, img_width, img_height)
    
    # 绘制标注
    result = draw_obb(
        img,
        annotations,
        class_names=class_names,
        colors=colors,
        line_thickness=line_thickness,
        font_scale=font_scale,
        show_label=show_label
    )
    
    # 保存结果
    if output_path:
        cv2.imwrite(str(output_path), result)
        print(f"已保存: {output_path}")
    
    # 显示结果
    if show:
        # 调整窗口大小以适应屏幕
        window_name = f"Visualization - {image_path.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # 如果图像太大，缩放显示
        max_display_size = 1200
        scale = min(max_display_size / img_width, max_display_size / img_height, 1.0)
        if scale < 1.0:
            display_w = int(img_width * scale)
            display_h = int(img_height * scale)
            cv2.resizeWindow(window_name, display_w, display_h)
        
        cv2.imshow(window_name, result)
        print("按任意键继续，按 'q' 退出...")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if key == ord('q'):
            return None
    
    return result


def visualize_batch(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Optional[Path] = None,
    class_names: Optional[list] = None,
    colors: Optional[list] = None,
    num_samples: Optional[int] = None,
    random_sample: bool = False,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    show_label: bool = True
):
    """批量可视化图像"""
    # 获取所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"错误: 在 {images_dir} 中没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 采样
    if random_sample and num_samples and num_samples < len(image_files):
        random.seed(42)
        image_files = random.sample(image_files, num_samples)
    elif num_samples:
        image_files = image_files[:num_samples]
    
    # 创建输出目录
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每张图像
    for image_path in tqdm(image_files, desc="可视化"):
        label_path = find_label_file(image_path, labels_dir)
        
        output_path = None
        if output_dir:
            output_path = output_dir / f"vis_{image_path.name}"
        
        visualize_single(
            image_path,
            label_path,
            class_names=class_names,
            colors=colors,
            output_path=output_path,
            show=False,
            line_thickness=line_thickness,
            font_scale=font_scale,
            show_label=show_label
        )


def interactive_viewer(
    images_dir: Path,
    labels_dir: Path,
    class_names: Optional[list] = None,
    colors: Optional[list] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    show_label: bool = True
):
    """交互式查看器"""
    # 获取所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(f"错误: 在 {images_dir} 中没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    print("控制说明:")
    print("  [→] 或 [d]: 下一张")
    print("  [←] 或 [a]: 上一张")
    print("  [r]: 随机跳转")
    print("  [s]: 保存当前可视化")
    print("  [q] 或 [ESC]: 退出")
    print()
    
    current_idx = 0
    
    while True:
        image_path = image_files[current_idx]
        label_path = find_label_file(image_path, labels_dir)
        
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"警告: 无法读取图像 {image_path}")
            current_idx = (current_idx + 1) % len(image_files)
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 解析标注
        annotations = parse_yolo_obb_label(label_path, img_width, img_height)
        
        # 绘制标注
        result = draw_obb(
            img,
            annotations,
            class_names=class_names,
            colors=colors,
            line_thickness=line_thickness,
            font_scale=font_scale,
            show_label=show_label
        )
        
        # 添加信息文本
        info_text = f"[{current_idx + 1}/{len(image_files)}] {image_path.name} | Objects: {len(annotations)}"
        cv2.putText(
            result,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        # 显示
        window_name = "YOLO OBB Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # 调整窗口大小
        max_display_size = 1200
        scale = min(max_display_size / img_width, max_display_size / img_height, 1.0)
        if scale < 1.0:
            display_w = int(img_width * scale)
            display_h = int(img_height * scale)
            cv2.resizeWindow(window_name, display_w, display_h)
        
        cv2.imshow(window_name, result)
        
        # 等待按键
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:  # q 或 ESC
            break
        elif key == ord('d') or key == 83:  # d 或 右箭头
            current_idx = (current_idx + 1) % len(image_files)
        elif key == ord('a') or key == 81:  # a 或 左箭头
            current_idx = (current_idx - 1) % len(image_files)
        elif key == ord('r'):  # 随机
            current_idx = random.randint(0, len(image_files) - 1)
        elif key == ord('s'):  # 保存
            save_path = Path(f"vis_{image_path.name}")
            cv2.imwrite(str(save_path), result)
            print(f"已保存: {save_path}")
    
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='YOLO OBB格式标注可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 交互式查看（推荐）
  python visualize_yolo_obb.py --dataset /home/shyue/Datasets/DroneVehicle_YOLO_OBB --split train
  
  # 可视化单张图像
  python visualize_yolo_obb.py --image /path/to/image.jpg --label /path/to/label.txt
  
  # 批量可视化并保存结果
  python visualize_yolo_obb.py --dataset /home/shyue/Datasets/DroneVehicle_YOLO_OBB --split train --output ./vis_output --num 50
  
  # 随机采样可视化
  python visualize_yolo_obb.py --dataset /home/shyue/Datasets/DroneVehicle_YOLO_OBB --split val --num 20 --random

交互式查看器控制:
  [→] 或 [d]: 下一张
  [←] 或 [a]: 上一张
  [r]: 随机跳转
  [s]: 保存当前可视化
  [q] 或 [ESC]: 退出
        """
    )
    
    # 数据集模式
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='YOLO OBB数据集根目录（包含train/val子目录）'
    )
    
    parser.add_argument(
        '--split', '-s',
        type=str,
        default='train',
        help='数据集分割 (默认: train)'
    )
    
    # 单图像模式
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='单张图像路径'
    )
    
    parser.add_argument(
        '--label', '-l',
        type=str,
        help='标注文件路径'
    )
    
    # 输出选项
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出目录（批量模式）或输出文件（单图像模式）'
    )
    
    parser.add_argument(
        '--num', '-n',
        type=int,
        help='可视化的图像数量（批量模式）'
    )
    
    parser.add_argument(
        '--random', '-r',
        action='store_true',
        help='随机采样图像'
    )
    
    # 显示选项
    parser.add_argument(
        '--no-label',
        action='store_true',
        help='不显示类别标签'
    )
    
    parser.add_argument(
        '--thickness', '-t',
        type=int,
        default=2,
        help='线条粗细 (默认: 2)'
    )
    
    parser.add_argument(
        '--font-scale', '-f',
        type=float,
        default=0.5,
        help='字体大小 (默认: 0.5)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='使用交互式查看器'
    )
    
    args = parser.parse_args()
    
    # 加载类别名称
    class_names = None
    colors = None
    
    if args.dataset:
        dataset_dir = Path(args.dataset)
        
        # 尝试加载YAML配置
        yaml_files = list(dataset_dir.glob('*.yaml')) + list(dataset_dir.glob('*.yml'))
        if yaml_files:
            class_names = load_class_names(yaml_files[0])
            if class_names:
                print(f"已加载类别: {class_names}")
                colors = generate_colors(len(class_names))
        
        # 确定图像和标注目录
        images_dir = dataset_dir / args.split / 'images'
        labels_dir = dataset_dir / args.split / 'labels'
        
        if not images_dir.exists():
            print(f"错误: 图像目录不存在: {images_dir}")
            return
        
        if not labels_dir.exists():
            print(f"错误: 标注目录不存在: {labels_dir}")
            return
        
        if args.interactive or (not args.output and not args.num):
            # 交互式模式
            interactive_viewer(
                images_dir,
                labels_dir,
                class_names=class_names,
                colors=colors,
                line_thickness=args.thickness,
                font_scale=args.font_scale,
                show_label=not args.no_label
            )
        else:
            # 批量模式
            output_dir = Path(args.output) if args.output else None
            visualize_batch(
                images_dir,
                labels_dir,
                output_dir=output_dir,
                class_names=class_names,
                colors=colors,
                num_samples=args.num,
                random_sample=args.random,
                line_thickness=args.thickness,
                font_scale=args.font_scale,
                show_label=not args.no_label
            )
    
    elif args.image:
        # 单图像模式
        image_path = Path(args.image)
        
        if args.label:
            label_path = Path(args.label)
        else:
            # 尝试自动查找标注文件
            label_path = image_path.with_suffix('.txt')
            if not label_path.exists():
                label_path = image_path.parent.parent / 'labels' / f"{image_path.stem}.txt"
        
        output_path = Path(args.output) if args.output else None
        
        visualize_single(
            image_path,
            label_path,
            class_names=class_names,
            colors=colors,
            output_path=output_path,
            show=True,
            line_thickness=args.thickness,
            font_scale=args.font_scale,
            show_label=not args.no_label
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

