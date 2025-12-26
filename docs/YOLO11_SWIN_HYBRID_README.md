# YOLO11 + Swin Hybrid OBB 模型

在 YOLO11 backbone 末尾（SPPF 之前）插入 Swin Transformer Block，增强全局上下文建模能力。

**✅ 不修改 ultralytics 源码，使用自定义模块注入方式**

## 📊 模型概览

| 配置 | 参数量 | GFLOPs | 说明 |
|------|--------|--------|------|
| `yolo11_swin_obb.yaml` | 75.71M | 453.4 | Swin 在 backbone 末尾 |
| 原版 `yolo11l-obb` | 26.22M | 91.3 | 纯卷积 backbone |

## 🏗️ 架构设计

```
YOLO11 Backbone:
┌─────────────────────────────────────────┐
│ Layer 0-2:  Conv + C3k2 (P2)            │
│ Layer 3-4:  Conv + C3k2 (P3/8)          │
│ Layer 5-6:  Conv + C3k2 (P4/16)         │
│ Layer 7-8:  Conv + C3k2 (P5/32)         │
│                                          │
│ Layer 9:  ★ Swin [1024, 2, 8, 7] ★      │ ← 新增
│           [c2, depth, heads, window]     │
│                                          │
│ Layer 10: SPPF                           │
│ Layer 11: C2PSA                          │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ FPN/PAN Head + OBB Detection            │
└─────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境要求

```bash
pip install ultralytics
```

### 2. 训练模型

**重要**：必须先注册自定义模块！

```python
# train_swin_hybrid.py
from custom_modules import register_custom_modules
register_custom_modules()  # 必须在导入 YOLO 之前

from ultralytics import YOLO

model = YOLO("models/yolo11_swin_obb.yaml", task="obb")
model.train(data="DOTAv1.yaml", epochs=100, device=0)
```

或直接运行：
```bash
python train_swin_hybrid.py
```

## 📁 文件结构

```
datov1/
├── custom_modules.py              # Swin 模块实现 + 注册函数
├── models/
│   └── yolo11_swin_obb.yaml       # 模型配置
├── train_swin_hybrid.py           # 训练脚本
└── runs/obb/                      # 训练结果
```

## ⚙️ YAML 配置详解

### Swin 模块参数

```yaml
# 在 SPPF 之前插入 Swin
- [-1, 1, Swin, [1024, 2, 8, 7]]  # [c2, depth, num_heads, window_size]
```

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `c2` | 输出通道数 | 与输入通道相同 (如 1024) |
| `depth` | Swin block 重复次数 | 2-4 |
| `num_heads` | 注意力头数 | 8 或 16 |
| `window_size` | 窗口大小 | 7 (适合 20x20 特征图) |

### 完整 YAML 示例

```yaml
# yolo11_swin_obb.yaml
nc: 15  # DOTA classes

backbone:
  - [-1, 1, Conv, [64, 3, 2]]           # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]          # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]   # 2
  - [-1, 1, Conv, [256, 3, 2]]          # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]   # 4
  - [-1, 1, Conv, [512, 3, 2]]          # 5-P4/16
  - [-1, 2, C3k2, [512, True]]          # 6
  - [-1, 1, Conv, [1024, 3, 2]]         # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]         # 8
  
  - [-1, 1, Swin, [1024, 2, 8, 7]]      # 9 ★ Swin Block
  
  - [-1, 1, SPPF, [1024, 5]]            # 10
  - [-1, 2, C2PSA, [1024]]              # 11

head:
  # ... FPN/PAN + OBB
```

## 🔧 自定义 Swin 参数

```yaml
# 更深的 Swin (更多全局信息)
- [-1, 1, Swin, [1024, 4, 8, 7]]   # depth=4

# 更多注意力头 (更细粒度)
- [-1, 1, Swin, [1024, 2, 16, 7]]  # heads=16

# 更大窗口 (更大感受野，需要更大特征图)
- [-1, 1, Swin, [1024, 2, 8, 14]]  # window=14
```

## 📝 技术细节

### custom_modules.py 核心实现

```python
class Swin(nn.Module):
    """Swin Transformer Block for YOLO backbone"""
    
    def __init__(self, c2, depth=2, num_heads=8, window_size=7):
        # Window-based Multi-head Self Attention
        # MLP with GELU activation
        # Layer Normalization
        ...

def register_custom_modules():
    """注入到 ultralytics 命名空间"""
    import ultralytics.nn.tasks as tasks
    tasks.Swin = Swin
```

### 为什么在 SPPF 之前插入？

1. **P5 特征图 (20×20)** 适合 window attention
2. **卷积已提取局部特征**，Swin 增强全局上下文
3. **不影响多尺度特征提取**，P3/P4 保持卷积结构

## 📈 训练建议

| 设备 | batch | imgsz | epochs |
|------|-------|-------|--------|
| CPU | 1 | 640 | 测试用 |
| 单 GPU | 4-8 | 1024 | 100-200 |
| 多 GPU | 16+ | 1024 | 200+ |

## 📚 参考

- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [YOLO11 Docs](https://docs.ultralytics.com/models/yolo11)
- [DOTA Dataset](https://captain-whu.github.io/DOTA/)

## 📝 更新日志

- **2024-12-26**: 
  - 创建 `Swin` 模块 (Window Attention)
  - 正确放置在 backbone 末尾 (SPPF 之前)
  - 不修改 ultralytics 源码，使用模块注入方式


