# YOLO11-Swin-OBB 训练 mAP=0 问题排查与解决报告

## 1. 问题描述

### 1.1 现象

使用自定义的 YOLO11-Swin-OBB 模型在 DOTA 数据集上训练时，训练了多个 epoch 后，所有评估指标（Precision、Recall、mAP50、mAP50-95）始终为 0。

```
训练命令:
python train_yolo11_swin_obb.py \
    --model models/yolo11n_swin_obb_perfect.yaml \
    --data datasets/DOTAv1-split-sub/dota_sub.yaml \
    --epochs 50 \
    --batch 8 \
    --device 1 \
    --lr0 0.0005
```

训练结果 (`results.csv`):
```
epoch  precision  recall  mAP50  mAP50-95  train/box_loss  train/cls_loss
1      0          0       0      0         3.22            5.35
2      0          0       0      0         3.33            4.17
3      0          0       0      0         3.32            4.18
4      0          0       0      0         3.32            4.15
```

### 1.2 初始模型配置

原模型 `yolo11n_swin_obb_perfect.yaml`:

```yaml
nc: 15  # DOTA dataset classes

backbone:
  - [-1, 1, TorchVision, [768, "swin_t", "DEFAULT", True, 3, False]]  # Swin-T
  - [-1, 1, Conv, [1024, 1, 1]]     # 768->1024
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # P4
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # P3
  - [-1, 2, C3k2, [256, False]]

head:
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 5], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 3], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  - [[7, 10, 13], 1, OBB, [nc, 1]]
```

---

## 2. 排查过程

### 2.1 初步检查

#### 2.1.1 数据集格式验证

首先检查数据集标注格式是否正确：

```bash
head -5 datasets/DOTAv1-split-sub/labels/train/P0000__1024__0___0.txt
```

输出：
```
10 0.700194 0.074219 0.708985 0.0761721 0.70508 0.0927732 0.697265 0.0878907
10 0.719728 0.0800783 0.726562 0.0820314 0.721681 0.098633 0.713867 0.0957031
```

✅ 标注格式正确：`class_id x1 y1 x2 y2 x3 y3 x4 y4` (OBB 8点格式)

#### 2.1.2 模型结构验证

检查模型是否能正确加载和推理：

```python
from ultralytics import YOLO
model = YOLO('models/yolo11n_swin_obb_perfect.yaml')
# ✅ 模型加载成功
```

#### 2.1.3 模型推理测试

使用训练后的模型进行推理测试：

```python
model = YOLO('runs/obb/yolo11_swin_obb_dota/weights/last.pt')
results = model.predict(img_path, conf=0.001)  # 极低置信度阈值

# 结果:
# 检测到 300 个目标
# 置信度范围: [0.0010, 0.0028]  # 最高置信度只有 0.28%!
```

⚠️ **发现问题**: 模型能检测到目标，但置信度极低（最高仅 0.28%），说明模型没有学到有效的特征。

### 2.2 深入分析：特征传递检查

#### 2.2.1 逐层特征统计

编写脚本检查各层特征的统计信息：

```python
model.model.eval()
x = torch.randn(1, 3, 1024, 1024)

with torch.no_grad():
    feat = x
    for i, layer in enumerate(model.model.model):
        feat = layer(feat)
        print(f'Layer {i}: std={feat.std():.4f}')
```

**结果发现严重问题**：

```
Layer 0 (TorchVision): std=0.6633  ✅ 正常
Layer 1 (Conv):        std=0.2114  ✅ 正常
Layer 2 (SPPF):        std=0.0400  ⚠️ 开始下降
Layer 3 (C2PSA):       std=0.0037  ⚠️ 急剧下降
Layer 4 (Upsample):    std=0.0037
Layer 5 (C3k2):        std=0.0003  ❌ 严重衰减
Layer 6 (Upsample):    std=0.0003
Layer 7 (C3k2):        std=0.0000  ❌ 接近零!
```

🔴 **关键发现**: 特征标准差从 0.66 衰减到接近 0，这就是 mAP=0 的根本原因！

### 2.3 原因分析

#### 2.3.1 对比标准 YOLO11 结构

标准 YOLO11 的 FPN 结构：
```yaml
backbone:
  - Conv -> P2 -> Conv -> P3 -> Conv -> P4 -> Conv -> P5
                          ↓           ↓           ↓
                         保存        保存        保存
head:
  - Upsample -> Concat(P4) -> Upsample -> Concat(P3) -> Detect
                   ↑                         ↑
              从backbone引入              从backbone引入

关键: 每次上采样后都有 Concat 补充原始特征!
```

原 Swin 模型结构 (问题版本)：
```yaml
backbone:
  - Swin -> P5 (唯一输出)
               ↓
  - Conv -> SPPF -> C2PSA -> Upsample -> C3k2 -> Upsample -> C3k2
      ↓        ↓        ↓                  ↓                  ↓
    0.21     0.04    0.004              0.0003             0.00003
                        ↓                  ↓                  ↓
                     特征衰减 ─────────────────────────> 接近零!

问题: 没有任何跳跃连接 (skip connection) 来补充特征!
```

改进后的 Swin 多尺度模型：
```yaml
backbone:
  - SwinMultiScale (多阶段输出)
               ↓
         ┌─────┼─────┐
         ↓     ↓     ↓
        P3    P4    P5         ← 直接从 Swin 各阶段提取
      (1/8) (1/16) (1/32)
      192ch  384ch  768ch
      std=0.91  std=4.41  std=5.14   ← 保持原始特征强度!
         ↓     ↓     ↓
       Conv  Conv  Conv+SPPF
       256ch 512ch  512ch

head (FPN Top-down):
  - P5 -> Upsample -> Concat(P4) -> Upsample -> Concat(P3)
                         ↑                         ↑
                    从backbone引入              从backbone引入

head (PAN Bottom-up):
  - P3 -> Conv -> Concat(P4) -> Conv -> Concat(P5) -> OBB Detect
                     ↑                     ↑
                从FPN引入               从FPN引入

优势: 
  1. 多尺度特征直接从 Swin 各阶段提取
  2. 每个尺度保持原始特征强度 (std > 0.9)
  3. FPN/PAN 中有 Concat 跳跃连接
  4. 处理后特征 std > 0.28，无衰减!
```

#### 2.3.2 特征衰减的数学原理

1. **权重初始化小**: 卷积权重 std ≈ 0.02（标准初始化）
2. **连续卷积衰减**: 每层输出 ≈ 输入 × 权重 → 值不断变小
3. **没有残差连接**: 无法保持原始信号强度
4. **上采样不增加信息**: 只是像素复制，后续卷积继续衰减

```
理论估算:
Layer 1: std ≈ 0.66 × 0.02 × √768 ≈ 0.36
Layer 2: std ≈ 0.36 × 0.02 × √1024 ≈ 0.22
Layer 3: std ≈ 0.22 × 0.02 × √1024 ≈ 0.13
... 持续衰减
```

#### 2.3.3 为什么标准 YOLO 没有这个问题？

标准 YOLO 在每次上采样后都有 `Concat` 操作，从 backbone 引入原始特征：

```yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]     # ← 关键: 从 backbone layer 6 补充特征!
```

这样即使处理过程中有衰减，也有新的强特征补充进来。

---

## 3. 解决方案

### 3.1 核心思路

使用 Swin Transformer 的**多尺度输出**，而不是只取最后一层的特征：

- **Stage 3**: P3 特征 (1/8 scale, 192 channels)
- **Stage 5**: P4 特征 (1/16 scale, 384 channels)
- **Stage 7**: P5 特征 (1/32 scale, 768 channels)

这样可以像标准 YOLO 一样使用 FPN/PAN 结构进行特征融合。

### 3.2 实现步骤

#### 3.2.1 创建自定义模块

在 `ultralytics/nn/modules/block.py` 中添加两个新模块：

**SwinMultiScale**: 提取 Swin 的多尺度特征

```python
class SwinMultiScale(nn.Module):
    """Swin Transformer backbone with multi-scale feature extraction."""
    
    def __init__(self, weights="DEFAULT"):
        super().__init__()
        import torchvision
        swin = torchvision.models.swin_t(weights=weights)
        self.features = swin.features
        self.stage_indices = [3, 5, 7]  # P3, P4, P5
        
    def forward(self, x):
        outputs = []
        feat = x
        for i, layer in enumerate(self.features):
            feat = layer(feat)
            if i in self.stage_indices:
                # NHWC -> NCHW
                out = feat.permute(0, 3, 1, 2).contiguous()
                outputs.append(out)
        return outputs  # [P3, P4, P5]
```

**SwinIndex**: 从多尺度输出中提取单个尺度

```python
class SwinIndex(nn.Module):
    """Extract specific scale from SwinMultiScale output."""
    
    def __init__(self, index=0):
        super().__init__()
        self.index = index
    
    def forward(self, x):
        feat = x[self.index] if isinstance(x, list) else x
        # 处理 NHWC -> NCHW 转换
        if feat.dim() == 4 and feat.shape[1] <= feat.shape[-1]:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat
```

#### 3.2.2 注册模块

在 `ultralytics/nn/modules/__init__.py` 中导出新模块：

```python
from .block import (
    ...
    SwinIndex,
    SwinMultiScale,
    ...
)
```

在 `ultralytics/nn/tasks.py` 中添加解析逻辑：

```python
elif m is SwinMultiScale:
    c2 = [192, 384, 768]  # Multi-output channels
    args = [*args] if args else ["DEFAULT"]
elif m is SwinIndex:
    c2 = args[0]  # Output channels
    args = [args[1]] if len(args) > 1 else [0]
```

#### 3.2.3 新模型配置

创建 `models/yolo11n_swin_multiscale_obb.yaml`:

```yaml
nc: 15

backbone:
  # SwinMultiScale 输出 [P3:192ch, P4:384ch, P5:768ch]
  - [-1, 1, SwinMultiScale, []]                   # 0: 多尺度输出
  
  # 提取各尺度特征
  - [0, 1, SwinIndex, [192, 0]]                   # 1: P3
  - [0, 1, SwinIndex, [384, 1]]                   # 2: P4
  - [0, 1, SwinIndex, [768, 2]]                   # 3: P5
  
  # 通道调整
  - [1, 1, Conv, [256, 1, 1]]                     # 4: P3 192->256
  - [2, 1, Conv, [512, 1, 1]]                     # 5: P4 384->512
  - [3, 1, Conv, [512, 1, 1]]                     # 6: P5 768->512
  - [-1, 1, SPPF, [512, 5]]                       # 7: SPPF

head:
  # FPN: Top-down
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]    # 8
  - [[-1, 5], 1, Concat, [1]]                     # 9: Concat P4
  - [-1, 2, C3k2, [512, False]]                   # 10

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]    # 11
  - [[-1, 4], 1, Concat, [1]]                     # 12: Concat P3
  - [-1, 2, C3k2, [256, False]]                   # 13: P3/8

  # PAN: Bottom-up
  - [-1, 1, Conv, [256, 3, 2]]                    # 14
  - [[-1, 10], 1, Concat, [1]]                    # 15: Concat P4
  - [-1, 2, C3k2, [512, False]]                   # 16: P4/16

  - [-1, 1, Conv, [512, 3, 2]]                    # 17
  - [[-1, 7], 1, Concat, [1]]                     # 18: Concat P5
  - [-1, 2, C3k2, [512, True]]                    # 19: P5/32

  - [[13, 16, 19], 1, OBB, [nc, 1]]               # 20: OBB Head
```

### 3.3 验证结果

修改后的特征统计：

```
Layer  0 (SwinMultiScale): 输出 3 个张量
   P3: torch.Size([1, 192, 128, 128]), std=0.9070  ✅
   P4: torch.Size([1, 384, 64, 64]),   std=4.4054  ✅
   P5: torch.Size([1, 768, 32, 32]),   std=5.1412  ✅

Layer  4 (Conv): std=0.2846  ✅
Layer  5 (Conv): std=1.7139  ✅
Layer  7 (SPPF): std=0.2855  ✅
Layer  8 (Upsample): std=0.2855  ✅
... 所有后续层 std > 0.01 ✅
```

**对比**:

| 尺度 | 原模型 std | 新模型 std | 改进 |
|------|-----------|-----------|------|
| P3 | 0.0003 ❌ | 0.91 ✅ | 3000x |
| P4 | 0.004 ⚠️ | 4.41 ✅ | 1100x |
| P5 | 0.66 | 5.14 ✅ | 8x |

---

## 4. 使用方法

### 4.1 训练命令

```bash
python train_swin_multiscale.py \
    --model models/yolo11n_swin_multiscale_obb.yaml \
    --data datasets/DOTAv1-split-sub/dota_sub.yaml \
    --epochs 50 \
    --batch 8 \
    --device 1 \
    --lr0 0.001
```

### 4.2 注意事项

1. 必须使用本地修改过的 ultralytics 代码（包含 SwinMultiScale 和 SwinIndex 模块）
2. 训练脚本中已添加 `sys.path.insert(0, 'ultralytics')` 来确保使用本地代码

---

## 5. 总结

### 5.1 问题根因

原模型只使用 Swin Transformer 的最终输出 (P5)，通过上采样生成 P3/P4 特征。由于：
1. 缺少跳跃连接 (skip connections)
2. 连续卷积导致特征值衰减
3. 上采样不增加信息量

导致 P3/P4 特征衰减到接近 0，模型无法有效学习。

### 5.2 解决方案

利用 Swin Transformer 的多阶段输出，直接提取 P3、P4、P5 三个尺度的特征，配合 FPN/PAN 结构进行特征融合，保持各尺度特征的有效性。

### 5.3 关键教训

1. **多尺度检测需要多尺度特征**: 不能只从单一尺度上采样生成
2. **特征融合需要跳跃连接**: 防止信息在深层网络中丢失
3. **调试时检查特征统计**: std、mean 等统计量能快速定位问题
4. **理解模型架构设计原理**: 知其然更要知其所以然

---

## 附录

### A. 修改的文件列表

1. `ultralytics/ultralytics/nn/modules/block.py` - 添加 SwinMultiScale、SwinIndex
2. `ultralytics/ultralytics/nn/modules/__init__.py` - 导出新模块
3. `ultralytics/ultralytics/nn/tasks.py` - 添加模块解析逻辑
4. `models/yolo11n_swin_multiscale_obb.yaml` - 新模型配置
5. `train_swin_multiscale.py` - 训练脚本

### B. 相关资源

- [YOLO11 官方文档](https://docs.ultralytics.com/models/yolo11/)
- [Swin Transformer 论文](https://arxiv.org/abs/2103.14030)
- [Feature Pyramid Networks 论文](https://arxiv.org/abs/1612.03144)

