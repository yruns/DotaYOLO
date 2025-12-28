# YOLO OBB 评估脚本使用说明

## 功能说明

`evaluate_obb.py` 脚本用于评估YOLO OBB格式的目标检测结果，支持计算以下指标：

- **Precision (精确率)**: TP / (TP + FP)
- **Recall (召回率)**: TP / (TP + FN)
- **F1-Score**: 2 * Precision * Recall / (Precision + Recall)
- **mAP@0.5**: IoU阈值为0.5时的平均精度
- **mAP@0.5:0.95**: IoU阈值从0.5到0.95（步长0.05）的平均精度
- **AP at different IoU thresholds**: 不同IoU阈值下的平均精度

## 标签格式

### Ground Truth 格式
每行一个目标，格式为：
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

其中：
- `class_id`: 类别ID（整数）
- `x1, y1, x2, y2, x3, y3, x4, y4`: 旋转框四个顶点的归一化坐标（0-1之间）

### Predicted 格式（可选置信度）
每行一个目标，格式为：
```
class_id x1 y1 x2 y2 x3 y3 x4 y4 [confidence]
```

其中：
- `class_id`: 类别ID（整数）
- `x1, y1, x2, y2, x3, y3, x4, y4`: 旋转框四个顶点的归一化坐标（0-1之间）
- `confidence`: 可选，预测置信度（0-1之间），如果没有则默认为1.0

## 使用方法

### 基本用法

```bash
python evaluate_obb.py --gt datasets/RSAR_YOLO_OBB/val/labels --pred predict/labels --num-classes 6
```

### 参数说明

- `--gt`: Ground truth标签目录路径（必需）
- `--pred`: 预测标签目录路径（必需）
- `--num-classes`: 类别数量（默认：6）
- `--output`: 输出JSON文件路径（默认：evaluation_results.json）

### 示例

```bash
# 评估RSAR数据集
python evaluate_obb.py \
    --gt datasets/RSAR_YOLO_OBB/val/labels \
    --pred predictions/20250128_120000/labels \
    --num-classes 6 \
    --output rsar_eval_results.json
```

## 输出说明

### 控制台输出

脚本会在控制台输出以下信息：

```
======================================================================
EVALUATION RESULTS
======================================================================

Overall Metrics (IoU=0.5):
  Precision: 0.8523
  Recall:    0.7891
  F1-Score:  0.8196
  TP: 1234, FP: 215, FN: 328

mAP Metrics:
  mAP@0.5:     0.8234
  mAP@0.5:0.95: 0.6789

Per-Class Metrics (IoU=0.5):
Class       Precision    Recall       F1-Score     TP     FP     FN
----------------------------------------------------------------------
ship        0.8756       0.8234       0.8488       456    65     98
bridge      0.8123       0.7567       0.7837       234    54     75
car         0.8345       0.8012       0.8175       312    62     77
tank        0.8678       0.8234       0.8450       123    19     26
aircraft    0.8234       0.7656       0.7936       78     17     24
harbor      0.8501       0.7745       0.8107       31     -2     28

AP at different IoU thresholds:
IoU         AP
----------------------
0.50        0.8234
0.55        0.8012
0.60        0.7789
0.65        0.7567
0.70        0.7345
0.75        0.7123
0.80        0.6901
0.85        0.6678
0.90        0.6456
0.95        0.6234

Results saved to evaluation_results.json
```

### JSON输出文件

脚本会生成一个JSON文件，包含详细的评估结果：

```json
{
  "overall_metrics": {
    "precision": 0.8523,
    "recall": 0.7891,
    "f1": 0.8196,
    "tp": 1234,
    "fp": 215,
    "fn": 328
  },
  "map_metrics": {
    "mAP@0.5": 0.8234,
    "mAP@0.5:0.95": 0.6789
  },
  "class_metrics": {
    "0": {
      "iou_50": {
        "precision": 0.8756,
        "recall": 0.8234,
        "f1": 0.8488,
        "tp": 456,
        "fp": 65,
        "fn": 98
      }
    },
    ...
  },
  "ap_per_iou": {
    "0.5": 0.8234,
    "0.55": 0.8012,
    ...
  },
  "class_names": ["ship", "bridge", "car", "tank", "aircraft", "harbor"]
}
```

## 注意事项

1. **文件名对应**: Ground truth和预测标签的文件名必须一一对应（不包括扩展名）
2. **坐标归一化**: 所有坐标必须是归一化的（0-1之间）
3. **类别ID**: 类别ID必须从0开始连续编号
4. **置信度**: 预测标签可以包含置信度，也可以不包含

## 依赖项

- Python 3.7+
- numpy
- opencv-python
- (其他依赖见 requirements.txt)

## 工作流程

1. 使用 `predict_obb.py` 生成预测结果
2. 使用 `evaluate_obb.py` 评估预测结果
3. 查看控制台输出和JSON文件获取详细指标

## 示例完整流程

```bash
# 1. 生成预测结果
python predict_obb.py \
    --pt runs/rsar/yolo11n-obb_20250128_120000/weights/best.pt \
    --source datasets/RSAR_YOLO_OBB/val/images \
    --conf 0.25

# 2. 评估预测结果
python evaluate_obb.py \
    --gt datasets/RSAR_YOLO_OBB/val/labels \
    --pred predictions/20250128_120000/labels \
    --num-classes 6 \
    --output rsar_eval_results.json

# 3. 查看结果
cat rsar_eval_results.json
```
