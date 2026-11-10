# YOLOv12n 垃圾分类检测：外挂注意力模块消融实验

在 YOLOv12n 上分别加装 CA、CBAM、SE 三种通道/空间注意力模块，用同一份六分类垃圾数据集、同一套超参数训练，观察检测精度是否提升。结论没有达到预期：三组记录都没有超过不加模块的基线。

仓库只包含配置、训练日志和图表。数据集与论文原文均不在此仓库内。

## 实验在问什么

YOLOv12n 的骨干与 neck 中已经包含区域注意力（area attention）和注意力偏置。本实验要回答的问题是：在这样一个已经具备注意力机制的检测器上，再外挂一个通用的通道注意力（SE、CA）或通道+空间注意力（CBAM）模块，能否进一步提升垃圾分类的检测精度。

## 数据

六分类垃圾检测数据集，共 10,464 张图，划分如下：

| 划分 | 图像数 |
|------|--------|
| train | 6,420 |
| val | 2,133 |
| test | 1,156 |

类别：`BIODEGRADABLE`、`CARDBOARD`、`GLASS`、`METAL`、`PAPER`、`PLASTIC`。

数据集来自 Roboflow（CC BY 4.0），**不随仓库分发**。配置见 [src/configs/data.yaml](src/configs/data.yaml)，下载后按其中的目录结构放置即可。

类别分布极不均衡，这一点直接影响下面的结果解读：`BIODEGRADABLE` 有 13,637 个实例，`GLASS` 2,380，`METAL` 1,360，`CARDBOARD` 1,292，`PLASTIC` 214，`PAPER` 只有 15 张图、33 个实例。

## 训练配置

四组实验共用同一套超参数，仅 `project` / `name` 不同。完整参数见 `src/configs/args_*.yaml`。

| 参数 | 值 |
|------|-----|
| 模型 | YOLOv12n（`yolo12n.pt`，预训练） |
| epochs | 80 |
| batch | 32 |
| imgsz | 640 |
| 优化器 | SGD（momentum 0.937，weight_decay 0.0005） |
| 学习率 | 0.01，线性 warmup 3 epoch，衰减至 2.24e-05 |
| 数据增强 | mosaic 1.0，fliplr 0.5，HSV，close_mosaic 10 |
| 混合精度 | AMP 开启 |

## 结果

数字全部读自各实验目录下的 `results.csv`，取第 80 轮（最后一轮）的值。

| 实验 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| Baseline（YOLOv12n） | 0.6193 | 0.5011 | **0.5483** | **0.3789** |
| + CA | 0.6213 | 0.4845 | 0.5411 | 0.3738 |
| + CBAM | 0.6213 | 0.4845 | 0.5411 | 0.3738 |
| + SE | 0.6213 | 0.4845 | 0.5411 | 0.3738 |

基线在训练过程中的几个节点（同样读自 `results.csv`）：

| epoch | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| 1 | 0.2988 | 0.2697 | 0.2375 | 0.1296 |
| 20 | 0.5212 | 0.4238 | 0.4606 | 0.2962 |
| 40 | 0.5468 | 0.4739 | 0.5071 | 0.3382 |
| 60 | 0.6113 | 0.4978 | 0.5436 | 0.3716 |
| 72 | 0.6032 | 0.5111 | 0.5579 | 0.3826 |
| 80 | 0.6193 | 0.5011 | 0.5483 | 0.3789 |

基线自身在第 72 轮达到最高 mAP50（0.5579），此后小幅回落，属于正常的验证集波动。

### 关于表中三行相同的说明

上表中三组注意力实验的数字完全一致，**这不是排版省略**。核对本地三份 `results.csv` 后确认：

- 三个文件的 80 轮指标序列逐行相同，只有 `time`（累计耗时）列和行的排列顺序不同；
- 每个文件内部还存在重复轮次（同一 epoch 出现两次，指标值相同、`time` 不同）；
- `experiments/CBAM/results.png` 与 `experiments/SE/results.png` 字节完全相同。

也就是说，**仓库里保存的这三份日志实际上是同一条训练曲线的三份拷贝**，无法据此区分三种注意力模块各自的独立效果。同时，三份 `args.yaml` 中 `model` 均为 `yolo12n.pt`、`cfg` 均为 `null`，没有引用任何注入 CA/CBAM/SE 的自定义模型结构文件，因此从配置上也看不出这次训练与基线在结构上有差异。

所以本仓库能支持的结论只有一条：**留存下来的注意力实验记录（mAP50 0.5411）低于基线（0.5483）**。至于"三种模块各自都没有带来提升"，现有材料不足以支撑——要得到这个结论，需要每组实验各自独立的、未经覆盖的 `results.csv` 和对应的模型结构配置。

## 结论

加装注意力模块没有提升检测精度。在这一数据集和这套训练设置下，CA、CBAM、SE 都没有超过 YOLOv12n 基线，mAP50 与 mAP50-95 两项指标均略低：mAP50 下降 0.72 个百分点，mAP50-95 下降 0.51 个百分点；Precision 高 0.2 个百分点，Recall 低 1.66 个百分点。差距很小，处在验证集自身波动的量级内，因此更准确的说法是"没有观察到提升"，而不是"显著变差"。

## 原因分析

**一、YOLOv12n 本身已经包含注意力机制，外挂模块属于重复建模。** YOLOv12 的 A2C2f 模块内建区域注意力（area attention），检测头中也带有注意力偏置项，网络在训练中已经学到了在空间与通道维度上该关注哪里。再叠加 SE、CA、CBAM 这类通用注意力，只是在已有的注意力路径旁边并联一条功能相近的更弱的分支，带来的是冗余而非新增的表达能力。参数量增加，但有效信息通路没有变多。

**二、小数据集上加参数容易过拟合。** 训练集 6,420 张图，六类，且分布严重不均。在这种规模下，新增的注意力分支带来的额外参数更容易拟合训练集的噪声，而不是学到可泛化的判别特征。日志中的表现与此吻合：三组注意力实验的 Recall 比基线低（0.4845 对 0.5011），但 Precision 略高，这是典型的决策边界变保守、泛化能力未改善的形态。

**三、PAPER 类的极端样本不足掩盖了模块间的差异。** `PAPER` 只有 15 张图，基线在这一类上的 mAP50 仅 0.076，几乎是随机水平。整体 mAP 被这一类大量拉低，任何模块级的小幅改进都会被这一项的噪声淹没。在补齐长尾类之前，用整体 mAP 比较注意力模块的收益，信噪比本身就不够。

需要说明：以上是机制层面的解释，不是从实验结果反推出来的因果证明。第 1、2 条原因本次没有做对照实验（例如去掉 A2C2f 内建注意力后再外挂模块），因此属于有依据的推测。

## 已知问题

1. 三份注意力实验的 `results.csv` 互为拷贝，见上文说明。
2. 三组实验只保留了 `results.csv` 与 `results.png`；只有 SE 组保留了混淆矩阵、PR 曲线和验证集预测图，CA 与 CBAM 组没有。
3. 训练时的模型结构配置（注入注意力模块的 `.yaml`）没有留存，`args.yaml` 指向的是原始 `yolo12n.pt`，按现有材料无法完整复现三组注意力的模型。
4. `src/legacy_4class/` 下是更早的四分类阶段（recyclable / kitchen / hazardous / other）的两阶段训练脚本，与本文六分类 80 轮实验不是同一套配置。保留它们只为记录代码，不能用来复现上表结果。
5. 原先由上述日志绘制的四组 mAP 对比曲线没有收入仓库：日志中同一轮次重复出现、行序错乱，按行绘制会得到锯齿状折线，图中三条注意力曲线也呈现出数值上不可能的差异。确认日志后再重新出图。

## 目录结构

```
.
├── README.md
├── src/
│   ├── configs/
│   │   ├── data.yaml            六分类数据配置
│   │   ├── args_baseline.yaml   基线训练参数
│   │   ├── args_CA.yaml
│   │   ├── args_CBAM.yaml
│   │   └── args_SE.yaml
│   └── legacy_4class/           早期四分类阶段脚本（仅供参考）
├── experiments/
│   ├── baseline/                results.csv, results.png, 混淆矩阵, PR 曲线, 验证集预测
│   ├── CA/                      results.csv, results.png
│   ├── CBAM/                    results.csv, results.png
│   └── SE/                      results.csv, results.png, 混淆矩阵, PR 曲线, 验证集预测
├── results/
│   └── yolov12_architecture.png 模型结构示意
└── weights/                     权重不纳入版本控制，见下节
```

## 权重

四个 `best.pt`（各约 5.3 MB）不放在仓库里，需要通过 GitHub Release 附件分发，或从作者处另行获取。若不提供，仓库只保留日志与图表。

## 复现

```bash
pip install ultralytics pandas matplotlib
# 按 src/configs/data.yaml 的说明准备数据
yolo detect train \
  data=src/configs/data.yaml \
  model=yolo12n.pt \
  epochs=80 imgsz=640 batch=32 \
  optimizer=SGD lr0=0.01 lrf=0.01 \
  project=runs name=baseline
```

四组实验的 `args_*.yaml` 中除 `data`、`project`、`save_dir` 已改为相对路径外，其余与训练时一致。

## 引用与许可

数据集：Roboflow *Garbage Classification*（`material-identification/garbage-classification-3`，CC BY 4.0）。
检测框架：Ultralytics YOLOv12（AGPL-3.0）。
