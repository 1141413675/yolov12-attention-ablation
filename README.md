# YOLOv12n 垃圾分类检测：外挂注意力模块消融实验

在 YOLOv12n 上分别加装 CA、CBAM、SE 三种通道/空间注意力模块，用同一份六分类垃圾数据集、同一套超参数各训练一次，观察检测精度是否提升。结论没有达到预期：留存下来的那一组没有超过不加模块的基线。

三组实验都实际跑过（三份 `best.pt` 的 md5 各不相同，可证是三次独立训练），但归档时日志出现重复——三份 `results.csv` 实际是同一份文件的多份拷贝。CA 与 CBAM 的原始日志已无法恢复（训练服务器已销毁），因此仓库只公开 SE 组一份完整记录，结论也只针对这一组。

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

各次实验共用同一套超参数，仅 `project` / `name` 不同。完整参数见 `src/configs/args_*.yaml`。

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

### 为什么只有一组注意力结果

CA、CBAM、SE 三组都训练过，但归档时出了问题：三份 `results.csv` 的 80 轮指标序列逐行相同，只有 `time`（累计耗时）列和行序不同，并且每个文件内部同一轮次还重复出现两次。也就是说，其中两份是第三份的拷贝，三组实验只剩一条曲线可用。`CBAM` 与 `SE` 目录下的 `results.png` 也是字节相同的同一张图。

三份 `best.pt` 的 md5 各不相同，说明训练确实分开跑过三次；重复发生在收集、打包日志的环节，而不是没有训练。但 CA 与 CBAM 的原始日志没有第二份备份，训练服务器已销毁，无法恢复。

因此仓库只保留 SE 组：它的目录最完整，混淆矩阵、PR 曲线和验证集预测图都配套。CA、CBAM 的训练参数仍保留在 `src/configs/args_CA.yaml`、`args_CBAM.yaml`，但没有任何指标数据。下文的结论和原因分析都只基于 SE 这一组，不能推广到 CA 与 CBAM。

## 结论

留存下来的 SE 组没有超过 YOLOv12n 基线。mAP50 与 mAP50-95 两项都略低：mAP50 下降 0.72 个百分点（0.5411 对 0.5483），mAP50-95 下降 0.51 个百分点（0.3738 对 0.3789）；Precision 高 0.2 个百分点，Recall 低 1.66 个百分点。差距很小，落在验证集自身波动的量级内，所以准确的说法是「没有观察到提升」，而不是「显著变差」。

这个结论只适用于 SE 组。CA 与 CBAM 的独立结果没有留存，无从判断。

## 原因分析

**一、YOLOv12n 本身已经包含注意力机制，外挂模块属于重复建模。** YOLOv12 的 A2C2f 模块内建区域注意力（area attention），检测头中也带有注意力偏置项，网络在训练中已经学到了在空间与通道维度上该关注哪里。再叠加 SE、CA、CBAM 这类通用注意力，只是在已有的注意力路径旁边并联一条功能相近的更弱的分支，带来的是冗余而非新增的表达能力。参数量增加，但有效信息通路没有变多。

**二、小数据集上加参数容易过拟合。** 训练集 6,420 张图，六类，且分布严重不均。在这种规模下，新增的注意力分支带来的额外参数更容易拟合训练集的噪声，而不是学到可泛化的判别特征。留存组的日志与这一判断相符：SE 组的 Recall 比基线低（0.4845 对 0.5011），但 Precision 略高，是决策边界变保守、泛化能力未改善的形态。

**三、PAPER 类的极端样本不足会淹没模块级差异。** `PAPER` 只有 15 张图，基线在这一类上的 mAP50 仅 0.076，几乎是随机水平。整体 mAP 被这一类大量拉低，任何模块级的小幅改进都会被这一项的噪声盖过去。在补齐长尾类之前，用整体 mAP 判断注意力模块是否有收益，信噪比本身就不够。

需要说明：以上是基于留存组观察到的现象所做的机制层面解释，不是因果证明。第 1、2 条没有做对照实验（例如去掉 A2C2f 内建注意力后再外挂模块），属于有依据的推测。

## 已知问题

1. 归档时三份注意力日志互为拷贝，CA 与 CBAM 的原始 `results.csv` 已不可恢复（训练服务器已销毁），仓库只保留 SE 组。两组的训练参数仍留在 `src/configs/`。
2. 三组实验的 `args.yaml` 中 `model` 均为 `yolo12n.pt`、`cfg` 均为 `null`，注入注意力模块的模型结构 `.yaml` 没有留存，按现有材料无法完整复现注意力模型的结构。
3. CA 与 CBAM 组原本也没有保存混淆矩阵和验证集预测图，只有 SE 组配套完整。
4. 原按重复日志绘制的四组 mAP 对比曲线未收入仓库：日志中同一轮次重复出现、行序错乱，按行绘制会得到锯齿状折线，且图中三条注意力曲线呈现出数值上不可能的差异。
5. `src/legacy_4class/` 下是更早的四分类阶段（recyclable / kitchen / hazardous / other）的两阶段训练脚本，与本文六分类 80 轮实验不是同一套配置。保留它们只为记录代码，不能用来复现上表结果。

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
│   └── SE/                      results.csv, results.png, 混淆矩阵, PR 曲线, 验证集预测
├── results/
│   └── yolov12_architecture.png 模型结构示意
└── weights/                     权重不纳入版本控制，见下节
```

## 权重

四个 `best.pt`（基线、CA、CBAM、SE，各约 5.3 MB）不纳入版本控制。目前它们只保存在本地，尚未发布到 Release，因此这里不提供下载链接。

发布后计划以 Release 附件形式提供：`baseline_best.pt`、`CA_best.pt`、`CBAM_best.pt`、`SE_best.pt`。在此之前如需权重，请直接联系作者。

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

各次实验的 `args_*.yaml` 中除 `data`、`project`、`save_dir` 已改为相对路径外，其余与训练时一致。

## 引用与许可

数据集：Roboflow *Garbage Classification*（`material-identification/garbage-classification-3`，CC BY 4.0）。
检测框架：Ultralytics YOLOv12（AGPL-3.0）。
