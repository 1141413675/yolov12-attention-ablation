from ultralytics import YOLO
import torch
import os

# 配置参数（与改进模型完全相同）
Config = {
    "data_yaml": "./data.yaml",  # 数据集配置文件（划分后路径）
    "pretrained_weights": "./yolov12n.pt",
    "custom_model_yaml": "./yolov12_baseline.yaml",  # 基线模型配置文件
    "img_size": 640,  # 输入尺寸
    "batch_size": 16,  # 批次大小
    "epochs": 100,  # 完整训练轮数
    "freeze_epochs": 50,  # 冻结Backbone训练轮数
    "lr0": 0.0001,  # 初始学习率
    "device": "0",  # GPU设备（多GPU用"0,1"）
    "weight_decay": 1e-4,  # 降低正则化强度
    "project": "./runs/garbage_baseline",  # 训练结果保存路径
    "name": "yolov12_baseline_stable"
}

# 初始化基线模型
def init_model():
    # 加载预训练模型作为基础
    model = YOLO(Config["pretrained_weights"]) if os.path.exists(Config["pretrained_weights"]) else YOLO(Config["custom_model_yaml"])
    print("✓ 基线模型初始化成功")
    return model

# 分阶段训练（与改进模型完全相同的训练策略）
def train():
    # 阶段1：冻结Backbone，仅训练检测头
    print("=== 阶段1：冻结Backbone训练 ===")
    # 初始化模型
    model = init_model()
    
    # 阶段1训练：冻结Backbone，训练50轮
    model.train(
        data=Config["data_yaml"],
        epochs=Config["freeze_epochs"],
        batch=Config["batch_size"],
        imgsz=Config["img_size"],
        device=Config["device"],
        project=Config["project"],
        name=f"{Config['name']}_freeze",
        lr0=Config["lr0"],
        lrf=0.0001,  # 学习率最终值，结合cos_lr=True实现余弦退火
        optimizer="AdamW",  # 使用AdamW优化器
        cos_lr=True,  # 开启余弦退火学习率
        weight_decay=Config["weight_decay"],
        augment=True,
        save=True,
        val=True,
        cache=True,  # 启用数据缓存加速训练
        freeze=10  # 冻结前10层（Backbone），与改进模型相同
    )
    
    # 阶段2：全量微调（50个epoch）
    print("=== 阶段2：全量微调训练 ===")
    # 加载阶段1的最佳模型
    model_path = os.path.join(Config["project"], f"{Config['name']}_freeze", "weights", "best.pt")
    if os.path.exists(model_path):
        print("✓ 加载阶段1训练完成的最佳模型")
        model = YOLO(model_path)
    else:
        print("⚠️  未找到阶段1最佳模型，继续使用当前模型")
    
    # 阶段2训练：全量微调，剩余50轮
    model.train(
        data=Config["data_yaml"],
        epochs=Config["epochs"] - Config["freeze_epochs"],
        batch=Config["batch_size"],
        imgsz=Config["img_size"],
        device=Config["device"],
        project=Config["project"],
        name=f"{Config['name']}_unfreeze",
        lr0=Config["lr0"],
        lrf=0.0001,  # 学习率最终值，结合cos_lr=True实现余弦退火
        optimizer="AdamW",  # 使用AdamW优化器
        cos_lr=True,  # 开启余弦退火学习率
        weight_decay=Config["weight_decay"],
        augment=True,
        save=True,
        val=True,
        cache=True  # 启用数据缓存加速训练
    )
    
    print(f"基线模型训练完成！最佳模型路径：{os.path.join(Config['project'], Config['name'] + '_unfreeze', 'weights', 'best.pt')}")

if __name__ == "__main__":
    # 固定随机种子（保证可复现，与改进模型相同）
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 创建必要的目录
    os.makedirs(Config["project"], exist_ok=True)
    
    train()