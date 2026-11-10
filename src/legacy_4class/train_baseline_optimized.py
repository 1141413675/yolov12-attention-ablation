from ultralytics import YOLO
import torch
import os

Config = {
    "data_yaml": "./data.yaml",
    "pretrained_weights": "./yolov12n.pt",
    "custom_model_yaml": "./yolov12_baseline.yaml",
    "img_size": 640,
    "batch_size": 32,
    "epochs": 100,
    "freeze_epochs": 20,
    "lr0": 0.001,
    "device": "0",
    "weight_decay": 5e-5,
    "project": "./runs/garbage_baseline",
    "name": "yolov12_baseline_optimized"
}

def init_model():
    model = YOLO(Config["pretrained_weights"]) if os.path.exists(Config["pretrained_weights"]) else YOLO(Config["custom_model_yaml"])
    print("✓ 基线模型初始化成功")
    return model

def train():
    print("=== 阶段1：冻结Backbone训练（20轮）===")
    model = init_model()
    
    model.train(
        data=Config["data_yaml"],
        epochs=Config["freeze_epochs"],
        batch=Config["batch_size"],
        imgsz=Config["img_size"],
        device=Config["device"],
        project=Config["project"],
        name=f"{Config['name']}_freeze",
        lr0=Config["lr0"],
        lrf=0.0001,
        optimizer="AdamW",
        cos_lr=True,
        weight_decay=Config["weight_decay"],
        augment=True,
        save=True,
        val=True,
        cache="disk",
        freeze=10,
        amp=True,
        patience=50
    )
    
    print("=== 阶段2：全量微调训练（80轮）===")
    model_path = os.path.join(Config["project"], f"{Config['name']}_freeze", "weights", "best.pt")
    if os.path.exists(model_path):
        print("✓ 加载阶段1训练完成的最佳模型")
        model = YOLO(model_path)
    else:
        print("⚠️  未找到阶段1最佳模型，继续使用当前模型")
    
    model.train(
        data=Config["data_yaml"],
        epochs=Config["epochs"] - Config["freeze_epochs"],
        batch=Config["batch_size"],
        imgsz=Config["img_size"],
        device=Config["device"],
        project=Config["project"],
        name=f"{Config['name']}_unfreeze",
        lr0=Config["lr0"] * 0.1,
        lrf=0.0001,
        optimizer="AdamW",
        cos_lr=True,
        weight_decay=Config["weight_decay"],
        augment=True,
        save=True,
        val=True,
        cache="disk",
        amp=True,
        patience=50
    )
    
    print(f"基线模型训练完成！最佳模型路径：{os.path.join(Config['project'], Config['name'] + '_unfreeze', 'weights', 'best.pt')}")

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.makedirs(Config["project"], exist_ok=True)
    
    train()