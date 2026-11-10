from ultralytics import YOLO
import torch
import os

Config = {
    "data_yaml": "./data.yaml",
    "pretrained_weights": "./yolov12n.pt",
    "custom_model_yaml": "./yolov12_baseline.yaml",
    "img_size": 640,
    "batch_size": 64,
    "epochs": 50,
    "lr0": 0.01,
    "device": "0",
    "weight_decay": 1e-4,
    "project": "./runs/garbage_baseline",
    "name": "yolov12_baseline_fast"
}

def init_model():
    model = YOLO(Config["pretrained_weights"]) if os.path.exists(Config["pretrained_weights"]) else YOLO(Config["custom_model_yaml"])
    print("✓ 基线模型初始化成功")
    return model

def train():
    print("=== 快速训练基线模型（50轮）===")
    model = init_model()
    
    model.train(
        data=Config["data_yaml"],
        epochs=Config["epochs"],
        batch=Config["batch_size"],
        imgsz=Config["img_size"],
        device=Config["device"],
        project=Config["project"],
        name=Config["name"],
        lr0=Config["lr0"],
        lrf=0.001,
        optimizer="AdamW",
        cos_lr=True,
        weight_decay=Config["weight_decay"],
        augment=True,
        save=True,
        val=True,
        cache="disk",
        amp=True,
        patience=20,
        close_mosaic=10
    )
    
    print(f"训练完成！最佳模型路径：{os.path.join(Config['project'], Config['name'], 'weights', 'best.pt')}")

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.makedirs(Config["project"], exist_ok=True)
    
    train()