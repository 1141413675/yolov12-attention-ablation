from ultralytics import YOLO
import torch
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

# 配置参数
Config = {
    "data_yaml": "./data.yaml",  # 数据集配置文件
    "img_size": 640,  # 论文要求的输入尺寸
    "batch_size": 16,  # 论文要求的批次大小
    "device": "0",  # GPU设备
    "project": "./runs/garbage_train",  # 训练结果保存路径
    "results_file": "./comparative_results.csv"  # 对比结果保存文件
}

# 评估单个模型
def evaluate_model(model_name, model_path):
    print(f"\n{'='*50}")
    print(f"开始评估模型：{model_name}")
    print(f"{'='*50}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"⚠️  未找到 {model_name} 的模型文件：{model_path}")
        return None
    
    # 加载模型
    model = YOLO(model_path)
    print(f"✓ {model_name} 模型加载成功")
    
    # 评估模型
    results = model.val(
        data=Config["data_yaml"],
        imgsz=Config["img_size"],
        batch=Config["batch_size"],
        device=Config["device"],
        save_json=True,  # 保存JSON结果
        split="test"  # 使用测试集进行最终评估
    )
    
    # 收集评估结果
    eval_results = {
        "model_name": model_name,
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
        "f1_score": results.box.f1,
        "params": model.model.n_params / 1e6,  # 参数量（百万）
        "flops": model.model.flops() / 1e9,  # 计算复杂度（GFLOPs）
        "model_path": model_path
    }
    
    print(f"\n{model_name} 评估结果：")
    print(f"mAP50: {eval_results['mAP50']:.4f}")
    print(f"mAP50-95: {eval_results['mAP50-95']:.4f}")
    print(f"Precision: {eval_results['precision']:.4f}")
    print(f"Recall: {eval_results['recall']:.4f}")
    print(f"F1 Score: {eval_results['f1_score']:.4f}")
    print(f"Params: {eval_results['params']:.2f}M")
    print(f"GFLOPs: {eval_results['flops']:.2f}")
    
    return eval_results

# 测试推理速度
def test_inference_speed(model_name, model_path):
    print(f"\n{'='*50}")
    print(f"开始测试 {model_name} 推理速度")
    print(f"{'='*50}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"⚠️  未找到 {model_name} 的模型文件：{model_path}")
        return None
    
    # 加载模型
    model = YOLO(model_path)
    
    # 使用benchmark测试推理速度
    results = model.benchmark(
        data=Config["data_yaml"],
        imgsz=Config["img_size"],
        device=Config["device"],
        half=True,  # 使用FP16加速
        batch=1
    )
    
    # 提取FPS
    fps = results.fps
    print(f"{model_name} 推理速度：{fps:.2f} FPS")
    
    return fps

# 保存结果到CSV文件
def save_results(results_list):
    # 定义CSV列名
    fieldnames = ["model_name", "mAP50", "mAP50-95", "precision", "recall", "f1_score", "params", "flops", "fps", "model_path"]
    
    # 写入CSV文件
    with open(Config["results_file"], "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results_list:
            writer.writerow(result)
    
    print(f"\n✓ 所有评估结果已保存到：{Config['results_file']}")

# 生成对比图表
def generate_plots(results_list):
    print(f"\n{'='*50}")
    print("开始生成对比图表")
    print(f"{'='*50}")
    
    # 提取数据
    model_names = [r["model_name"] for r in results_list]
    mAP50_95 = [r["mAP50-95"] for r in results_list]
    fps = [r["fps"] for r in results_list]
    params = [r["params"] for r in results_list]
    flops = [r["flops"] for r in results_list]
    
    # 设置图表风格
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    
    # 1. mAP50-95对比柱状图
    plt.figure()
    sns.barplot(x=model_names, y=mAP50_95)
    plt.xlabel("Model")
    plt.ylabel("mAP50-95")
    plt.title("Comparison of mAP50-95 across different models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("./compare_mAP.png", dpi=300)
    print("✓ mAP50-95对比图表已保存：./compare_mAP.png")
    
    # 2. FPS对比柱状图
    plt.figure()
    sns.barplot(x=model_names, y=fps)
    plt.xlabel("Model")
    plt.ylabel("FPS")
    plt.title("Comparison of inference speed (FPS) across different models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("./compare_fps.png", dpi=300)
    print("✓ FPS对比图表已保存：./compare_fps.png")
    
    # 3. 参数量对比柱状图
    plt.figure()
    sns.barplot(x=model_names, y=params)
    plt.xlabel("Model")
    plt.ylabel("Parameters (M)")
    plt.title("Comparison of model parameters across different models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("./compare_params.png", dpi=300)
    print("✓ 参数量对比图表已保存：./compare_params.png")
    
    # 4. 计算复杂度对比柱状图
    plt.figure()
    sns.barplot(x=model_names, y=flops)
    plt.xlabel("Model")
    plt.ylabel("GFLOPs")
    plt.title("Comparison of computational complexity (GFLOPs) across different models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("./compare_flops.png", dpi=300)
    print("✓ 计算复杂度对比图表已保存：./compare_flops.png")
    
    print("\n✓ 所有对比图表生成完成！")

# 主函数
def main():
    # 定义需要评估的模型列表
    models_to_evaluate = [
        # 消融实验模型
        {"name": "yolov12_baseline", "path": f"{Config['project']}/yolov12_baseline_unfreeze/weights/best.pt"},
        {"name": "yolov12_ca", "path": f"{Config['project']}/yolov12_ca_garbage_unfreeze/weights/best.pt"},
        
        # 横向对比模型
        {"name": "yolov8_s", "path": f"{Config['project']}/yolov8_s_unfreeze/weights/best.pt"},
        {"name": "yolov9_se", "path": f"{Config['project']}/yolov9_se_unfreeze/weights/best.pt"}
    ]
    
    # 评估所有模型
    results_list = []
    for model_info in models_to_evaluate:
        # 评估模型性能
        eval_result = evaluate_model(model_info["name"], model_info["path"])
        if eval_result:
            # 测试推理速度
            fps = test_inference_speed(model_info["name"], model_info["path"])
            eval_result["fps"] = fps
            results_list.append(eval_result)
    
    # 保存结果到CSV
    if results_list:
        save_results(results_list)
        
        # 生成对比图表
        generate_plots(results_list)
        
        # 打印最终对比结果
        print(f"\n{'='*70}")
        print("最终对比实验结果汇总")
        print(f"{'='*70}")
        for result in results_list:
            print(f"{result['model_name']}:")
            print(f"  mAP50-95: {result['mAP50-95']:.4f} | FPS: {result['fps']:.2f} | Params: {result['params']:.2f}M | GFLOPs: {result['flops']:.2f}")
        print(f"{'='*70}")
    else:
        print("\n⚠️  未评估到任何模型，请检查模型文件路径是否正确")

if __name__ == "__main__":
    main()