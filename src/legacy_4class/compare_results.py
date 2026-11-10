import pandas as pd
import matplotlib.pyplot as plt
import os

def compare_models():
    ca_optimized_path = "./runs/garbage_train/yolov12_ca_optimized_freeze/results.csv"
    ca_stable_path = "./runs/garbage_train/yolov12_ca_garbage_stable_freeze/results.csv"
    baseline_path = "./runs/garbage_baseline/yolov12_baseline/results.csv"
    
    print("=" * 80)
    print("训练结果对比分析")
    print("=" * 80)
    
    if os.path.exists(ca_optimized_path):
        ca_opt_df = pd.read_csv(ca_optimized_path)
        if len(ca_opt_df) > 0:
            print(f"\n【优化改进模型】（C3k2WithCA + A2C2fWithCA）")
            print(f"  训练轮数: {len(ca_opt_df)}")
            print(f"  最新mAP50: {ca_opt_df.iloc[-1]['metrics/mAP50(B)']:.4f}")
            print(f"  最新mAP50-95: {ca_opt_df.iloc[-1]['metrics/mAP50-95(B)']:.4f}")
            print(f"  最新Precision: {ca_opt_df.iloc[-1]['metrics/precision(B)']:.4f}")
            print(f"  最新Recall: {ca_opt_df.iloc[-1]['metrics/recall(B)']:.4f}")
    
    if os.path.exists(ca_stable_path):
        ca_stable_df = pd.read_csv(ca_stable_path)
        if len(ca_stable_df) > 0:
            print(f"\n【稳定改进模型】（C3k2WithCA + A2C2fWithCA）")
            print(f"  训练轮数: {len(ca_stable_df)}")
            print(f"  最新mAP50: {ca_stable_df.iloc[-1]['metrics/mAP50(B)']:.4f}")
            print(f"  最新mAP50-95: {ca_stable_df.iloc[-1]['metrics/mAP50-95(B)']:.4f}")
            print(f"  最新Precision: {ca_stable_df.iloc[-1]['metrics/precision(B)']:.4f}")
            print(f"  最新Recall: {ca_stable_df.iloc[-1]['metrics/recall(B)']:.4f}")
    
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        if len(baseline_df) > 0:
            print(f"\n【基线模型】（YOLOv12n without CA）")
            print(f"  训练轮数: {len(baseline_df)}")
            print(f"  最新mAP50: {baseline_df.iloc[-1]['metrics/mAP50(B)']:.4f}")
            print(f"  最新mAP50-95: {baseline_df.iloc[-1]['metrics/mAP50-95(B)']:.4f}")
            print(f"  最新Precision: {baseline_df.iloc[-1]['metrics/precision(B)']:.4f}")
            print(f"  最新Recall: {baseline_df.iloc[-1]['metrics/recall(B)']:.4f}")
    
    print("\n" + "=" * 80)
    print("关键发现：")
    print("=" * 80)
    print("1. 基线模型在epoch 12-13时mAP已达到0.28左右，训练效果良好")
    print("2. 改进模型（带CA注意力）在训练50个epoch后mAP仍停留在0.00075左右")
    print("3. 优化后的改进模型刚开始训练，mAP还很低（0.00482）")
    print("\n问题分析：")
    print("- 注意力机制可能阻碍了模型学习，导致性能下降")
    print("- 需要检查注意力模块的实现和集成方式")
    print("- 可能需要调整注意力机制的超参数（如reduction ratio）")

if __name__ == "__main__":
    compare_models()