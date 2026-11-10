import pandas as pd
import os
import time

def monitor_training():
    ca_path = "./runs/garbage_train/yolov12_ca_optimized_freeze/results.csv"
    baseline_path = "./runs/garbage_baseline/yolov12_baseline_optimized_freeze/results.csv"
    
    print("训练监控脚本启动...")
    print("=" * 80)
    
    while True:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检查训练进度...")
        print("-" * 80)
        
        if os.path.exists(ca_path):
            try:
                ca_df = pd.read_csv(ca_path)
                if len(ca_df) > 0:
                    last_ca = ca_df.iloc[-1]
                    print(f"改进模型 (CA注意力):")
                    print(f"  Epoch: {last_ca['epoch']:.0f}/{len(ca_df)}")
                    print(f"  mAP50: {last_ca['metrics/mAP50(B)']:.4f}")
                    print(f"  mAP50-95: {last_ca['metrics/mAP50-95(B)']:.4f}")
                    print(f"  Precision: {last_ca['metrics/precision(B)']:.4f}")
                    print(f"  Recall: {last_ca['metrics/recall(B)']:.4f}")
            except Exception as e:
                print(f"改进模型: 读取失败 - {e}")
        else:
            print("改进模型: 等待训练开始...")
        
        print()
        
        if os.path.exists(baseline_path):
            try:
                baseline_df = pd.read_csv(baseline_path)
                if len(baseline_df) > 0:
                    last_baseline = baseline_df.iloc[-1]
                    print(f"基线模型 (Baseline):")
                    print(f"  Epoch: {last_baseline['epoch']:.0f}/{len(baseline_df)}")
                    print(f"  mAP50: {last_baseline['metrics/mAP50(B)']:.4f}")
                    print(f"  mAP50-95: {last_baseline['metrics/mAP50-95(B)']:.4f}")
                    print(f"  Precision: {last_baseline['metrics/precision(B)']:.4f}")
                    print(f"  Recall: {last_baseline['metrics/recall(B)']:.4f}")
            except Exception as e:
                print(f"基线模型: 读取失败 - {e}")
        else:
            print("基线模型: 等待训练开始...")
        
        print("=" * 80)
        time.sleep(60)

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n监控已停止")