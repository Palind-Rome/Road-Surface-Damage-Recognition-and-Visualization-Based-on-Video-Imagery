# -*- coding: utf-8 -*-
"""
YOLOv8 模型独立训练脚本。
"""
import os
import sys

# 抑制 OpenMP 冲突警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import yaml
from pathlib import Path
from ultralytics import YOLO

def main() -> None:
    # ====== 参数与路径 ======
    workers = 1
    batch_size = 8
    epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 获取项目根目录 (SITPproject/)
    project_root = Path(__file__).resolve().parent.parent.parent
    data_yaml_path = project_root / "datasets" / "data" / "data.yaml"
    
    if not data_yaml_path.exists():
        print(f"数据配置文件不存在：{data_yaml_path}")
        print("请在运行前从原项目迁移 datasets/ 文件夹至 SITPproject 目录下。")
        sys.exit(1)
        
    # ====== 重写 data.yaml 内部绝对路径 ======
    unix_style_path = str(data_yaml_path).replace(os.sep, "/")
    dir_path = os.path.dirname(unix_style_path)
    
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.full_load(f)
        
    if all(k in yaml_data for k in ["train", "val", "test"]):
        yaml_data["train"] = f"{dir_path}/train"
        yaml_data["val"] = f"{dir_path}/val"
        yaml_data["test"] = f"{dir_path}/test"
        
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, sort_keys=False)

    print(f"已更新 {data_yaml_path} 中的路径: 数据集根目录定位到此。")

    # ====== 初始化与训练 ======
    # 默认选 yolov8s-seg 预训练
    yaml_config = "yolov8s-seg.yaml"
    pt_config = project_root / "weights" / "yolov8s-seg.pt"
    
    # Ultralytics 可能会到当前工作目录同级寻找
    model = YOLO(yaml_config, task="segment")
    if pt_config.exists():
        model.load(str(pt_config))
    else:
        print(f"警告：找不到预训练权重 {pt_config}，模型将从头初始化。")

    print("\n" + "=" * 50)
    print("🚀 SITP 独立重训引擎初始化中...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ 成功接管本地算力硬件: [{gpu_name}]")
        print(f"💎 硬件物理显存探测: {gpu_mem:.2f} GB")
        print("⚡ CUDA 核心加速引擎已满载启动！准备榨干显卡性能...")
    else:
        print("⚠️ 警告: 系统未检测到可用 GPU！即将切换至 CPU 降级训练，计算过程将极其漫长。")
        print("💡 解决方案: 请检查显卡驱动，或确认环境中安装了支持 CUDA 的 PyTorch。")
    print("=" * 50 + "\n")

    model.train(
        data=str(data_yaml_path),
        device=str(device),
        workers=workers,
        imgsz=640,
        epochs=epochs,
        batch=batch_size,
    )
    print("模型训练结束！结果保存在 runs/segment/ 下。")

if __name__ == "__main__":
    main()
