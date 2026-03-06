import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_academic_plots(csv_path="runs/segment/train/results.csv", output_dir="runs/segment/train/academic_plots"):
    """
    读取 YOLO 训练产生的 results.csv，生成高质量的学术级图表（支持论文、报告插入）
    """
    if not os.path.exists(csv_path):
        print(f"❌ 未找到训练日志文件: {csv_path}")
        print("💡 请先使用 `uv run pavement-train` 完成至少一轮训练。")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 整理数据表
    df = pd.read_csv(csv_path)
    # YOLO 的列名通常带有前置空格，清理掉
    df.columns = df.columns.str.strip()
    
    # 设置学术级字体和绘图风格 (类似 IEEE 论文风格)
    sns.set_theme(style="whitegrid", context="paper")
    
    # 完美解决 Windows 下 matplotlib 中文字体显示问题
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]  # 优先使用黑体或微软雅黑
    plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.2
    
    epochs = df['epoch']

    # ==========================================
    # 图 1: 训练与验证的 Loss 走势比较 (Box/Seg/Cls)
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
    
    # Box Loss
    axes[0].plot(epochs, df['train/box_loss'], label='训练集定位损失 (Train Box Loss)', color='#1f77b4', linewidth=2)
    axes[0].plot(epochs, df['val/box_loss'], label='验证集定位损失 (Val Box Loss)', color='#ff7f0e', linestyle='--', linewidth=2)
    axes[0].set_title("边界框定位损失", fontweight='bold')
    axes[0].set_xlabel("训练轮次 (Epochs)")
    axes[0].set_ylabel("损失值 (Loss)")
    axes[0].legend()

    # Segmentation Loss
    axes[1].plot(epochs, df['train/seg_loss'], label='训练集掩码损失 (Train Seg Loss)', color='#2ca02c', linewidth=2)
    axes[1].plot(epochs, df['val/seg_loss'], label='验证集掩码损失 (Val Seg Loss)', color='#d62728', linestyle='--', linewidth=2)
    axes[1].set_title("多边形分割损失", fontweight='bold')
    axes[1].set_xlabel("训练轮次 (Epochs)")
    axes[1].legend()

    # Class Loss
    axes[2].plot(epochs, df['train/cls_loss'], label='训练集分类损失 (Train Cls Loss)', color='#9467bd', linewidth=2)
    axes[2].plot(epochs, df['val/cls_loss'], label='验证集分类损失 (Val Cls Loss)', color='#8c564b', linestyle='--', linewidth=2)
    axes[2].set_title("类别分类损失", fontweight='bold')
    axes[2].set_xlabel("训练轮次 (Epochs)")
    axes[2].legend()

    plt.tight_layout()
    loss_out = os.path.join(output_dir, "Academic_Loss_Curves.png")
    plt.savefig(loss_out, dpi=600, bbox_inches='tight')
    plt.savefig(loss_out.replace('.png', '.pdf'), bbox_inches='tight') # 生成高清PDF
    print(f"✅ 学术级中文 Loss 图表已导出至: {loss_out}")
    plt.close()

    # ==========================================
    # 图 2: 模型精度评估 (mAP50-95 与 mAP50)
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    ax.plot(epochs, df['metrics/mAP50(B)'], label='边界框精度 (Bbox mAP@0.5)', color='#1f77b4', linewidth=2.5)
    ax.plot(epochs, df['metrics/mAP50-95(B)'], label='综合边界框精度 (Bbox mAP@0.5:0.95)', color='#aec7e8', linestyle='-.', linewidth=2)
    ax.plot(epochs, df['metrics/mAP50(M)'], label='掩码分割精度 (Mask mAP@0.5)', color='#2ca02c', linewidth=2.5)
    ax.plot(epochs, df['metrics/mAP50-95(M)'], label='综合掩码分割精度 (Mask mAP@0.5:0.95)', color='#98df8a', linestyle='-.', linewidth=2)

    ax.set_title("模型验证集精确度 (Validation mAP)", fontweight='bold', fontsize=14)
    ax.set_xlabel("训练轮次 (Epochs)", fontsize=12)
    ax.set_ylabel("平均运算精度 (Mean Average Precision)", fontsize=12)
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.7)

    acc_out = os.path.join(output_dir, "Academic_Accuracy_mAP.png")
    plt.savefig(acc_out, dpi=600, bbox_inches='tight')
    plt.savefig(acc_out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ 学术级中文 mAP 精度图表已导出至: {acc_out}")
    plt.close()

if __name__ == "__main__":
    generate_academic_plots()
