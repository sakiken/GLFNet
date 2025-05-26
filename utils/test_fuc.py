import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize=normalize)
    cm_percentage = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 10))
    im_ = disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f' if normalize else 'd', colorbar=False)  # 禁用默认颜色刻度条

    # 设置x轴和y轴的标题
    ax.set_xlabel('Predicted', fontsize=16, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=16, fontweight='bold')

    # 设置横坐标标签旋转避免重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # 添加网格线在每个格子的四周
    ax.set_xticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.75)
    ax.tick_params(which='minor', bottom=False, left=False)

    # 隐藏横轴和纵轴的刻度线样式
    ax.xaxis.set_tick_params(labelbottom=True, bottom=False)
    ax.yaxis.set_tick_params(labelleft=True, left=False)

    # 手动添加颜色刻度条，并确保它只显示一次
    cbar = fig.colorbar(im_.im_, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.set_ticks_position('right')

    plt.tight_layout()  # 调整布局以防止标签重叠
    plt.savefig(save_path)
    plt.close(fig)


def plot_binary_confusion_matrix_from_multiclass(y_true, y_pred, class_names, save_path):
    # 计算原多分类混淆矩阵（百分比形式）
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_percentage = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100

    # 索引0假设是Safe Driving，其余是Distracted Driving
    safe_idx = 0
    distracted_indices = list(range(1, len(class_names)))

    # 左上：Safe → Safe
    safe_to_safe = cm_percentage[safe_idx, safe_idx]

    # 右上：Safe → Distracted
    safe_to_distracted = cm_percentage[safe_idx, distracted_indices].sum()

    # 左下：Distracted → Safe
    distracted_to_safe = cm_percentage[distracted_indices, safe_idx].sum()

    # 右下：Distracted → Distracted
    distracted_to_distracted = cm_percentage[distracted_indices, :][:, distracted_indices].diagonal().sum()

    # 构造 2x2 confusion matrix
    binary_cm = np.array([
        [safe_to_safe, safe_to_distracted],
        [distracted_to_safe, distracted_to_distracted]
    ])

    # 绘制
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(binary_cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Safe Driving', 'Distracted Driving'], rotation=45)
    ax.set_yticklabels(['Safe Driving', 'Distracted Driving'])

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Binary Confusion Matrix (Aggregated)')

    # 在格子中写数值
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{binary_cm[i, j]:.2f}%', ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def save_classification_report(y_true, y_pred, class_names, save_path):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=1)
    df = pd.DataFrame(report).transpose()
    df.index.name = 'class_name'  # 设置索引名为 class_name
    df.reset_index(inplace=True)  # 将索引转换为列
    df.to_csv(os.path.join(save_path, 'classification_report.csv'), index=False)


def check_dataset_integrity(dataset_dir):
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            num_files = len(os.listdir(class_dir))
            print(f"Class {class_name} has {num_files} files.")