import torch
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, task):
    """
    绘制并保存混淆矩阵图
    """
    cm = confusion_matrix(y_true, y_pred)
    # 将数值转换为百分比
    cm_percentage = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f')
    # 设置x轴标签旋转，避免重叠
    plt.xticks(rotation=45, ha='right')
    # 调整布局，避免标签被裁剪
    plt.tight_layout()
    if task == 'train':
        save_path = os.path.join(save_path, 'train_confusion_matrix_percentage.png')
    else:
        save_path = os.path.join(save_path, 'Val_confusion_matrix_percentage.png')
    plt.savefig(save_path)
    plt.close()

def plot_training_loss(losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss', linewidth=1, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'training_loss.png'))
    plt.close()

def save_classification_report(y_true, y_pred, class_names, save_path):
    """
    生成并保存分类报告
    """
    report = classification_report(y_true, y_pred, target_names=class_names,zero_division=1)
    with open(os.path.join(save_path, 'classification_report.txt'), 'w') as report_file:
        report_file.write(report)
    # print("Classification Report:")
    # print(report)

def save_log_file(losses, save_path):

    with open(os.path.join(save_path, 'training_loss.log'), 'w') as f:
        for loss in losses:
            f.write(f"{loss}\n")
def save_tesetlog_file(losses, save_path):

    with open(os.path.join(save_path, 'training_loss.log'), 'w') as f:
        for loss in losses:
            f.write(f"{loss}\n")

def create_run_directory(title,base_dir='../run', prefix='train'):
    # 获取当前时间戳
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # 创建一个基于时间的新目录
    run_dir = os.path.join(base_dir, f"{title}_{prefix}_{timestamp}")
    # 如果目录不存在，则创建它
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    return run_dir

# def plot_val_loss(main_losses, global_losses, local_losses, keypoint_losses, save_path):
#     """
#     绘制各个任务损失随epoch变化的图像并保存。
#
#     :param main_losses: 主损失列表
#     :param global_losses: 全局损失列表
#     :param local_losses: 局部损失列表
#     :param keypoint_losses: 关键点损失列表
#     :param save_path: 保存图像的路径
#     """
#     epochs = range(1, len(main_losses) + 1)
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, main_losses, label='Main Loss', marker='o')
#     plt.plot(epochs, global_losses, label='Global Loss', marker='o')
#     plt.plot(epochs, local_losses, label='Local Loss', marker='o')
#     plt.plot(epochs, keypoint_losses, label='Keypoint Loss', marker='o')
#     plt.title('Losses Over Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid()
#     plt.savefig(os.path.join(save_path, 'losses_over_epochs.png'))  # 保存图像

def plot_val_loss(main_losses, save_path):
    """
    绘制各个任务损失随epoch变化的图像并保存。

    :param main_losses: 主损失列表
    :param global_losses: 全局损失列表
    :param local_losses: 局部损失列表
    :param keypoint_losses: 关键点损失列表
    :param save_path: 保存图像的路径
    """
    epochs = range(1, len(main_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, main_losses, label='Main Loss', marker='o')
    plt.title('Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, 'validation_losses.png'))  # 保存图像

def plot_train_loss(train_losses, save_path):
    # 绘制损失曲线
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    # plt.plot(val_main_losses, label='Val Loss', color='orange')
    plt.title('Train and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Train_losses.png'))
    plt.close()

def plot_metrics_val(accuracies, precisions, recalls, f1_scores, mean_ap, save_path):
    # 绘制各个指标曲线
    plt.figure(figsize=(12, 8))
    plt.plot(accuracies, label='Accuracy', color='blue')
    plt.plot(precisions, label='Precision', color='orange')
    plt.plot(recalls, label='Recall', color='green')
    plt.plot(f1_scores, label='F1 Score', color='red')
    plt.plot(mean_ap, label='Mean AP', color='purple')  # 新增 AP 曲线
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'validation_metrics.png'))
    plt.close()

def plot_metrics_train(accuracies, precisions, recalls, f1_scores, save_path):
    # 绘制各个指标曲线
    plt.figure(figsize=(12, 8))
    plt.plot(accuracies, label='Accuracy', color='blue')
    plt.plot(precisions, label='Precision', color='orange')
    plt.plot(recalls, label='Recall', color='green')
    plt.plot(f1_scores, label='F1 Score', color='red')
    plt.title('Train Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Train_metrics.png'))
    plt.close()

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j


def check_labels(dataloader, n_classes):
    for image, features, labels in dataloader:
        if not ((labels >= 0) & (labels < n_classes)).all():
            print(f"Invalid labels found: {labels}")
            print(f"Label range: {labels.min()} to {labels.max()}")
            print(f"Expected range: 0 to {n_classes-1}")
            return False
    print("All labels are valid.")
    return True

#GPU内存使用情况
def get_gpu_memory_usage():
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # 转换为GB
    reserved_memory = torch.cuda.memory_reserved() / 1024**3  # 转换为GB
    max_allocated_memory = torch.cuda.max_memory_allocated() / 1024**3  # 转换为GB
    max_reserved_memory = torch.cuda.max_memory_reserved() / 1024**3  # 转换为GB
    return allocated_memory, reserved_memory, max_allocated_memory, max_reserved_memory

# 打印每个类别在训练集和验证集中的数量
def print_class_distribution(dataset, data):
    labels = data['classname'].values
    label_counts = {label: 0 for label in np.unique(labels)}

    for idx in dataset.indices:
        label = labels[idx]
        label_counts[label] += 1

    return label_counts
