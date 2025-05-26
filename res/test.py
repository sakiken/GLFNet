import os
import time
from collections import Counter

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from models.model import GLFNet
from utils.dataloder import DistractedDriverDataset_test
from tqdm import tqdm  # 导入 tqdm

from utils.test_fuc import plot_confusion_matrix, check_dataset_integrity, save_classification_report

# 配置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 定义测试函数
def test_model(model, dataloader, device, class_names, save_path):
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_scores = []

    top1_correct = 0
    top5_correct = 0
    acc = 0
    precision = 0
    recall = 0
    f1 = 0
    mAP = 0
    top1_accuracy = 0

    start_time = time.time()  # 记录开始时间

    total_samples = 0

    with torch.no_grad():
        for images, local_features, labels ,save_path1 in tqdm(
                dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            local_features = local_features.to(device)

            # 获取模型输出
            outputs = model(images,local_features)
            # outputs = model(images)
            # 选择输出的类别
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

            # 保存每个类别的分数
            all_scores.append(outputs.cpu().numpy())

            # 更新总样本数
            total_samples += len(labels)

            # 计算 Top-1 和 Top-5 准确率
            top1 = torch.argmax(outputs, dim=1)
            top5 = torch.topk(outputs, k=5, dim=1)[1]

            top1_correct += (top1 == labels).sum().item()
            top5_correct += (labels.unsqueeze(1) == top5).any(dim=1).sum().item()

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time
    fps = total_samples / elapsed_time  # 计算 FPS

    # 将所有批次的分数合并
    all_scores = np.concatenate(all_scores, axis=0)

    # 保存预测结果
    predictions_df = pd.DataFrame({
        'true_class': all_true_labels,
        'true_class_name': [class_names[i] for i in all_true_labels],
        'predicted_class': all_predictions,
        'predicted_class_name': [class_names[i] for i in all_predictions]
    })

    predictions_df.to_csv(os.path.join(save_path, 'predictions_with_labels.csv'), index=False)


    # 计算并保存性能指标
    if all_true_labels:
        true_labels = np.array(all_true_labels)

        # 计算 mAP
        aps = [average_precision_score((true_labels == i).astype(int), all_scores[:, i]) for i in range(len(class_names))]
        mAP = np.mean(aps)

        # 计算 Precision, Recall, F1 Score
        acc = accuracy_score(true_labels, all_predictions)
        precision = precision_score(true_labels, all_predictions, average='weighted', zero_division=1)
        recall = recall_score(true_labels, all_predictions, average='weighted', zero_division=1)
        f1 = f1_score(true_labels, all_predictions, average='weighted', zero_division=1)

        # 计算 Top-1 和 Top-5 准确率
        top1_accuracy = top1_correct / total_samples
        top5_accuracy = top5_correct / total_samples

        print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, mAP: {mAP:.4f}, Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}, FPS: {fps:.2f}")


        # 保存性能指标到文件
        with open(os.path.join(save_path, 'performance_metrics.txt'), 'w') as f:
            f.write(f"mAP: {mAP:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"FPS: {fps:.2f}\n")
            f.write(f"Top-1 Accuracy: {top1_accuracy:.4f}\n")
            f.write(f"Top-5 Accuracy: {top5_accuracy:.4f}\n")

        # 绘制混淆矩阵
        plot_confusion_matrix(true_labels, all_predictions, class_names,
                               os.path.join(save_path, 'confusion_matrix_percent.png'), normalize='true')

        # 保存分类报告
        save_classification_report(true_labels, all_predictions, class_names, save_path)

    return (acc, precision,recall, f1, mAP, top1_accuracy,fps)



def start_test(model_dict, model_name, model_path):

    # 数据集路径
    features_test_dir = '../datasets/state-farm-distracted-driver-detection/tests/features'
    images_test_dir = '../datasets/state-farm-distracted-driver-detection/imgs/tests'

    # 检查数据集完整性
    check_dataset_integrity(images_test_dir)

    # 保存路径
    model_weights_path = os.path.join(model_path, 'best_model.pth')

    #超参数
    batch_size = 32
    num_workers = 8

    class_names = ['C0', 'C1','C2', 'C3','C4', 'C5', 'C6', 'C7','C8', 'C9']

    if not os.path.exists(model_weights_path):
        os.makedirs(model_weights_path)

    # 创建数据加载器
    test_dataset = DistractedDriverDataset_test(features_test_dir, images_test_dir)

    # 统计每个类别的样本数量
    class_counts = Counter()

    for idx in range(len(test_dataset)):
        _, _, _, label = test_dataset[idx]
        class_counts[label.item()] += 1

    # 打印每个类别的样本数量
    for class_idx, count in sorted(class_counts.items()):
        class_name = list(test_dataset.label_mapping.keys())[list(test_dataset.label_mapping.values()).index(class_idx)]
        print(f"Class {class_name} has {count} samples")

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 模型和设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class = model_dict[model_name]
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    # 开始测试
    test_model(model, test_dataloader, device, class_names, model_path)

if __name__ == '__main__':

    model_name = 'GLFNet'
    model_dict = {
        'GLFNet': GLFNet,
        # 'ResNet50': ResNet50,
        # 'ViT': ViTModel,
    }
    model_path = '../run/othermodel/GLFNet/test_result/'
    start_test(model_dict, model_name, model_path)