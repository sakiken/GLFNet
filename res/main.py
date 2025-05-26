import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from models.model import GLFNet
from res.train import train_model
from res.test import start_test
from utils.dataloder import DistractedDriverDataset
from utils.util import create_run_directory, print_class_distribution

# 主函数
if __name__ == "__main__":

    # 设置数据集路径
    image_dir = '../datasets/state-farm-distracted-driver-detection/imgs/trains/'
    features_dir = '../datasets/state-farm-distracted-driver-detection/trains/features/'
    csv_file = '../datasets/state-farm-distracted-driver-detection/driver_imgs_list.csv'

    class_names = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    model_dict = {
        'GLFNet': GLFNet,
        # 'ResNet50': ResNet50,
        # 'ViT': ViTModel,
    }

    model_name = 'GLFNet'

    # 加载数据集
    all_dataset = DistractedDriverDataset(features_dir, image_dir, csv_file)

    # 设置随机种子，以确保结果可复现
    torch.manual_seed(42)

    # 加载CSV文件并进行分层划分
    data = pd.read_csv(csv_file)

    # 提取类别标签
    labels = data['classname'].values

    # 创建分层划分对象
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # 定义训练和验证子集
    train_indices = None
    val_indices = None

    # 执行分层划分
    for train_index, val_index in sss.split(np.zeros(len(labels)), labels):
        train_indices = train_index
        val_indices = val_index

    # 创建训练集和验证集
    train_subset = Subset(all_dataset, train_indices)
    val_subset = Subset(all_dataset, val_indices)

    # 确保每个驾驶行为的数据量均衡
    class_counts = data['classname'].value_counts()
    min_count = class_counts.min()  # 每个类别的最小数量
    total_samples_needed = len(val_indices)  # 获取原始验证集的大小

    # 创建均衡的验证集索引
    balanced_val_indices = []
    for class_name in class_counts.index:
        class_indices = data[data['classname'] == class_name].index.tolist()
        samples_needed_per_class = min(min_count, total_samples_needed // len(class_counts))
        balanced_val_indices.extend(class_indices[:samples_needed_per_class])  # 选取每个类别的样本

    # 创建均衡的验证集
    balanced_val_subset = Subset(all_dataset, balanced_val_indices)

    # 打印训练集和验证集的大小
    print(f"Training set size: {len(train_subset)}")
    print(f"Original Validation set size: {len(val_subset)}")
    print(f"Balanced Validation set size: {len(balanced_val_subset)}")

    train_distribution = print_class_distribution(train_subset, data)
    val_distribution = print_class_distribution(balanced_val_subset, data)

    print("Training set class distribution:")
    for label, count in train_distribution.items():
        print(f"{label}: {count}")

    print("\nBalanced Validation set class distribution:")
    for label, count in val_distribution.items():
        print(f"{label}: {count}")

    # 使用 train_indices 和 balanced_val_subset 来创建 DataLoader
    train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(balanced_val_subset, batch_size=32, shuffle=False, num_workers=8)

    run_dir = create_run_directory(model_name)
    log_file = os.path.join(run_dir, 'training_log.txt')

    model_class = model_dict[model_name]
    model = model_class().cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = 150
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True,
                                                     min_lr=1e-6)

    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, scheduler,
                run_dir, log_file, class_names)

    # 测试
    model_name = 'GLFNet'
    start_test(model_dict, model_name, run_dir)