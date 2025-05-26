import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from tqdm import tqdm


def validate_model(model, val_dataloader, criterion,class_names):
    model.eval()
    val_labels = []
    val_predictions = []
    val_probabilities = []  # 用于保存概率分布

    # 初始化每种损失的列表
    loss_main_list = []

    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc='Validating', leave=False)  # 添加进度条
        for i, (images, local_features, labels) in enumerate(pbar):
            labels = labels.long().cuda()  # 使用 CUDA
            images = images.float().cuda()
            local_features = local_features.float().cuda()
            outputs = model(images,local_features)
            # outputs = model(images)
            # 计算损失
            loss_main = criterion(outputs, labels)
            # 将损失添加到列表中
            loss_main_list.append(loss_main.item())

            _, predicted = torch.max(outputs, 1)
            val_labels.extend(labels.cpu().numpy())
            val_predictions.extend(predicted.cpu().numpy())
            val_probabilities.extend(outputs.softmax(dim=1).cpu().numpy())  # 保存每个类别的概率

            # 更新进度条
            pbar.set_postfix_str(f'Loss: {loss_main.item():.4f}')  # 更新主任务损失进度条

    # 计算平均损失
    avg_val_loss_main = np.mean(loss_main_list)

    # 计算验证集的指标
    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_precision = precision_score(val_labels, val_predictions, average='weighted', zero_division=1)
    val_recall = recall_score(val_labels, val_predictions, average='weighted', zero_division=1)
    val_f1 = f1_score(val_labels, val_predictions, average='weighted', zero_division=1)

    # 计算每一类的AP值
    ap_values = []
    val_labels_onehot = np.eye(len(class_names))[val_labels]  # 将标签转换为one-hot格式
    for i in range(len(class_names)):
        ap = average_precision_score(val_labels_onehot[:, i], np.array(val_probabilities)[:, i])
        ap_values.append(ap)

    # 计算mAP
    mean_ap = np.mean(ap_values)

    # 返回所有损失的平均值和验证集的指标
    return (avg_val_loss_main,
            val_accuracy, val_precision, val_recall, val_f1, ap_values, mean_ap,
            val_labels, val_predictions)  # 返回所有损失的平均值