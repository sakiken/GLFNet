import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from tqdm import tqdm
from utils.util import save_classification_report, save_log_file, \
    plot_confusion_matrix, plot_val_loss, plot_train_loss, plot_metrics_val, \
    plot_metrics_train

from earlystop import EarlyStopping
from val import validate_model

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, scheduler,
                save_path,log_file, class_names):

    train_losses = []
    early_stopping = EarlyStopping(patience=11, verbose=True, save_path=save_path)  # 初始化早停对象

    # 记录每个 epoch 的验证损失和指标
    val_main_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
    val_ap_values = []
    val_mean_ap = []

    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1_scores = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(log_file, 'w') as f:
        for epoch in range(num_epochs):
            model.train()
            running_train_loss = 0.0
            pbar = tqdm(train_dataloader, desc=f'Epoch [{epoch + 1}/{num_epochs}]', leave=False)

            all_labels = []
            all_predictions = []

            for i, (images, local_features, labels) in enumerate(pbar):
                labels = labels.long().cuda()
                images = images.float().cuda()
                local_features = local_features.float().cuda()

                optimizer.zero_grad()
                outputs = model(images,local_features)

                # 计算任务损失
                loss_main = criterion(outputs, labels)

                loss_main.backward()  # 反向传播
                optimizer.step()

                running_train_loss += loss_main.item()  # 记录任务损失

                pbar.set_postfix_str(f'Batch {i}, Loss: {loss_main.item():.4f}')  # 更新损失进度条

                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

            train_loss = running_train_loss / len(train_dataloader)

            # 计算训练集的指标
            train_accuracy = accuracy_score(all_labels, all_predictions)
            train_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
            train_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=1)
            train_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=1)

            # 验证模型
            (avg_val_loss_main,
             val_accuracy, val_precision, val_recall, val_f1, ap_values, mean_ap,
             val_labels, val_predictions) = validate_model(model, val_dataloader, criterion,class_names)

            # # 根据当前epoch数决定使用哪个指标
            use_accuracy = (epoch > 200 and avg_val_loss_main < 0.01) and early_stopping.checkpoint_triggered
            if use_accuracy:
                scheduler.step(val_accuracy)  # 使用负数因为是最大化准确率
            else:
                scheduler.step(avg_val_loss_main)  # 主要基于验证损失

            # 调用早停机制
            early_stopping(avg_val_loss_main, val_accuracy, epoch, use_accuracy)

            # 记录验证损失和指标
            val_main_losses.append(avg_val_loss_main)

            val_accuracies.append(val_accuracy)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)
            val_f1_scores.append(val_f1)
            val_ap_values.append(ap_values)
            val_mean_ap.append(mean_ap)

            train_accuracies.append(train_accuracy)
            train_precisions.append(train_precision)
            train_recalls.append(train_recall)
            train_f1_scores.append(train_f1)

            # Logging
            log_message = (
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Avg Train Loss: {train_loss:.6f}, "
                f"Validation Loss: {avg_val_loss_main:.6f}, "
                f"Val Acc: {val_accuracy:.6f}, "
                f"Val Precision: {val_precision:.6f}, "
                f"Val Recall: {val_recall:.6f}, "
                f"Val F1: {val_f1:.6f},"
                f"mAP: {mean_ap:.6f}"
            )
            print(log_message)
            f.write(log_message + '\n')

            save_classification_report(val_labels, val_predictions, class_names, save_path)  # 使用验证集的标签和预测

            train_losses.append(train_loss)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    # 保存训练损失日志
    save_log_file(train_losses, save_path)

    # 绘制验证损失曲线
    plot_train_loss(train_losses, save_path)
    plot_val_loss(val_main_losses, save_path)

    # 混淆矩阵
    plot_confusion_matrix(all_labels, all_predictions, class_names, save_path, task='train')
    plot_confusion_matrix(val_labels, val_predictions, class_names, save_path, task='val')

    # 绘制训练指标曲线
    plot_metrics_train(train_accuracies, train_precisions, train_recalls, train_f1_scores, save_path)
    # 绘制验证指标曲线
    plot_metrics_val(val_accuracies, val_precisions, val_recalls, val_f1_scores, val_mean_ap, save_path)
