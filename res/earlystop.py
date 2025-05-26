import numpy as np
import torch
import os


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, save_path=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = -np.Inf
        self.val_acc_max = 0  # 用于记录最高的验证准确率
        self.delta = delta
        self.save_path = save_path
        self.checkpoint_triggered = False  # 新增标志，记录是否触发了 save_checkpoint

    def __call__(self, val_loss, val_acc, epoch, use_accuracy=False):
        if use_accuracy:
            score = val_acc
            best_score = self.val_acc_max
            comparison = (score > best_score + self.delta)
            metric_name = 'accuracy'
            current_metric = val_acc

        else:
            score = -val_loss  # 将损失转换为负数，以便使用相同的比较逻辑
            best_score = self.val_loss_min
            comparison = (score > best_score + self.delta)
            metric_name = 'loss'
            current_metric = val_loss

        if self.best_score is None or comparison:
            self.best_score = score
            self.save_checkpoint(metric_name, current_metric, model)
            self.counter = 0  # 重置计数器
            self.checkpoint_triggered = True  # 设置标志为True
        else:
            self.counter += 1  # 增加计数器
            self.checkpoint_triggered = False  # 设置标志为False
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True  # 触发早停

    def save_checkpoint(self, metric_name, current_metric, model):
        """Saves model when validation metric improves."""
        if self.verbose:
            if metric_name == 'loss':
                print(
                    f'Validation {metric_name} improved ({-self.val_loss_min:.6f} --> {current_metric:.6f}). Saving model ...')
            else:
                print(
                    f'Validation {metric_name} improved ({self.val_acc_max:.6f} --> {current_metric:.6f}). Saving model ...')

        torch.save(model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))

        if metric_name == 'loss':
            self.val_loss_min = -current_metric
        else:
            self.val_acc_max = current_metric  # 更新最高准确率