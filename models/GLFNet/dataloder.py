import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as F

# 定义关键点索引
LEFT_ARM_INDEX = [5, 7, 9]
RIGHT_ARM_INDEX = [6, 8, 10]
LEFT_HAND_INDEX = list(range(91, 112))
RIGHT_HAND_INDEX = list(range(112, 133))
FACE_INDEX = list(range(23, 91))

def generate_region_heatmap(keypoints, scores, indices, img_shape, sigma=10):
    """生成某个区域的热图。"""
    heatmap = np.zeros((img_shape[0], img_shape[1]), dtype=np.float32)

    for i in indices:
        if scores[i] > 0:
            x, y = keypoints[i]
            color_intensity = int(scores[i] * 255)
            cv2.circle(heatmap, (int(x), int(y)), sigma, color_intensity, -1)

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 将灰度热图转换为彩色热图
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap, heatmap_color


def extract_local_features(global_img, keypoints, scores, img_shape, target_size=(64, 64)):
    batch_size = global_img.size(0)
    local_features_list = []
    count = 0
    heatmaps = {
        "Left Hand": generate_region_heatmap(keypoints, scores, LEFT_HAND_INDEX, img_shape),
        "Right Hand": generate_region_heatmap(keypoints, scores, RIGHT_HAND_INDEX, img_shape),
        "Face": generate_region_heatmap(keypoints, scores, FACE_INDEX, img_shape)
    }

    local_features = []
    for region, (heatmap, heatmap_color) in heatmaps.items():
        # 使用热图作为掩码来确定裁剪边界
        _, binary_mask = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(binary_mask)
        if w > 0 and h > 0:
            cropped_image = global_img[ :, y:y + h, x:x + w].cpu().numpy()
            cropped_image = np.transpose(cropped_image, (1, 2, 0))  # 转换通道顺序
            cropped_image = (cropped_image * 255).astype(np.uint8)  # 将值转换为 0-255 范围
            cropped_image = cv2.resize(cropped_image, target_size)
            # cv2.imwrite(f'_{region}_{count}.png', cropped_image)  # 保存图像
            cropped_image = torch.tensor(cropped_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            # count += 1
        else:
            cropped_image = torch.zeros((1, 3, *target_size), dtype=torch.float32)
        local_features.append(cropped_image)
    local_features = torch.cat(local_features, dim=1)
    local_features_list.append(local_features)
    local_features = torch.cat(local_features_list, dim=0)
    return local_features

class DistractedDriverDataset_test(Dataset):
    def __init__(self, features_dir, image_dir, num_samples=None, expand_ratio=0.2):
        self.features_dir = features_dir
        self.image_dir = image_dir
        self.class_names = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])

        # 图像转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.label_mapping = {'c0': 0, 'c1': 1, 'c2': 2, 'c3': 3, 'c4': 4,
                              'c5': 5, 'c6': 6, 'c7': 7, 'c8': 8, 'c9': 9}

        # 创建一个图像列表，每个元素包含 (class_name, img_name)
        self.image_list = []
        for class_name in self.class_names:
            image_files = os.listdir(os.path.join(image_dir, class_name))
            self.image_list.extend([(class_name, img_name) for img_name in image_files])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        class_name, img_name = self.image_list[idx]

        features_path = os.path.join(self.features_dir, class_name, img_name.replace('.jpg', '_features.npy'))
        features_path = features_path.replace("\\", "/")

        # 加载关键点数据
        keypoints_data = np.load(features_path)[0]
        keypoints = keypoints_data[:, :2]
        scores = torch.tensor(keypoints_data[:, 2], dtype=torch.float32)

        # 加载图像文件
        img_path = os.path.join(self.image_dir, class_name, img_name)

        try:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Image not loaded: {img_path}")
                return None
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Error loading image: {img_path}, {e}")
            return None

        # 记录原始大小
        original_size = np.array([image.width, image.height])

        # 调整图像大小和关键点缩放
        image = F.resize(image, [224, 224])
        scale = np.array([224, 224]) / original_size
        keypoints *= scale

        # 映射标签到整数
        label = self.label_mapping[class_name]

        # 最终图像转换
        image_tensor = self.transform(image)
        local_features = extract_local_features(image_tensor, keypoints, scores, (224, 224), target_size=(64, 64))
        return (image_tensor, local_features, torch.tensor(label, dtype=torch.long))