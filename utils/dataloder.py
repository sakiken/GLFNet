import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as F

from utils.loder import extract_local_features

class DistractedDriverDataset(Dataset):
    def __init__(self, features_dir, image_dir, csv_file, num_samples=None, expand_ratio=0.2):
        self.features_dir = features_dir
        self.image_dir = image_dir
        self.data_frame = pd.read_csv(csv_file)
        self.class_names = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])

        # 数据增强和图像转换
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
            transforms.ToTensor(),  # 转换为Tensor
        ])
        self.label_mapping = {'c0': 0, 'c1': 1, 'c2': 2, 'c3': 3, 'c4': 4,
                              'c5': 5, 'c6': 6, 'c7': 7, 'c8': 8, 'c9': 9}
        self.images_per_class = {class_name: os.listdir(os.path.join(image_dir, class_name)) for class_name in
                                 self.class_names}

        if num_samples is not None:
            self.data_frame = self.data_frame.head(num_samples)

    def __len__(self):
        return sum([len(self.images_per_class[c]) for c in self.class_names])

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 2]
        class_name = self.data_frame.iloc[idx, 1]
        subject = self.data_frame.iloc[idx, 0]

        # 如果features_dir包含subject目录
        features_path = os.path.join(self.features_dir,subject, class_name, img_name.replace('.jpg', '_features.npy'))

        # 如果features_dirb不包含subject目录
        # features_path = os.path.join(self.features_dir, class_name, img_name.replace('.jpg', '_features.npy'))

        features_path = features_path.replace("\\", "/")

        # 加载关键点数据
        keypoints_data = np.load(features_path)[0]
        keypoints = keypoints_data[:, :2]
        scores = torch.tensor(keypoints_data[:, 2], dtype=torch.float32)
        # print(f'keypoints',keypoints)
        # print(f'scores',scores)

        # 加载图像文件
        img_path = os.path.join(self.image_dir, class_name, img_name)
        image = cv2.imread(str(img_path))
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 记录原始大小
        original_size = np.array([image.width, image.height])

        # # 随机应用旋转、水平翻转等增强
        angle = np.random.uniform(-30, 30)
        if np.random.rand() > 0.5:
            image = F.hflip(image)
            keypoints[:, 0] = original_size[0] - keypoints[:, 0]

        image = F.rotate(image, angle)
        rotation_matrix = cv2.getRotationMatrix2D((original_size[0] / 2, original_size[1] / 2), angle, 1)
        keypoints = np.dot(np.column_stack((keypoints, np.ones(keypoints.shape[0]))), rotation_matrix.T)

        # 调整图像大小和关键点缩放
        image = F.resize(image, [224, 224])
        scale = np.array([224, 224]) / original_size
        keypoints *= scale

        # 映射标签到整数
        label = self.label_mapping[class_name]

        # 最终图像转换
        image_tensor = self.transform(image)

        local_features = extract_local_features(image_tensor, keypoints, scores, (224, 224), target_size=(64, 64))

        # print(f'image_tensor',image_tensor.shape)
        return ( image_tensor, local_features, torch.tensor(label, dtype=torch.long))


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
        return (image_tensor, local_features, torch.tensor(label, dtype=torch.long), img_path)