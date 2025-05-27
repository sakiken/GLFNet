"""
脚本名称：extract_keypoints_rtmpose.py

脚本作用：
本脚本基于 mmpose 框架，使用 RTMPose 模型对给定图像数据集中每张图片的人体关键点进行检测，并将提取到的关键点信息（包括关键点坐标和对应置信度分数）保存为 .npy 文件，供后续特征分析、行为识别或机器学习任务使用。

主要功能：
✅ 初始化 RTMPose 全身关键点检测模型；
✅ 遍历指定图像数据集（按照类别组织的文件夹结构）；
✅ 对每张图像进行人体关键点检测；
✅ 提取关键点坐标（shape: [人数, 133, 2]）和关键点置信度（shape: [人数, 133]）；
✅ 将关键点坐标和置信度拼接为 shape: [人数, 133, 3]，并以 .npy 文件格式保存到输出目录。

输入：
- 图像数据集目录（image_dir），按照类别分文件夹组织，如：
    datasets/auc/imgs/trains/c0/*.jpg
    datasets/auc/imgs/trains/c1/*.jpg
    ...
- RTMPose 配置文件和权重文件，用于加载关键点检测模型。

输出：
- 保存关键点信息的 .npy 文件，存放在 output_root_dir 下，结构为：
    datasets/auc/trains/features/c0/xxx_features.npy
    datasets/auc/trains/features/c1/xxx_features.npy
    ...

依赖：
- mmpose
- OpenCV (cv2)
- numpy
- tqdm

使用说明：
运行前请确保：
1️⃣ 已安装 mmpose 及其依赖库；
2️⃣ 提供的 config_file 和 checkpoint_file 路径正确；
3️⃣ 输入图像目录和输出目录存在或可创建。



作者：
sakiken

最后更新日期：
2025-05-27
"""

import os
import cv2
import numpy as np
from mmpose.apis import inference_topdown, init_model
from tqdm import tqdm

# 初始化关键点检测模型
config_file = 'projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py'
checkpoint_file = 'weights/rtmpose/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth'
keypoint_model = init_model(config_file, checkpoint_file, device='cuda:0')

# 准备保存关键点数据的根目录
output_root_dir = 'datasets/auc/trains/features'
if not os.path.exists(output_root_dir):
    os.makedirs(output_root_dir)

# 定义类别映射
class_mapping = {
    'c0': 0, 'c1': 1, 'c2': 2, 'c3': 3, 'c4': 4,
    'c5': 5, 'c6': 6, 'c7': 7, 'c8': 8, 'c9': 9
}

# 遍历所有图像并提取关键点
image_dir = 'datasets/auc/imgs/trains'
classes = os.listdir(image_dir)

for class_name in tqdm(classes, desc='Processing classes'):
    class_dir = os.path.join(image_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    subject_dir = os.path.join(output_root_dir, class_name)
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)

    for img_name in os.listdir(class_dir):
        if img_name.endswith('.jpg'):
            image_path = os.path.join(class_dir, img_name)

            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"图像文件不存在或路径错误：{image_path}")
                continue

            # 使用关键点检测模型提取关键点
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像：{image_path}")
                continue

            results = inference_topdown(keypoint_model, image)
            if results:
                keypoints = results[0].pred_instances.keypoints  # 提取关键点坐标
                keypoint_scores = results[0].pred_instances.keypoint_scores  # 提取关键点置信度分数

                # 将关键点和置信度分数合并
                keypoints_with_scores = np.concatenate((keypoints, keypoint_scores[:, :, np.newaxis]), axis=2)

                # 构建保存路径
                filename = f"{img_name.split('.')[0]}_features.npy"
                save_path = os.path.join(subject_dir, filename)

                # 保存关键点数据
                np.save(save_path, keypoints_with_scores)
            else:
                print(f"未检测到关键点：{image_path}")

print("关键点数据提取和保存完成。")
