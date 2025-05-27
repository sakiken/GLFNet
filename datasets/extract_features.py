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