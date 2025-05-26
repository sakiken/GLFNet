import cv2
import numpy as np
import torch

# 定义关键点索引  from RTMPose
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