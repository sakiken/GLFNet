# AUC Dataset Structure

This repository contains the dataset and feature files for the AUC driver distraction recognition project.

---

## 📂 Directory Structure

```
auc/
├── imgs/
│ ├── tests/ # Test images organized by class
│ └── trains/ # Train images organized by class
├── tests/
│ └── features/ # Extracted keypoint features for test set
├── trains/
│ └── features/ # Extracted keypoint features for train set
├── Test_data_list.csv # CSV file listing test data details
└── Train_data_list.csv # CSV file listing train data details
```

---

## 📄 Description

- **imgs/**: Contains raw image data.
    - **trains/**: Training images, organized by class (e.g., c0, c1, c2, ...).
    - **tests/**: Testing images, organized by class.
  
- **trains/features/**: Numpy `.npy` files of extracted keypoints from training images (using RTMPose).

- **tests/features/**: Numpy `.npy` files of extracted keypoints from testing images (using RTMPose).

- **Train_data_list.csv**: Metadata or list of training samples, including image paths, labels, or additional info.

- **Test_data_list.csv**: Metadata or list of test samples, similar structure to the training CSV.

---

## ⚙️ Feature Extraction

Keypoints are extracted from images using the **RTMPose** model, providing human skeletal keypoint coordinates and confidence scores, saved as `.npy` files for further processing or model training.

---

## 🚀 Usage Example

1. Place raw images under `imgs/trains/` and `imgs/tests/`.
2. Run the Python extraction script to generate keypoint features.
3. Use the generated `.npy` files and CSV lists to train or evaluate your models.

---
