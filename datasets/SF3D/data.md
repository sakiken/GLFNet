# SF3D Dataset Structure

This repository contains the dataset and extracted features for the **State Farm Distracted Driver Detection (SF3D)** project.

---

## 📂 Directory Structure
```
SF3D/
├── imgs/
│ ├── trains/ # Training images by class
│ └── tests/ # Test images
├── trains/
│ └── features/ # Extracted keypoint features for training images
├── tests/
│ └── features/ # Extracted keypoint features for test images
├── driver_imgs_list.csv # CSV file listing training image details
└── sample_submission.csv # Example submission format for evaluation
```

---

## 📄 Description

- **imgs/trains/**: Contains raw training images, organized by class (e.g., c0, c1, ...).
- **imgs/tests/**: Contains raw test images.
- **trains/features/**: Numpy `.npy` files of extracted keypoints from training images (using RTMPose or other models).
- **tests/features/**: Numpy `.npy` files of extracted keypoints from test images.
- **driver_imgs_list.csv**: Metadata CSV listing driver IDs, class labels, and image file names for the training set.
- **sample_submission.csv**: Example CSV file showing the required format for test set predictions.

---

## ⚙️ Feature Extraction

Keypoints are extracted from the images using a pose estimation model (e.g., **RTMPose**), providing skeletal keypoint coordinates and confidence scores, which are saved as `.npy` files for downstream tasks such as classification or behavior analysis.

---

## 🚀 Usage Paths

In the training and testing pipeline, the following paths are used:

- **Training:**
  - `image_dir`: `../datasets/SF3D/imgs/trains/`
  - `features_dir`: `../datasets/SF3D/trains/features/`
  - `csv_file`: `../datasets/SF3D/driver_imgs_list.csv`

- **Testing:**
  - `images_test_dir`: `../datasets/SF3D/imgs/tests/`
  - `features_test_dir`: `../datasets/SF3D/tests/features/`

---

