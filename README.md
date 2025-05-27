# GLFNet
# 🚗 Project Driver Distraction Recognition Project

**正在整理**


**Sorting out**


# 📦 Datasets Overview

This folder contains two main datasets used in this project:

---

## 📁 SF3D

The **SF3D** dataset refers to the **State Farm Distracted Driver Detection** dataset, originally from the Kaggle competition:
https://www.kaggle.com/competitions/state-farm-distracted-driver-detection

It includes images of drivers performing various tasks (safe driving, texting, talking on the phone, etc.) for distraction behavior classification.

---

## 📁 AUC

The **AUC** dataset refers to the **AUC Distracted Driver Dataset**, which is an academic dataset published by the American University in Cairo.
It contains annotated images of drivers under different distraction categories for training and evaluating recognition models.

https://www.kaggle.com/datasets/tejakalepalle/auc-distracted-driver-dataset-v1

unzip answer Ref：  https://heshameraqi.github.io/distraction_detection

---

## ⚙️ Environment Setup

| Environment & Hyperparameter      | Details                   |
|-----------------------------------|---------------------------|
| Platform                          | NVIDIA A30 GPU (24 GB)    |
| Operating System                  | Ubuntu 20.04              |
| Framework                         | PyTorch 2.0.0             |
| Optimizer                         | SGD                      |
| Momentum                          | 0.9                      |
| Weight Decay                      | 0.00001                  |
| Max Epochs                        | 150                      |
| Batch Size                        | 32                       |
| Initial Learning Rate (LR)        | 0.001                   |
| Patience (Early Stop)             | 4                        |
| LR Decay                          | 0.1                      |
| Minimum LR                        | 0.000001                 |
| Loss Function                     | Cross-entropy Loss        |

---


## 🏗️ Models

This folder contains:
- `model.py` → My proposed main model.
- `Local_methods.py` → Comparison methods for local feature modeling.
- `Fusion_Methods.py` → Comparison methods for global-local feature fusion.
- Other files (e.g., `ConvNext.py`, `DenseNet121.py`, `FDAN.py`) → Complete backbone models used for overall performance benchmarking.

---

## 🛠️ Training & Evaluation Scripts (`res`)

This folder contains:
- `train.py` → Training pipeline.
- `val.py` → Validation pipeline.
- `test.py` → Testing pipeline.
- `main.py` → Unified run script.
- `earlystop.py` → Early stopping helper.

---

## 🔧 Utility Tools (`utils`)

This folder contains:
- `dataloader.py` → Data loading and preprocessing functions.
- `loder.py` → Auxiliary data handling helpers.
- `test_fuc.py` → Test phase helper functions.
- `util.py` → General-purpose utilities.

## 📢 Summary

This repository is designed for research on distracted driving recognition using deep learning.  
It provides a complete pipeline from datasets and feature extraction to model training, evaluation, and comparisons.  

Feel free to explore, use, or extend the code — and reach out for collaboration or questions!
