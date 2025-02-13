# 🦴 X-Ray Fracture Classification Using Deep Learning

## 📌 Project Overview

This project aims to classify X-ray images into **7 different fracture types** using **deep learning**.  
We fine-tuned **ResNet50** on an X-ray fracture dataset, experimenting with **data augmentations, Mixup training, fine-tuning strategies, and learning rate schedules** to improve classification accuracy.

## 🏥 **Dataset**

- The dataset consists of **X-ray images** categorized into **7 fracture types**:
  - **Avulsion Fracture**
  - **Comminuted Fracture**
  - **Fracture Dislocation**
  - **Curved Fractures**
  - **Linear Fractures**
  - **Internal Fractures**
  - **Hairline Fracture**
- The dataset was **cleaned, resized (224x224), normalized**, and augmented with **random rotations, color jitter, and elastic transformations**.

## 🏗️ **Model Architecture & Training**

We experimented with **multiple training strategies**, ultimately selecting **ResNet50** as the best-performing model.

### ✅ **Final Model: ResNet50 Fine-Tuned**

- **Architecture:** ResNet50 with a modified fully connected layer for **7-class classification**.
- **Preprocessing:** Images resized to **224x224**, normalized with **ImageNet mean & std**.
- **Optimizer:** `AdamW` with weight decay (`1e-2`).
- **Learning Rate Schedule:** Used `OneCycleLR` for stabilization.
- **Data Augmentations:** Random cropping, flipping, brightness adjustment, and **Mixup Training**.

### 🧪 **Experiments Conducted**

| **Experiment**                                                          | **Outcome**                  |
| ----------------------------------------------------------------------- | ---------------------------- |
| **Baseline ResNet50** (pretrained weights)                              | 42.7% accuracy               |
| **Data Augmentation (rotation, brightness, color jitter)**              | 45.3% accuracy               |
| **Fine-Tuning Full Model**                                              | 48.6% accuracy               |
| **Fine-Tuning with Regularization (Dropout & Weight Decay)**            | 53.6% accuracy               |
| **Final Optimized Training (Lower LR, Augmentations, SGD fine-tuning)** | **55.0% accuracy (best)** 🎯 |

## 🔥 **Grad-CAM Visualizations**

To understand what the model **focuses on** when classifying fractures, we used **Grad-CAM** to generate heatmaps.  
Here are some **example visualizations**:

| **Original X-ray**     | **Grad-CAM Heatmap**     |
| ---------------------- | ------------------------ |
| ![X-ray 1](./images/1) | ![Heatmap 1](./images/2) |
| ![X-ray 2](./images/3) | ![Heatmap 2](./images/4) |

## 📈 **Final Results**

- **Best Test Accuracy:** **55.0%** (ResNet50 Fine-Tuned)
- **Model struggles with subtle fractures (hairline, internal).**
- **Overcomes dataset limitations with augmentations & Mixup Training.**

## 🚀 **Future Work**

- ✅ **Test EfficientNet** (More medical image-friendly model).
- ✅ **Explore Object Detection (YOLOv8/Faster R-CNN)** for localizing fractures. (Although, I can't really tell fractures)
- ✅ **Increase Dataset Size** to improve generalization.
- ✅ **Test Self-Supervised Learning (SimCLR, BYOL).**
