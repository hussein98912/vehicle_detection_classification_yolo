# 🚗 Vehicle Detection and Classification using YOLO

This project applies deep learning to detect and classify vehicles such as **cars**, **buses**, **trucks**, and **motorcycles** , **ambulance** . It uses multiple YOLO variants and data augmentation strategies to achieve high accuracy.

> Developed in a single Jupyter Notebook and trained on a public dataset from Kaggle.

---

## 📘 Notebook

- `vehicle_detection_and_classification.ipynb`: End-to-end pipeline including data preprocessing, training, evaluation, and inference using YOLOv8 and YOLOv11 models.

---

## 📊 Experimental Results

Fifteen experiments were conducted to tune performance using different YOLO versions, learning rates, label strategies, and training settings:

| #  | Method                                | Epochs | Batch | LR      | YOLO Ver. | Acc. | Freeze     | Dropout |
|----|----------------------------------------|--------|--------|----------|-----------|------|------------|---------|
| 1  | Data augmentation (all labels)         | 150    | 32     | 0.0001   | YOLOv8n   | 59%  | No         | No      |
| 2  | Same as above                          | 100    | 16     | //       | //        | 65%  | No         | No      |
| 3  | Remove class "car"                     | 100    | 16     | //       | YOLOv8s   | 37%  | No         | No      |
| 4  | Same as above                          | 100    | 16     | //       | YOLOv8n   | 40%  | No         | No      |
| 5  | Reduce "car" class to 250 samples      | 100    | 16     | //       | YOLOv8s   | 67%  | No         | No      |
| 6  | Augment only "car" and "truck"         | 100    | 16     | //       | YOLOv8s   | 56%  | No         | No      |
| 7  | Set manual class weights               | 100    | 32     | 0.01     | YOLOv8n   | 60%  | No         | No      |
| 8  | Balance all label classes              | 100    | 64     | 0.0001   | YOLOv8n   | 57%  | No         | No      |
| 9  | Try YOLOv11n + dropout                 | 120    | 32     | 0.001    | YOLOv11n  | 59%  | No         | 0.3     |
| 10 | Freeze 15 layers                       | 120    | 8      | 0.0001   | YOLOv11m  | 72%  | 15 layers  | 0.3     |
| 11 | Freeze + data augmentation             | 80     | 16     | 0.0003   | YOLOv11m  | 74%  | 10 layers  | 0.4     |
| 12 | Freeze + data augmentation             | 80     | 16     | 0.0001   | YOLOv11m  | 75%  | 15 layers  | 0.4     |
| 13 | Add more validation data               | 80     | 16     | 0.001    | YOLOv11m  | 85%  | 20 layers  | 0.5     |
| 14 | Try YOLOv11l                           | 150    | 32     | 0.001    | YOLOv11l  | 81%  | 16 layers  | 0.5     |
| 15 | YOLOv11l fine-tuning                   | 150    | 32     | 0.001    | YOLOv11l  | 84%  | 20 layers  | 0.5     |

> 🏆 **Best Accuracy**: 85% with YOLOv11m using frozen layers, augmentation, and extended validation set.

---

## 📂 Dataset

- **Dataset Name**: [Vehicle Detection Dataset by Alkaner Turan](https://www.kaggle.com/datasets/alkanerturan/vehicledetection)
- **Classes**: car, bus, truck, motorbike , Ambulance
- **License**: Public (Kaggle usage)

> Data was augmented, cleaned, and rebalanced for better performance.

---

## 📦 Installation

Install required packages:

```bash
pip install -r requirements.txt
