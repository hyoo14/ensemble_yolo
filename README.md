# Ensemble Yolo

This project implements an ensemble-based approach for vehicle detection, classification, and counting using YOLO models. The method integrates YOLOv5, YOLOv8, and YOLOv10 to enhance detection accuracy and robustness.





## Overview

Deep learning-based object detection models, such as YOLO, perform well in real-time vehicle detection but may suffer from false positives or missed detections. This project employs an ensemble strategy that merges predictions from multiple YOLO models using an **Intersection over Union (IoU)-based fusion technique**, improving recall and reducing detection errors.

## Methodology

1. **YOLO Model Selection & Inference**  
   - Utilizes YOLOv5, YOLOv8, and YOLOv10, pre-trained on the COCO dataset.
   - Each model generates bounding boxes with confidence scores.

2. **Ensemble Strategy**  
   - Merges overlapping detections using an IoU threshold (τ=0.5).
   - Averages bounding box coordinates while retaining the highest confidence score.

3. **Evaluation Metrics**  
   - **Mean Average Precision (mAP):** Measures detection accuracy.
   - **F1-score:** Balances precision and recall.
   - **Mean Absolute Error (MAE):** Evaluates vehicle counting accuracy.

## Results

The ensemble model **outperforms individual YOLO nano models** in F1-score and MAE, demonstrating improved robustness with COCO 2017 validation dataset:

| Model  | Precision | Recall  | F1-score | MAE (Vehicle Count) |
|--------|----------|--------|---------|------------------|
| YOLOv5  | 1.0000   | 0.8576 | 0.8576  | 1.7809 |
| YOLOv8  | 1.0000   | 0.7773 | 0.8576  | 1.7809 |
| YOLOv10 | 1.0000   | 0.7287 | 0.8576  | 1.9560 |
| **Ensemble** | 1.0000   | 0.8148 | **0.8980**  | **1.6609** |

The ensemble model achieves the **highest recall and F1-score**, minimizing false negatives while improving vehicle counting accuracy.

## Dataset

- **COCO Dataset**: Used for model training and evaluation.
- **KITTI Dataset (Appendix)**: Additional evaluation to validate performance across diverse real-world scenes.

## How to Run Example code

This project contains an example code that can be executed in Google Colab.  
You can run it directly from the provided Colab link.


1. Open the [Google Colab link](https://colab.research.google.com/drive/1ukYChz8LGlad3PC3VaG6tS8ugEb1KgsZ?usp=sharing).
2. Go to **Runtime > Change runtime type** and select the appropriate environment (GPU/TPU if needed).
3. Run all cells sequentially.


## References

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [COCO Dataset](https://cocodataset.org/)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)

