import json
import cv2
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import os
import matplotlib.pyplot as plt

# Load YOLO models
model_v5 = YOLO('yolov5n.pt')  # YOLOv5 nano model
model_v8 = YOLO('yolov8n.pt')   # YOLOv8 model
model_v10 = YOLO('yolov10n.pt') # YOLOv10 model

# Load and process image
image_path = 'sample.jpg'
image = cv2.imread(image_path)

# Get predictions from each model
results_v5 = model_v5(image)
results_v8 = model_v8(image)
results_v10 = model_v10(image)

def extract_boxes(results):
    """Extract bounding boxes from YOLO model results."""
    boxes = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            boxes.append([x1, y1, x2 - x1, y2 - y1, conf, cls])  # Convert to (x, y, w, h)
    return boxes

boxes_v5 = extract_boxes(results_v5)
boxes_v8 = extract_boxes(results_v8)
boxes_v10 = extract_boxes(results_v10)

def iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1, y1, w1, h1 = box1[:4]
    x2, y2, w2, h2 = box2[:4]

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0

def ensemble_boxes(pred_v5, pred_v8, pred_v10, iou_threshold=0.5):
    """Ensemble bounding boxes from three models based on IoU threshold."""
    all_boxes = pred_v5 + pred_v8 + pred_v10
    final_boxes = []

    while all_boxes:
        base_box = all_boxes.pop(0)
        matched_boxes = [base_box[:4]]

        for box in all_boxes[:]:
            if iou(base_box, box) > iou_threshold:
                matched_boxes.append(box[:4])
                all_boxes.remove(box)

        # Compute average of matched boxes
        avg_box = np.mean(matched_boxes, axis=0).tolist()
        final_boxes.append(avg_box)

    return final_boxes

# Apply ensemble method
ensemble_boxes_list = ensemble_boxes(boxes_v5, boxes_v8, boxes_v10)

def draw_boxes(image, boxes, color=(0, 255, 0)):
    """Draw bounding boxes on image."""
    for box in boxes:
        x, y, w, h = map(int, box[:4])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image

# Draw ensemble boxes
annotated_image = draw_boxes(image.copy(), ensemble_boxes_list)
cv2.imwrite('output_ensemble.jpg', annotated_image)

# Display the output image
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print("Ensemble detection complete. Output saved as output_ensemble.jpg")
