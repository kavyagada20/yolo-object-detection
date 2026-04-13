import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# Load the model
model_path = 'yolo11n.pt'  # or 'yolo11l.pt' for larger model
model = YOLO(model_path)

# Evaluate the model on validation set
print("Evaluating model...")
val_results = model.val(data='data.yaml', split='val')
print(f"mAP50: {val_results.box.map50:.4f}")
print(f"mAP50-95: {val_results.box.map:.4f}")
print(f"Precision: {val_results.box.mp:.4f}")
print(f"Recall: {val_results.box.mr:.4f}")

# Function to visualize predictions
def visualize_predictions(image_path, results):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{model.names[int(cls)]}: {conf:.2f}', color='red', fontsize=10, weight='bold')

    ax.axis('off')
    plt.title(f'Predictions on {Path(image_path).name}')
    plt.show()

# Get some test images
test_images = list(Path('test/images').glob('*.jpg'))[:5]  # First 5 images

# Run inference and visualize
for img_path in test_images:
    print(f"Processing {img_path.name}...")
    results = model.predict(str(img_path), conf=0.25, iou=0.45)
    visualize_predictions(str(img_path), results)

# Plot metrics
metrics = {
    'mAP50': val_results.box.map50,
    'mAP50-95': val_results.box.map,
    'Precision': val_results.box.mp,
    'Recall': val_results.box.mr
}

plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values())
plt.title('Model Evaluation Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
plt.show()