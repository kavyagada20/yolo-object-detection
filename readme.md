# YOLO Object Detection Project

## 📌 Overview

This project performs object detection using YOLO (Ultralytics) on a custom dataset.

The model is trained to detect underwater plastic pollution objects such as bottles, masks, nets, etc.

---

## 📁 Project Structure

```
project/
│── data/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
│── weights/
│   └── best.pt
│
│── train.py
│── detect.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Setup

### 1. Create virtual environment

```bash
python -m venv yolo_env
```

### 2. Activate environment

```bash
# Windows
yolo_env\Scripts\activate

# Linux/Mac
source yolo_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

```bash
yolo detect train \
    data=data.yaml \
    model=yolo11l.pt \
    epochs=100 \
    imgsz=640 \
    batch=16
```

---

## 🔍 Inference (Detection)

```bash
yolo detect predict \
    model=weights/best.pt \
    source=path/to/image_or_video
```

---

## 🧠 Notes

* Make sure your `data.yaml` is correctly configured.
* GPU is recommended for training.
* You can use pretrained models like:

  * yolo11n.pt (fast)
  * yolo11s.pt (balanced)
  * yolo11l.pt (accurate)

---

## 📊 Dataset

Dataset contains multiple classes of underwater waste such as:

* Mask
* Bottle
* Plastic
* Net
* etc.

---

## 🛠️ Dependencies

* Python 3.10+
* PyTorch
* Ultralytics YOLO

---

## ✅ Output

Training results will be saved in:

```
runs/detect/train/
```

---

## 📌 Reference

YOLO by Ultralytics:
https://docs.ultralytics.com/
