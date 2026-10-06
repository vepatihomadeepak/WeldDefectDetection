# AI-Based Weld Seam Defect Detection Using Computer Vision

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end computer vision system for automated weld defect detection using YOLOv8. Designed as an undergraduate engineering project with emphasis on reproducibility, ease of use, and clean documentation.

## 🎯 Project Overview

This project implements a complete pipeline for detecting weld defects in images using deep learning. The system can identify 8 common weld defect types and provides a user-friendly drag-and-drop web interface for inspection.

### Key Features

- **8 Defect Classes**: Crack, Porosity, Undercut, Lack of Fusion, Overlap, Slag Inclusion, Burn Through, Spatter
- **YOLOv8 Based**: State-of-the-art object detection with transfer learning
- **Web Interface**: Simple Streamlit app - drag & drop → analyze → see results
- **Complete Pipeline**: Data preparation → Training → Evaluation → Inference
- **CPU/GPU Support**: Works on both NVIDIA GPU and CPU
- **Evaluation Metrics**: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, Confusion Matrix
- **Batch Processing**: Process multiple images or video files
- **Well Documented**: Comprehensive README with academic presentation support

---

## 🏗️ Architecture

```
weld-defect-detection/
│
├── app.py                      # Main entry point (launches Streamlit)
├── requirements.txt            # Python dependencies
├── data.yaml                   # Dataset configuration
├── README.md                   # This file
│
├── data/                       # Dataset (user provided)
│   ├── train/images/           # Training images
│   ├── train/labels/           # Training labels (YOLO format)
│   ├── val/images/             # Validation images
│   ├── val/labels/             # Validation labels
│   ├── test/images/            # Test images
│   └── test/labels/            # Test labels
│
├── models/                     # Trained models
│   └── best.pt                 # Best trained weights
│
├── results/                    # Evaluation results
│   ├── evaluation_val_metrics.json
│   └── plots/
│       ├── confusion_matrix_val.png
│       └── per_class_metrics_val.png
│
├── runs/                       # Training outputs (auto-generated)
│   └── detect/
│       └── weld_defect/
│           └── weights/
│               ├── best.pt
│               └── last.pt
│
└── src/
    ├── __init__.py
    ├── app.py                  # Streamlit web application
    ├── train.py                # Training script
    ├── evaluate.py             # Evaluation script
    ├── predict.py              # Inference script (CLI)
    ├── preprocess.py           # Preprocessing & augmentation
    └── prepare_dataset.py      # Dataset preparation utilities
```

---

## 📋 Requirements

### Hardware
- **Minimum**: 8 GB RAM, CPU only (slow inference)
- **Recommended**: 16 GB RAM, NVIDIA GPU with 6+ GB VRAM (RTX 3060/4060 or better)
- **Storage**: 5 GB for code + dataset + models

### Software
- Python 3.10+
- NVIDIA CUDA 12.1+ (for GPU acceleration)
- Git

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd weld-defect-detection

# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Prepare Dataset

**Option A: Use your own dataset (YOLO format)**
```
data/
├── train/images/     # .jpg, .png files
├── train/labels/     # .txt files (YOLO format)
├── val/images/
├── val/labels/
├── test/images/
└── test/labels/
```

**Option B: Download from Kaggle (3-class dataset)**
```bash
# Requires: pip install kaggle + API token from kaggle.com
python src/prepare_dataset.py download --dataset sukmaadhiwijaya/welding-defect-object-detection
python src/prepare_dataset.py split --images-dir data/raw/images --labels-dir data/raw/labels --output-dir data
```

**Option C: Create empty structure for manual placement**
```bash
python src/prepare_dataset.py create-structure --data-root data
```

**Generate data.yaml**
```bash
python src/prepare_dataset.py generate-yaml --data-root data
```

### 3. Train Model

```bash
# Quick training (YOLOv8n, 30 epochs)
python src/train.py --model yolov8n.pt --epochs 30 --batch 16

# Better model (YOLOv8s, 50 epochs)
python src/train.py --model yolov8s.pt --epochs 50 --batch 16

# Full training (YOLOv8s, 100 epochs)
python src/train.py --model yolov8s.pt --epochs 100 --batch 16 --optimizer AdamW --lr0 0.001
```

Training outputs saved to `runs/detect/weld_defect/weights/best.pt`

### 4. Evaluate Model

```bash
# Evaluate on validation set
python src/evaluate.py --model runs/detect/weld_defect/weights/best.pt --split val

# Evaluate on test set
python src/evaluate.py --model runs/detect/weld_defect/weights/best.pt --split test --save_plots
```

Results saved to `results/` directory.

### 5. Run Web Application

```bash
# Simple launch
python app.py

# Or directly with streamlit
streamlit run src/app.py
```

Open browser at **https://welddefectdetection.streamlit.app/**

### 6. Command Line Inference

```bash
# Single image
python src/predict.py --model models/best.pt --source test_image.jpg

# Folder of images
python src/predict.py --model models/best.pt --source data/test/images/

# Video
python src/predict.py --model models/best.pt --source weld_video.mp4
```

---

## 📊 Dataset

### Supported Datasets

| Dataset | Source | Classes | Format | License |
|---------|--------|---------|--------|---------|
| **Welding Defect Object Detection** | Kaggle | 3 (Bad Weld, Good Weld, Defect) | YOLO | CC0 |
| **Weld Defect Detection** | GitHub (Abhinand-Nr) | 8 (see below) | YOLO | MIT |
| **Weld Defect Dataset** | GitHub (Linkesh-K-V) | Multiple | Pascal VOC | MIT |

### Our Default 8 Classes (data.yaml)

```yaml
names:
  0: crack
  1: porosity
  2: undercut
  3: lack_of_fusion
  4: overlap
  5: slag_inclusion
  6: burn_through
  7: spatter
nc: 8
```

### YOLO Label Format

Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1] relative to image dimensions.

---

## ⚙️ Configuration

### Training Arguments (train.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `yolov8n.pt` | Model variant (n/s/m/l/x) |
| `--epochs` | 50 | Training epochs |
| `--batch` | 16 | Batch size |
| `--imgsz` | 640 | Image size |
| `--device` | auto | `cuda`, `cpu`, or `0` |
| `--optimizer` | `auto` | `SGD`, `Adam`, `AdamW` |
| `--lr0` | 0.01 | Initial learning rate |
| `--patience` | 20 | Early stopping patience |
| `--amp` | True | Mixed precision |

### Inference Arguments (predict.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--conf` | 0.25 | Confidence threshold |
| `--iou` | 0.6 | NMS IoU threshold |
| `--save_txt` | False | Save YOLO labels |
| `--save_crop` | False | Save cropped detections |

---

## 📈 Evaluation Metrics

The evaluation script computes:

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × P × R / (P + R)
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: COCO-style mAP across IoU thresholds
- **Confusion Matrix**: Per-class prediction accuracy

### Sample Output

```
EVALUATION RESULTS
============================================================
Precision:      0.4102
Recall:         0.3621
F1-Score:       0.3845
mAP@0.5:        0.3312
mAP@0.5:0.95:   0.1401

Per-class metrics:
  crack               P=0.512 R=0.421 F1=0.462 mAP50=0.372 mAP50-95=0.156
  porosity            P=0.389 R=0.345 F1=0.366 mAP50=0.312 mAP50-95=0.134
  undercut            P=0.356 R=0.298 F1=0.325 mAP50=0.292 mAP50-95=0.118
  lack_of_fusion      P=0.401 R=0.312 F1=0.351 mAP50=0.321 mAP50-95=0.142
  overlap             P=0.334 R=0.287 F1=0.309 mAP50=0.294 mAP50-95=0.121
  slag_inclusion      P=0.245 R=0.198 F1=0.219 mAP50=0.170 mAP50-95=0.089
  burn_through        P=0.678 R=0.589 F1=0.630 mAP50=0.611 mAP50-95=0.287
  spatter             P=0.312 R=0.267 F1=0.288 mAP50=0.279 mAP50-95=0.103
============================================================
```

> **Note**: These are example metrics. Actual results depend on your dataset and training.

---

## 🖥️ Web Interface

The Streamlit app provides:

1. **Drag & Drop Upload** - JPG, JPEG, PNG, WEBP
2. **Real-time Analysis** - Click "Analyze Weld"
3. **Visual Results** - Annotated image with bounding boxes
4. **Detailed Detections** - Class, confidence, location for each defect
5. **Download** - Save annotated result

### Interface Preview

```
┌─────────────────────────────────────────────────────────────┐
│  🔍 AI Weld Seam Inspector                                  │
│  Upload a weld image to automatically detect potential     │
│  weld defects                                               │
├─────────────────────┬───────────────────────────────────────┤
│  Original Image     │  Detection Result                     │
│  ┌─────────────┐    │  ┌─────────────┐                      │
│  │             │    │  │  [BOUNDING  │                      │
│  │   WELD IMG  │    │  │   BOXES]    │                      │
│  │             │    │  │             │                      │
│  └─────────────┘    │  └─────────────┘                      │
│                     │                                       │
│  [🔬 Analyze Weld]  │  DEFECT DETECTED                      │
│                     │  Type: Porosity                       │
│                     │  Confidence: 94.2%                    │
│                     │  Count: 3                             │
└─────────────────────┴───────────────────────────────────────┘
```

---

## 🎓 Academic Presentation Support

This section provides structured content for undergraduate project reports.

### Problem Statement

Manual weld inspection is labor-intensive, time-consuming, and prone to human error. In safety-critical industries (shipbuilding, oil & gas, aerospace, construction), undetected weld defects can cause catastrophic failures. There is a need for automated, AI-assisted visual inspection systems that can reliably detect and classify weld defects in real-time.

### Existing System

Current industrial practice relies on:
- Visual inspection by certified weld inspectors (AWS CWI, CSWIP)
- Radiographic testing (RT) and ultrasonic testing (UT) for subsurface defects
- Manual interpretation of radiographic films
- High dependency on inspector experience and fatigue levels

Limitations: Subjective, slow, expensive, inconsistent, cannot scale to high-volume production.

### Proposed System

An AI-based computer vision system using YOLOv8 object detection that:
1. Takes weld images as input (from camera, file upload, or video)
2. Detects and localizes 8 common weld defect types
3. Provides confidence scores for each detection
4. Outputs annotated images with bounding boxes and labels
5. Offers a simple web interface for operators

### Objectives

1. Implement a YOLOv8-based weld defect detection pipeline
2. Achieve reasonable mAP@0.5 on validation data
3. Create an easy-to-use web interface for demonstration
4. Provide complete training/evaluation/inference scripts
5. Document the system for reproducibility

### Methodology

1. **Data Collection**: Public weld defect datasets (Kaggle, GitHub)
2. **Preprocessing**: Resize to 640×640, normalization, augmentation (flip, rotate, brightness, noise, CLAHE)
3. **Model**: YOLOv8 (nano/small/medium) with transfer learning from COCO pre-trained weights
4. **Training**: 50-100 epochs, AdamW optimizer, mosaic/mixup augmentation, early stopping
5. **Evaluation**: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, confusion matrix
6. **Deployment**: Streamlit web app with drag-and-drop interface

### Algorithm

**YOLOv8 (You Only Look Once v8)** - Single-stage object detector:
- Backbone: CSPDarknet with C2f modules
- Neck: PAN-FPN (Path Aggregation Network - Feature Pyramid Network)
- Head: Decoupled head (separate classification and regression branches)
- Loss: CIOU loss + DFL (Distribution Focal Loss) + BCE loss
- Anchor-free design with task-aligned learning

### Expected Output

- Annotated images with bounding boxes around detected defects
- Defect class labels (Crack, Porosity, Undercut, etc.)
- Confidence percentages for each detection
- Summary statistics (count per defect type)
- Downloadable result images

### Advantages

- ✅ Fast inference (~50-150 FPS on GPU)
- ✅ Single model for detection + classification
- ✅ Transfer learning reduces data requirements
- ✅ Simple deployment (single .pt file)
- ✅ Web interface requires no coding skills
- ✅ Open-source, no licensing costs
- ✅ Extensible to new defect classes

### Limitations

- ❌ Requires labeled training data for each defect type
- ❌ 2D images only - cannot detect subsurface defects
- ❌ Performance depends on image quality, lighting, angle
- ❌ Not a replacement for certified NDT methods (RT, UT)
- ❌ Small defects may be missed at 640×640 resolution
- ❌ False positives possible on noisy backgrounds

### Future Scope

- Instance segmentation (YOLOv8-seg / SABE-YOLO) for precise defect boundaries
- Real-time video stream processing for production lines
- Integration with robotic inspection systems
- Defect severity scoring based on size/location
- Multi-scale training for better small defect detection
- ONNX/TensorRT export for edge deployment (Jetson, Raspberry Pi)
- REST API for factory MES/ERP integration

---

## 🔧 Troubleshooting

### Model Not Found
```
Error: No trained model found.
```
**Solution**: Train a model first or place `best.pt` in `models/`

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `--batch` size, use `--imgsz 416`, or use CPU (`--device cpu`)

### No Dataset Found
```
Error: No training images found
```
**Solution**: Check `data/` folder structure matches YOLO format

### Import Errors
```
ModuleNotFoundError: No module named 'ultralytics'
```
**Solution**: `pip install -r requirements.txt`

### Streamlit Not Opening
```
Connection refused
```
**Solution**: Check port 8501 is free, try `streamlit run src/app.py --server.port 8502`

---

## 📚 References & Credits

This project builds upon the following open-source work:

1. **Ultralytics YOLOv8** - https://github.com/ultralytics/ultralytics
   - Primary detection framework
   - License: AGPL-3.0

2. **Weld-Defect-Detection** by Abhinand-Nr - https://github.com/Abhinand-Nr/Weld-Defect-Detection
   - Dataset structure, 8-class definition, certification logic
   - License: MIT

3. **SABE-YOLO** by Re-y - https://github.com/Re-y/sabe-yolo
   - Weld seam segmentation approach
   - License: Check repository

4. **Welding-Defect-Object-Detection** by Shilpa-Golla - https://github.com/Shilpa-Golla/Welding-Defect-Object-Detection
   - Kaggle dataset usage, notebook workflow
   - License: Check repository

5. **Weld-Defect-Dataset** by Linkesh-K-V - https://github.com/Linkesh-K-V/Weld-Defect-Dataset
   - Pascal VOC format dataset
   - License: MIT

6. **Kaggle Welding Defect Dataset** - https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection
   - 3-class dataset (Bad Weld, Good Weld, Defect)
   - License: CC0

### Standards Referenced

- **AWS D1.1** - Structural Welding Code (Steel)
- **ISO 5817** - Quality levels for imperfections in fusion-welded joints
- **ISO 17635** - Non-destructive testing of welds

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Important**: The YOLOv8 framework (ultralytics) is licensed under AGPL-3.0. If you modify and distribute the YOLOv8 code, you must comply with AGPL-3.0. Using the pre-trained weights and API as-is is generally acceptable for internal/commercial use per Ultralytics' policy.

---

## 👨‍💻 Author

Developed as an undergraduate engineering project.

**Acknowledgments**: Thanks to the open-source computer vision community, especially the Ultralytics team for YOLOv8, and all dataset contributors.

---

## 🙋 Support

For issues and questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing GitHub issues
3. Create a new issue with details (OS, Python version, error message, steps to reproduce)

---

*Last updated: September 2026*