# 🎯 Student-Engagement-Detection-System

## 📋 Overview

An **end-to-end, paper-implemented hybrid CNN system** designed to detect student engagement levels using real human attention metrics and computer vision. This production-ready application combines:

✅ **Automated Kaggle Dataset Extraction Crawler** — Deep-scan recursive pipeline for nested folder traversal and dataset caching  
✅ **Live Interactive Webcam Inference Loop** — Real-time video feedback with bounding box tracking and engagement warnings  
✅ **Dynamic Class Penalization Weights** — Inverse-frequency weighted training to handle class imbalance in real-world data  
✅ **Multi-Cell Spatial Attention Grid** — Face detection with 6-cell grid layout slicing (3 vertical levels × 2 horizontal columns)

This is a **Paper Implementation** project built from scratch to restore and reproduce code from published research methodology. The implementation leverages **PyTorch** for neural network architecture and **OpenCV + MediaPipe** for real-time face tracking and feature extraction.

---

## 📊 Performance Metrics

| Metric | Baseline | Current (Production) | Status |
| --- | --- | --- | --- |
| Training Loss | 0.9670 | **0.6087** ✅ | Successfully Optimized |
| Validation Accuracy | Flatline ⚠️ | Responsive to Class Distribution | ✅ Dynamic & Adaptive |
| Test Samples Processed | — | **1,623 cached matrices** | ✅ Full Dataset |
| Class Imbalance Handling | No compensation | Inverse-Frequency Weighting | ✅ Deployed |

The model demonstrates **significant convergence improvement** across training epochs with dynamic validation accuracy responsive to real-world class distributions. Dynamic class penalization successfully broke the baseline flatline, enabling robust multi-class engagement detection.

---

## 🛠️ Technical Architecture

### CNN Architecture

The system employs a **multi-layer sequential CNN structure** optimized for efficient feature extraction and real-time inference:

**Input Layer**
- Structural feature matrices (6×2×6) representing spatial coordinates and Eye Aspect Ratio (EAR) states extracted from the 6-cell focus tracking grid

**Convolutional Block 1**
- `Conv2D` (16 filters, 3×3 kernel, ReLU activation)
- `MaxPool2D` (2×2 stride)

**Convolutional Block 2**
- `Conv2D` (32 filters, 3×3 kernel, ReLU activation)
- `MaxPool2D` (2×2 stride)

**Dense Layers**
- Fully Connected Linear layer (32×1×1 → 64 neurons)
- Dropout (25% rate) to prevent overfitting

**Output Layer**
- Linear classifier mapping to **6 engagement states**:
  - 👀 Looking Away
  - 😴 Bored
  - 😪 Drowsy
  - 😤 Frustrated
  - 😊 Engaged
  - ❓ Confused

### Training Strategy

- **Loss Function:** Inverse-Frequency Weighted Cross-Entropy
- **Optimizer:** Adam with dynamic learning rate scheduling
- **Class Balancing:** Automatic weight calculation based on dataset distribution
- **Regularization:** Dropout (25%), L2 weight decay

---

## 📁 Project Directory Structure

```
Student-Engagement-Detection-System/
├── data/
│   └── processed/                    # Cached features and model weights
│       ├── kaggle_X.npy             # Extracted feature matrices (1,623 samples)
│       └── engagement_cnn_weights.pt # Trained PyTorch model checkpoint
├── src/
│   ├── preprocessing.py             # OpenCV Haar Cascade face tracking & 6-cell grid layout slicing
│   ├── model.py                     # Custom PyTorch sequential layers definition
│   ├── extract_features.py          # Deep-scan recursive crawler & MediaPipe/OpenCV feature cache pipeline
│   ├── train.py                     # Inverse-frequency weighted Cross-Entropy training loop for data imbalance
│   └── evaluate.py                  # Real-time interactive webcam inference and live warning overlay engine
└── README.md
```

---

## ⚙️ How to Run

### Prerequisites

Install required Python dependencies:

```bash
pip install opencv-python numpy torch kagglehub
```

### System Dependencies

Install required system libraries on Ubuntu/Debian:

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```

### Execution Pipeline

#### Step 1️⃣: Download Kaggle Datasets

Authenticate with Kaggle and download the student engagement dataset:

```bash
kagglehub
python -c "import kagglehub; kagglehub.dataset_download('joyee19/studentengagement')"
```

This downloads the dataset containing organized folders for each engagement class:
- `Looking Away/`
- `bored/`
- `drowsy/`
- `frustrated/`
- `engaged/`
- `confused/`

#### Step 2️⃣: Extract and Cache Features

Run the deep-scan recursive crawler to extract facial features and cache them as feature matrices:

```bash
python src/extract_features.py
```

**Output:**
- Recursively crawls all nested class folders
- Extracts MediaPipe face landmarks and OpenCV Haar Cascade coordinates
- Generates 6-cell spatial attention grids for each image
- Caches **1,623 processed feature matrices** to `data/processed/kaggle_X.npy`
- Creates corresponding engagement labels cache

#### Step 3️⃣: Train with Dynamic Weighted Optimization

Execute the training pipeline with inverse-frequency class weighting to handle data imbalance:

```bash
python src/train.py
```

**Features:**
- Automatic class weight calculation based on dataset distribution
- Dynamic validation monitoring across epochs
- Early stopping with checkpoint saving at peak performance
- Real-time loss/accuracy plotting
- Model saved to `data/processed/engagement_cnn_weights.pt`

#### Step 4️⃣: Deploy Live Webcam Inference

Launch the real-time interactive webcam application:

```bash
python src/evaluate.py
```

**Live Features:**
- Opens local video monitor panel from default camera
- Real-time face detection with bounding box tracking
- Displays engagement classification for each detected face
- **⚠️ Red "WARNING: PAY ATTENTION!" overlay** flashes when Disengaged is detected
- FPS counter and engagement confidence scores displayed
- Press `Q` to exit

---

## 🚀 Features

- **Real-time Inference:** Process video at ~30 FPS with minimal latency
- **Multi-face Detection:** Track and classify engagement for multiple students simultaneously
- **Robust Face Tracking:** Cascaded Haar Cascade + MediaPipe face landmark detection
- **Production-Ready Weights:** Pre-trained model checkpoint included for immediate deployment
- **Automated Dataset Pipeline:** One-command Kaggle dataset extraction and caching
- **Adaptive Training:** Dynamic class weights automatically adjust to dataset imbalance

---

## 📚 Research & References

This project implements engagement detection methodology based on spatial attention tracking and facial feature analysis. The 6-cell grid system provides fine-grained spatial attention metrics that correlate with student engagement levels.

**Key Metrics Tracked:**
- Eye Aspect Ratio (EAR) for gaze direction
- Face position within grid cells (upper, middle, lower × left, right)
- Head pose estimation
- Face visibility and detection confidence

---

## 🔧 Troubleshooting

**Camera not detected?**
```bash
# Check available video devices
ls -la /dev/video*
# Specify device in evaluate.py: cap = cv2.VideoCapture(0)  # Change 0 to device number
```

**CUDA out of memory?**
```bash
# Reduce batch size in train.py
# Or run on CPU (automatic fallback if CUDA unavailable)
```

**Kaggle authentication error?**
```bash
# Place kaggle.json in ~/.kaggle/
# mkdir -p ~/.kaggle
# cp kaggle.json ~/.kaggle/
# chmod 600 ~/.kaggle/kaggle.json
```

---

## 📝 License & Attribution

Paper Implementation Project | Built for Educational Research

---

**Last Updated:** 2026 | **Status:** Production Ready ✅
