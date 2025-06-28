## 🧠 WalkNav: YOLOv8-Based Assistive Navigation for the Visually Impaired

> **Empowering independent mobility using real-time path detection and audio guidance.**

---

### 📌 Project Overview

**WalkNav** is a computer vision-based assistive navigation system designed to help visually impaired individuals navigate outdoor environments. It uses a **YOLOv8 segmentation model** to detect walkable paths in real-time and provides **audio feedback** to guide users safely and effectively.

---

### 🛠️ Features

* ✅ Real-time walkable path detection using YOLOv8
* 🎯 Accurate segmentation trained on a custom-labeled dataset
* 🎙️ Audio feedback (e.g., “Turn Left”, “Move Forward”)
* 📷 Live camera input processing
* 🧩 Modular design for future upgrades like GPS navigation

---

### 📂 Project Structure

```plaintext
ASEP2/
├── dataset/
│   ├── images/                # Original images
│   │   ├── train/
│   │   └── val/
│   └── labels/                # YOLO-format labels
│       ├── train/
│       └── val/
├── annotations/               # Original polygonal JSON annotations
├── masks/                     # Masked-out images for segmentation
├── convert_masks_to_yolo.py   # Script to convert JSON to YOLO format
├── yolov8_train.py            # Model training script
├── yolov8_infer.py            # Inference with real-time feedback
└── README.md                  # You are here!
```

---

### 🚀 How It Works

1. **Custom dataset** of 2,800+ annotated outdoor images
2. Images annotated using polygons to define footpaths (stored as JSON)
3. Converted to YOLOv8 segmentation format
4. Model trained using Ultralytics' YOLOv8 on CPU/GPU
5. Inference script uses webcam + TTS to give spoken directions

---

### 🧪 Model Performance

| Metric            | Score     |
| ----------------- | --------- |
| mAP\@0.5 (Mask)   | 0.94      |
| mAP\@0.5:0.95     | 0.79      |
| Precision (Mask)  | 0.94      |
| Recall (Mask)     | 0.87      |
| FPS (Live Camera) | 15–20 FPS |

---

### 🗣️ Audio Feedback

Real-time voice instructions:

* “Move Forward”
* “Turn Left”
* “Obstacle Ahead”

Powered by Python's `pyttsx3` for text-to-speech.

---

### ⚙️ Tech Stack

* 🐍 Python 3.13
* 📦 Ultralytics YOLOv8 (v8.3+)
* 📸 OpenCV
* 🔊 pyttsx3 (for voice feedback)
* 🧠 Custom annotation & training scripts

---

### 🔧 Setup Instructions

```bash
git clone https://github.com/Sujay-Korde/WalkNav-YOLOSeg.git
cd WalkNav-YOLOSeg
pip install -r requirements.txt
```

---

### 🧠 Train the Model

```bash
python yolov8_train.py  # Ensure paths to dataset are configured
```

---

### 📸 Run Live Inference

```bash
python yolov8_infer.py --weights runs/segment/train8/weights/best.pt
```

=======
## 📈 Future Scope

- GPS + Google Maps integration
- Multi-class segmentation (stairs, curbs, potholes)
- Vibration feedback for enhanced accessibility

---

## 👨‍💻 Contributors

- **Sujay Korde** – 
- **Anish Kshirsagar**
- **Kritika Ingle**
- **Kimaya Kolhe**
- **Abhishek Kulkarni**

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
>>>>>>> 64c278e (Add README file with contributors)
