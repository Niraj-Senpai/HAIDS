# 🌌 HAIDS — Human Activity and Incident Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=ai&logoColor=white)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

**HAIDS** is an AI-powered surveillance portal designed for real-time monitoring and incident response. Built with a premium **"Deep Space" Dark Mode** aesthetic and utilizing advanced computer vision models, HAIDS transforms standard video feeds into intelligent, actionable security data.

---

## ✨ Key Features

- 🛡️ **Intelligent Surveillance**: Four specialized detection modules working in tandem or independently.
- ⚡ **Real-Time Priority Alerts**: Intelligent alert system that prioritizes critical incidents (e.g., Car Crashes) over minor violations.
- 📊 **Centralized Dashboard**: A glassmorphism-inspired incident portal to review, filter, and manage captured events.
- 🎥 **Universal Feed Support**: Seamlessly process local video files, webcam streams, or live YouTube broadcasts.
- 🎨 **Premium UI/UX**: Modern "Deep Space" theme featuring glassmorphism, dynamic status badges, and smooth transitions.
- 📧 **Automated Notifications**: Integrated SMTP mailer for instant incident reporting.

---

## 🚀 Detection Modules

### 🚨 Car Crash Detection
- **Model**: YOLOv8s  + SORT Tracker
- **Function**: Detects vehicle collisions and tracks unique accident events to prevent redundant alerts.

### 🧤 Shoplifting Detection
- **Model**: Custom YOLOv8s 
- **Function**: Identifies suspicious shoplifting behavior with high-confidence thresholds (0.7+) for accuracy.

### 🏃 Fall Detection
- **Model**: YOLOv8n via OpenCV DNN.
- **Function**: Monitors human posture and triggers alerts when a fall-like orientation is detected.

### 📏 Social Distancing
- **Model**: YOLOv8n via OpenCV DNN.
- **Function**: Calculates Euclidean distance between individuals to flag unsafe proximity (Serious vs. Abnormal risk).

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python, Flask, Subprocess |
| **AI / Computer Vision** | YOLOv8 (Ultralytics), YOLOv4-tiny, OpenCV DNN, SORT (Simple Online and Realtime Tracking) |
| **Frontend** | HTML5, CSS3 (Vanilla), JavaScript, Jinja2 Templates |
| **Video Processing** |  `cvzone`, `NumPy` |
| **Design** | Glassmorphism, Deep Space Dark Theme, Google Fonts (Inter/Outfit) |

---

## 📥 Installation & Setup

### Prerequisites
- **Python 3.8+**
- **NVIDIA GPU** (Optional, but recommended for CUDA acceleration)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Niraj-Senpai/HAIDS.git
cd HAIDS
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Model Weights
Ensure the following weights are in the `models/` and root directory:
- `models/i1-yolov8s.pt` (Car Crash)
- `models/shoplifting_v8.pt` (Shoplifting)
- `yolov4-tiny.weights` (General Detection)

---

## 🎮 Usage

1. **Launch the Portal**:
   ```bash
   python app.py
   ```
2. **Access the Interface**:
   Open your browser and navigate to `http://127.0.0.1:5000`.

3. **Navigate**:
   - Use the **Portal** to select specific detection modules.
   - Use the **Combined Detection** to run multiple monitors simultaneously.
   - Use the **Incident Dashboard** to review past captures.

---

## 🧠 System Intelligence

### Alert Priority Logic
In **Combined Mode**, HAIDS employs a hierarchical priority system to ensure the most critical incident is handled first:
1. 🏎️ **Vehicle Crash** (Highest Priority)
2. 🛍️ **Shoplifting**
3. 📉 **Fall Detection**
4. 👥 **Social Distancing** (Lowest Priority)

### Alert Cooldown
To prevent notification fatigue, the system enforces a **10-second cooldown** between alerts for each specific module.

---

## 🌌 UI Design Philosophy
The HAIDS interface is built on the **Deep Space** design system:
- **Glassmorphism**: Translucent cards with subtle blur effects.
- **Vibrant Accents**: Neon blue, purple, and red for status indicators.
- **Responsive Layout**: Seamlessly transitions between desktop and monitoring stations.

---

## 📄 License & Credits
Developed as part of the **CS719 - Data Science Project** at the University.

**Author**: Niraj

*Note: This project is intended for educational and research purposes.*

