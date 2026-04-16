# HAIDS - Human Activity and Incident Detection System

HAIDS is a comprehensive web-based application for real-time monitoring and detection of various activities and incidents using computer vision.

## Features
- **Object Detection**: Detect and label 80 different types of common objects.
- **Social Distancing Detection**: Monitor and visualize social distancing violations with serious and abnormal categorization.
- **Fall Detection**: Detect potential human falls in real-time.
- **Vehicle Crash Detection**: Identify potential vehicle collisions and crashes.
- **YouTube Support**: Process live streams or videos directly from YouTube URLs.

## Requirements
- Python 3.7+
- A webcam or a stable internet connection for YouTube streams.
- (Optional) NVIDIA GPU with CUDA for real-time performance.

## Installation

1. **Clone or Download** the `HAIDS` folder to your computer.
2. **Install Dependencies**:
   Open a terminal/command prompt in the `HAIDS` folder and run:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Open a terminal in the `HAIDS` folder.
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Configuration
You can adjust detection sensitivity, distance thresholds, and email alert settings in `mylib/config.py`.

### GPU Support
If you have an NVIDIA GPU, edit `mylib/config.py` and set:
```python
USE_GPU = True
```
*Note: Requires OpenCV compiled with CUDA support.*
