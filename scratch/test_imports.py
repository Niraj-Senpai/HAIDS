import os
import sys

try:
    from ultralytics import YOLO
    print("ultralytics imported successfully")
except ImportError:
    print("ultralytics NOT found")

try:
    import cvzone
    print("cvzone imported successfully")
except ImportError:
    print("cvzone NOT found")

try:
    from modules.sort import Sort
    print("Sort module imported successfully")
except Exception as e:
    print(f"Sort module error: {e}")

try:
    model_path = os.path.join(os.getcwd(), 'models/i1-yolov8s.pt')
    if os.path.exists(model_path):
        print(f"Model found at {model_path}")
        model = YOLO(model_path)
        print("Model loaded successfully")
    else:
        print(f"Model NOT found at {model_path}")
except Exception as e:
    print(f"Model loading error: {e}")
