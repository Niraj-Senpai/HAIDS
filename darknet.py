import cv2
import numpy as np
import os

# Mocking the darknet C module using OpenCV DNN
# This allows scripts written for the original darknet.py to work without the darknet binaries.

class DarknetNet:
    def __init__(self, config, weights):
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

class DarknetMeta:
    def __init__(self, names_list):
        self.classes = names_list

class DarknetImage:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.frame = None

def load_net_custom(config_path, weights_path, batch, ignore):
    return DarknetNet(config_path.decode(), weights_path.decode())

def load_meta(meta_path):
    names_path = "cfg/coco.names"
    if os.path.exists(meta_path.decode()):
        with open(meta_path.decode(), 'r') as f:
            for line in f:
                if 'names' in line:
                    names_path = line.split('=')[1].strip()
    
    # Try local cfg if not found
    if not os.path.exists(names_path) and os.path.exists("cfg/coco.names"):
        names_path = "cfg/coco.names"
        
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return DarknetMeta(classes)

def make_image(w, h, c):
    return DarknetImage(w, h)

def copy_image_from_bytes(darknet_image, frame_bytes):
    nparr = np.frombuffer(frame_bytes, np.uint8)
    darknet_image.frame = nparr.reshape((darknet_image.h, darknet_image.w, 3))

def detect_image(net, meta, darknet_image, thresh=0.25):
    img = darknet_image.frame
    if img is None:
        return []
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0,0,0), True, crop=False)
    net.net.setInput(blob)
    outs = net.net.forward(net.output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > thresh:
                center_x = detection[0] * darknet_image.w
                center_y = detection[1] * darknet_image.h
                w = detection[2] * darknet_image.w
                h = detection[3] * darknet_image.h
                boxes.append([int(center_x), int(center_y), int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, thresh, 0.4)
    
    results = []
    # Handle different OpenCV versions for NMSBoxes return type
    if len(indexes) > 0:
        for i in indexes.flatten():
            label = meta.classes[class_ids[i]]
            results.append((label.encode(), confidences[i], (float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3]))))
    return results
