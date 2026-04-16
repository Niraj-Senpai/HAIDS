from flask import Flask, render_template, Response, request
import math
import random
import os
import cv2
import numpy as np
import time
from itertools import combinations
import subprocess
from scipy.spatial import distance as dist
from mylib import config
from mylib.mailer import Mailer
from ultralytics import YOLO
import cvzone
from modules.sort import Sort

car_crash_model = None
net_dnn = None
classes = None
output_layers = None

netMain = None
metaMain = None
altNames = None

video_link = None
case = None

def is_close(p1, p2):
    """
    #================================================================
    # Purpose : Calculate Euclidean Distance between two points
    #================================================================    
    :param:
    p1, p2 = two points for calculating Euclidean Distance

    :return:
    dst = Euclidean Distance between two 2d points
    """
    dst = math.sqrt(p1**2 + p2**2)
    #=================================================================#
    return dst 

def convertBack(x, y, w, h): 
    #================================================================
    # Purpose : Converts center coordinates to rectangle coordinates
    #================================================================  
    """
    :param:
    x, y = midpoint of bbox
    w, h = width, height of the bbox
    
    :return:
    xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes_fall(detections, img):
    """
    :param:
    detections = total detections in one frame
    img = image from detect_image method of darknet

    :return:
    img with bbox
    """
    
    #================================================================
    # Purpose : Filter out Persons class from detections
    #================================================================
    if len(detections) > 0:  						# At least 1 detection in the image and check detection presence in a frame  
        centroid_dict = dict() 						# Function creates a dictionary and calls it centroid_dict
        objectId = 0								# We inialize a variable called ObjectId and set it to 0
        for detection in detections:				# In this if statement, we filter all the detections for persons only
            # Check for the only person name tag 
            name_tag = str(detection[0].decode())   # Coco file has string of all the names
            if name_tag == 'person':                
                x, y, w, h = detection[2][0],\
                            detection[2][1],\
                            detection[2][2],\
                            detection[2][3]      	# Store the center points of the detections
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))   # Convert from center coordinates to rectangular coordinates, We use floats to ensure the precision of the BBox            
                # Append center point of bbox for persons detected.
                centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) # Create dictionary of tuple with 'objectId' as the index center points and bbox
    #=================================================================
    
    #=================================================================
    # Purpose : Determine whether the fall is detected or not 
    #=================================================================            	
        fall_alert_list = [] # List containing which Object id is in under threshold distance condition. 
        red_line_list = []
        for id,p in centroid_dict.items():
            dx, dy = p[4] - p[2], p[5] - p[3]  	# Check the difference
            difference = dy-dx			
            if difference < 0:						
                fall_alert_list.append(id)       #  Add Id to a list
        
        for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
            if idx in fall_alert_list:   # if id is in red zone list
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2) # Create Red bounding boxes
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # Create Green bounding boxes
		#=================================================================#

		#=================================================================
    	# Purpose : Displaying the results
    	#================================================================= 
        if len(fall_alert_list)!=0:
            text = "Fall Detected"
        
        else:
            text = "Fall Not Detected"
            
        location = (10, 30)
        if len(fall_alert_list) != 0:
            cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)  # Display Text
        else:
            cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)  # Display Text

        #=================================================================#
    return img

def cvDrawBoxes_social(detections, img):
    """
    :param:
    detections = total detections in one frame
    img = image to draw on
    """
    results = []
    for (label, conf, bbox) in detections:
        if label.decode() == 'person':
            # bbox is [center_x, center_y, w, h]
            (cX, cY, w, h) = bbox
            startX = int(cX - (w / 2))
            startY = int(cY - (h / 2))
            endX = int(cX + (w / 2))
            endY = int(cY + (h / 2))
            results.append((conf, (startX, startY, endX, endY), (cX, cY)))

    serious = set()
    abnormal = set()

    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < config.MIN_DISTANCE:
                    serious.add(i)
                    serious.add(j)
                elif D[i, j] < config.MAX_DISTANCE:
                    abnormal.add(i)
                    abnormal.add(j)

    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        if i in serious:
            color = (0, 0, 255) # Red (BGR)
        elif i in abnormal:
            color = (0, 255, 255) # Yellow (BGR)

        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
        cv2.circle(img, (cX, cY), 5, color, 2)

    # Violation counts
    text = "Serious: {}".format(len(serious))
    cv2.putText(img, text, (10, img.shape[0] - 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    text1 = "Abnormal: {}".format(len(abnormal))
    cv2.putText(img, text1, (10, img.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if len(serious) >= config.Threshold:
        cv2.putText(img, "-ALERT: Violations over limit-", (10, img.shape[0] - 65),
            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
        if config.ALERT:
            # Note: Mailer requires setup in mylib/config.py
            print("[INFO] Sending mail...")
            Mailer().send(config.MAIL)
            print("[INFO] Mail sent")
            
    return img
    

def gen_frames(): 
    global case
    global net_dnn, classes, output_layers, video_link, car_crash_model
    
    if case == 'vehicle':
        if car_crash_model is None:
            print('Loading YOLOv8 model for Car Crash Detection...')
            car_crash_model = YOLO(os.path.join(os.path.dirname(__file__), 'models/i1-yolov8s.pt'))
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        totalAccidents = []
    configPath = "./cfg/yolov4-tiny.cfg"                                 # Path to cfg
    weightPath = "./yolov4-tiny.weights"                                 # Path to weights
    namesPath = "./cfg/coco.names"                                       # Path to names
    
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" + os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath)+"`")
    if not os.path.exists(namesPath):
        # Create coco.names if it doesn't exist (fallback)
        if not os.path.exists("./cfg"): os.makedirs("./cfg")
        with open(namesPath, "w") as f:
            # We can't easily write all 80 names here, but we can assume it was downloaded
            pass

    if case != 'vehicle' and net_dnn is None:
        print(f"Loading OpenCV DNN network for mode: {case}...")
        try:
            net_dnn = cv2.dnn.readNetFromDarknet(configPath, weightPath)
            if config.USE_GPU:
                print("[INFO] Looking for GPU")
                net_dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net_dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                net_dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net_dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            layer_names = net_dnn.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net_dnn.getUnconnectedOutLayers()]
            with open(namesPath, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error loading OpenCV DNN: {e}")
            # If dnn fails, we can still continue if case was not strictly requiring it, 
            # but usually it is required if case != 'vehicle'
    
    print(f"Loading video from: {video_link}")
    
    url = video_link
    if url is not None and os.path.exists(url):
        print(f"Loading local video file: {url}")
        cap = cv2.VideoCapture(url)
    else:
        try:
            # Use yt-dlp to get the actual stream URL reliably
            print("Fetching stream URL using yt-dlp...")
            result = subprocess.run(
                ['yt-dlp', '-g', '-f', 'b', url],
                capture_output=True, text=True, check=True
            )
            stream_url = result.stdout.strip()
            print(f"Stream URL obtained: {stream_url[:50]}...")
            
            cap = cv2.VideoCapture(stream_url)
        except Exception as e:
            print(f"Error getting stream URL: {e}")
            return

    if not cap.isOpened():
        print("Failed to open video capture.")
        return
        
    frame_width = int(cap.get(3))     # Returns the width and height of capture video   
    frame_height = int(cap.get(4))
    
    if frame_width == 0 or frame_height == 0:
        print("Invalid video dimensions.")
        cap.release()
        return

    new_height, new_width = frame_height // 2, frame_width // 2
    print(f"Video Resolution: {frame_width}x{frame_height} -> Resized to: {new_width}x{new_height}")
    
    print("Starting the YOLO loop...")

    while True:
        ret, frame_read = cap.read()
        if not ret:
            break

        # High resolution dimensions for drawing
        h_orig, w_orig = frame_read.shape[:2]

        if case == 'vehicle':
            results = car_crash_model(frame_read, stream=True)
            detections_v8 = np.empty((0, 5))
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    if conf > 0.4:
                        cvzone.cornerRect(frame_read, (x1, y1, w, h))
                        cvzone.putTextRect(frame_read, f'Accident {conf}', (max(0, x1), max(35, y1)), colorR=(0, 165, 255))
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections_v8 = np.vstack((detections_v8, currentArray))

            trackerResults = tracker.update(detections_v8)

            for result in trackerResults:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                id = int(id)
                w, h = x2 - x1, y2 - y1

                if totalAccidents.count(id) == 0:
                    cvzone.cornerRect(frame_read, (x1, y1, w, h), colorR=(255, 0, 255))
                    cvzone.putTextRect(frame_read, f'{id}', (max(0, x1), max(35, y1)))
                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(frame_read, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    totalAccidents.append(id)
            
            image = frame_read
        else:
            # OpenCV DNN detection - uses a 416x416 blob (independent of display resolution)
            blob = cv2.dnn.blobFromImage(frame_read, 1/255.0, (416, 416), (0,0,0), True, crop=False)
            net_dnn.setInput(blob)
            outs = net_dnn.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.25:
                        # Scale coordinates back to high-resolution frame
                        center_x = int(detection[0] * w_orig)
                        center_y = int(detection[1] * h_orig)
                        w = int(detection[2] * w_orig)
                        h = int(detection[3] * h_orig)
                        boxes.append([center_x, center_y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.4)
            
            detections = []
            for i in range(len(boxes)):
                if i in indexes:
                    label = classes[class_ids[i]]
                    detections.append((label.encode(), confidences[i], boxes[i]))

        if case != 'vehicle':
            # Draw directly on the high-resolution frame (frame_read is BGR)
            if case == 'social':
                image = cvDrawBoxes_social(detections, frame_read)
            elif case == 'fall':
                image = cvDrawBoxes_fall(detections, frame_read)
            else:
                image = frame_read
                    
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
                
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

#Deploying the app using flask
#Initialize the Flask app
app = Flask(__name__)
from werkzeug.utils import secure_filename
import os

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
@app.route('/')
def Home():
    return render_template('Shady.html')
    
@app.route('/FallDetection', methods=['GET', 'POST'])
def FallDetection():
    global case
    case = 'fall'
    return render_template('FallDetection.html')
    

    
@app.route('/SocialDistancingDetection', methods=['GET', 'POST'])
def SocialDistancingDetection():
    global case
    case = 'social'
    return render_template('SocialDistancingDetection.html')
    
    
@app.route('/VehicleCrashDetection', methods=['GET', 'POST'])
def VehicleCrashDetection():
    global case
    case = 'vehicle'
    return render_template('VehicleCrashDetection.html')

@app.route('/ContactUs')
def ContactUs():
    return render_template('ContactUs.html')
    
@app.route('/Video', methods=['GET', 'POST'])
def Video():
    global video_link
    if 'video_file' in request.files and request.files['video_file'].filename != '':
        file = request.files['video_file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        video_link = file_path
    else:
        video_link = request.form.get('videolink')
    return render_template('Video.html')
	
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)