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
shoplifting_model = None
net_dnn = None
classes = None
output_layers = None

netMain = None
metaMain = None
altNames = None

video_link = None
case = None

# Alert System Globals
current_alert = {"id": 0, "module": "", "image": "", "timestamp": 0}
last_alert_time = {
    "fall": 0,
    "social": 0,
    "vehicle": 0,
    "shoplifting": 0
}
alert_cooldown = 10 # 10 seconds cooldown

def trigger_alert(module_label, frame):
    """
    Saves a screenshot and updates the global alert state.
    """
    global current_alert, last_alert_time
    now = time.time()
    
    # Check cooldown for this module
    if now - last_alert_time.get(module_label.lower(), 0) < alert_cooldown:
        return

    # Create timestamped filename
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    filename = f"alert_{module_label.lower()}_{timestamp_str}.jpg"
    filepath = os.path.join('static/assets/images/alerts', filename)
    
    # Save the current frame
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, frame)
    
    # Update state
    last_alert_time[module_label.lower()] = now
    current_alert = {
        "id": int(now * 1000), # Unique ID based on ms
        "module": module_label,
        "image": f"/static/assets/images/alerts/{filename}",
        "timestamp": now
    }
    print(f"[ALERT] {module_label} detected! Screenshot saved to {filepath}")

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
            trigger_alert("Fall", img)
        
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
    Improved Social Distancing Detection Logic
    Ported from Social-Distancing-Detector-main
    """
    results = []
    # Identify persons and calculate centroids
    for (label, conf, bbox) in detections:
        if label.decode() == 'person':
            (cX, cY, w, h) = bbox
            startX = int(cX - (w / 2))
            startY = int(cY - (h / 2))
            endX = int(cX + (w / 2))
            endY = int(cY + (h / 2))
            results.append((conf, (startX, startY, endX, endY), (cX, cY)))

    serious = set()
    abnormal = set()
    
    # Calculate pairwise distances
    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # Draw lines between people in proximity
                if D[i, j] < config.MIN_DISTANCE:
                    serious.add(i)
                    serious.add(j)
                    # Connection line for high risk
                    cv2.line(img, (int(centroids[i][0]), int(centroids[i][1])), 
                            (int(centroids[j][0]), int(centroids[j][1])), config.YELLOW, 1, cv2.LINE_AA)
                    cv2.circle(img, (int(centroids[i][0]), int(centroids[i][1])), 3, config.ORANGE, -1, cv2.LINE_AA)
                    cv2.circle(img, (int(centroids[j][0]), int(centroids[j][1])), 3, config.ORANGE, -1, cv2.LINE_AA)
                elif D[i, j] < config.MAX_DISTANCE:
                    abnormal.add(i)
                    abnormal.add(j)

    stat_H, stat_L = 0, 0
    # Render bounding boxes and status labels
    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        
        if i in serious:
            color = config.RED
            label = "unsafe"
            label_bg = config.WHITE
            label_text = config.ORANGE
            stat_H += 1
            
            # Draw Person Box (Only for UNSAFE members as requested)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            
            # Draw status label above head
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y1label = max(startY, labelSize[1])
            cv2.rectangle(img, (startX, y1label - labelSize[1]), (startX + labelSize[0], startY + baseLine), label_bg, cv2.FILLED)
            cv2.putText(img, label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text, 1, cv2.LINE_AA)
        else:
            stat_L += 1

    # Render Stats Dashboard
    dashboard_w, dashboard_h = 250, 30
    cv2.rectangle(img, (13, 10), (dashboard_w, dashboard_h + 10), config.GREY, cv2.FILLED)
    
    cv2.putText(img, "--", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"LOW RISK: {stat_L} people", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.BLUE, 1, cv2.LINE_AA)

    # Trigger system alerts
    if stat_H > 0:
        cv2.putText(img, "Social Distancing Violation", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.RED, 2)
        trigger_alert("Social Distancing", img)
        if stat_H >= config.Threshold and config.ALERT:
            Mailer().send(config.MAIL)
            
    return img
    

def gen_frames(): 
    global case
    global net_dnn, classes, output_layers, video_link, car_crash_model, shoplifting_model
    
    if case == 'vehicle':
        if car_crash_model is None:
            print('Loading YOLOv8 model for Car Crash Detection...')
            car_crash_model = YOLO(os.path.join(os.path.dirname(__file__), 'models/i1-yolov8s.pt'))
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        totalAccidents = []
    
    if case == 'shoplifting':
        if shoplifting_model is None:
            print('Loading YOLOv8 model for Shoplifting Detection...')
            shoplifting_model = YOLO(os.path.join(os.path.dirname(__file__), 'models/shoplifting_v8.pt'))
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
                    trigger_alert("Vehicle", frame_read)
            
            image = frame_read
        elif case == 'shoplifting':
            results = shoplifting_model.predict(frame_read)
            cc_data = np.array(results[0].boxes.data)

            if len(cc_data) != 0:
                xywh = np.array(results[0].boxes.xywh).astype("int32")
                xyxy = np.array(results[0].boxes.xyxy).astype("int32")
                
                status = ""
                high_conf_shoplifting = False
                # ZIP: (x1, y1, x2, y2), (cx, cy, w, h), (x1, y1, x2, y2, conf, cls)
                for (x1, y1, _, _), (_, _, w, h), (_, _, _, _, conf, clas) in zip(xyxy, xywh, cc_data):
                    if clas == 1: # Shoplifting ALERT
                        # Draw green rectangle (cls1_rect_color from params)
                        cv2.rectangle(frame_read, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                        
                        # Draw red circle at top center (as in source)
                        cx = int(x1 + w // 2)
                        cv2.circle(frame_read, (cx, y1), 6, (0, 0, 255), 8)

                        # Draw confidence
                        text = "{}%".format(np.round(conf * 100, 2))
                        cv2.putText(frame_read, text, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        status = "Shoplifting"
                        if conf > 0.69:
                            high_conf_shoplifting = True
                
                if status:
                    cv2.putText(frame_read, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if high_conf_shoplifting:
                        trigger_alert("Shoplifting", frame_read)
            
            image = frame_read
        else:
            # OpenCV DNN detection - uses a 416x416 blob (independent of display resolution)
            blob = cv2.dnn.blobFromImage(frame_read, 1/255.0, (416, 416), (0,0,0), True, crop=False)
            net_dnn.setInput(blob)
            outs = net_dnn.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            # Use custom thresholds for specific modules
            if case == 'fall':
                conf_thresh = 0.9
            elif case == 'social':
                conf_thresh = 0.7
            else:
                conf_thresh = 0.6
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > conf_thresh:
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
    return render_template('haids.html')
    
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

@app.route('/ShopliftingDetection', methods=['GET', 'POST'])
def ShopliftingDetection():
    global case
    case = 'shoplifting'
    return render_template('ShopliftingDetection.html')

@app.route('/ContactUs')
def ContactUs():
    return render_template('ContactUs.html')
    
@app.route('/Portal')
def Portal():
    return render_template('portal.html')

@app.route('/Video', methods=['GET', 'POST'])
def Video():
    global video_link, current_alert, last_alert_time, case
    # Reset alert state for a new session
    current_alert = {"id": 0, "module": "", "image": "", "timestamp": 0}
    last_alert_time = {k: 0 for k in last_alert_time}
    
    # Set the case (module) dynamically from the form
    selected_module = request.form.get('module')
    if selected_module:
        case = selected_module
        print(f"[INFO] Detection mode set to: {case}")

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

@app.route('/get_latest_alert')
def get_latest_alert():
    global current_alert
    return current_alert

if __name__ == "__main__":
    app.run(debug=True)