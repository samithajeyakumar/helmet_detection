from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np  # Added missing import
import os
import time
import threading
from utils.detection_utils import load_yolo, detect_objects, detect_helmet_violation
from utils.tracking_utils import Tracker

app = Flask(__name__)

# Paths to model files
MODEL_DIR = "models"
CONFIG_PATH = "yolo.cfg"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov3.weights")
NAMES_PATH = "obj.names"

# Path to input video
VIDEO_PATH = os.path.join("data", "BB_6d315b1e-956e-43e8-9d62-7d18efed3dd2.mp4")

# Global variables
frame_lock = threading.Lock()
current_frame = None
violation_stats = {
    "helmet_violations": 0,
    "total_objects": 0,
    "vehicles_detected": 0
}
stats_lock = threading.Lock()

# Initialize YOLO model
print("Loading YOLO model...")
try:
    net, output_layers, classes = load_yolo(CONFIG_PATH, WEIGHTS_PATH, NAMES_PATH)
    print(f"Model loaded successfully with {len(classes)} classes")
except Exception as e:
    print(f"Error loading model: {e}")
    # Set default values in case model loading fails
    net, output_layers, classes = None, [], []

# Initialize tracker
tracker = Tracker(max_age=20, min_hits=5)

def process_video():
    """
    Process video in a separate thread
    """
    global current_frame, violation_stats
    
    # Make sure model is loaded
    if net is None:
        print("Error: YOLO model not loaded. Cannot process video.")
        return
    
    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return
    
    print(f"Processing video: {VIDEO_PATH}")
    
    # Reset violation stats
    with stats_lock:
        violation_stats = {
            "helmet_violations": 0,
            "total_objects": 0,
            "vehicles_detected": 0
        }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop back to the beginning of the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        try:
            # Detect objects
            boxes, confidences, class_ids, indices = detect_objects(frame, net, output_layers, classes)
            
            # Prepare detections for tracker
            detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    detections.append([*boxes[i], class_ids[i], confidences[i]])
            
            # Update tracker
            active_tracks = tracker.update(detections)
            
            # Draw active tracks
            for track_id, track in active_tracks.items():
                x, y, w, h = track['box']
                class_id = track['class_id']
                
                # Make sure class_id is within range
                if class_id < len(classes):
                    label = classes[class_id]
                else:
                    label = "unknown"
                
                # Set color based on class
                if label == "helmet":
                    color = (0, 255, 0)  # Green for helmet
                elif label == "no-helmet":
                    color = (0, 0, 255)  # Red for no-helmet
                elif label == "motorbike":
                    color = (255, 165, 0)  # Orange for motorbike
                else:
                    color = (255, 0, 0)  # Blue for other objects
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label and track ID
                cv2.putText(frame, f"{label} ID:{track_id}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # For standard YOLOv3, count motorcycles as potential helmet violations
            # In a real system, you'd need more sophisticated logic to detect actual helmet violations
            current_violations = len([t for t in active_tracks.values() 
                                    if class_id < len(classes) and classes[t['class_id']] == "motorbike"])
            
            # Count vehicles
            vehicles_detected = len([t for t in active_tracks.values() 
                                    if class_id < len(classes) and classes[t['class_id']] in 
                                    ["car", "motorbike", "bus", "truck"]])
            
            # Update violation statistics
            with stats_lock:
                violation_stats["helmet_violations"] = current_violations
                # Only increment total violations when we detect new ones
                if current_violations > 0:
                    violation_stats["total_objects"] += 1
                violation_stats["vehicles_detected"] = vehicles_detected
            
            # Display violation count
            cv2.putText(frame, f"Current Violations: {current_violations}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Total objects: {violation_stats['total_objects']}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Vehicles: {vehicles_detected}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Update current frame
            with frame_lock:
                current_frame = frame.copy()
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # If there's an error, still update the current frame
            with frame_lock:
                current_frame = frame.copy()
        
        # Sleep to control processing rate
        time.sleep(0.03)  # ~30 FPS

def generate_frames():
    """
    Generate frames for video streaming
    """
    while True:
        with frame_lock:
            if current_frame is not None:
                try:
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', current_frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Yield frame in multipart response
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error generating frame: {e}")
            else:
                # If no frame is available, yield a blank frame
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', blank_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Sleep to control streaming rate
        time.sleep(0.03)  # ~30 FPS

@app.route('/')
def index():
    """
    Render dashboard index page
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """
    Get current violation statistics
    """
    with stats_lock:
        return jsonify(violation_stats)

if __name__ == '__main__':
    # Start video processing in a separate thread
    video_thread = threading.Thread(target=process_video)
    video_thread.daemon = True
    video_thread.start()
    
    # Start Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)  # Changed debug to False for production