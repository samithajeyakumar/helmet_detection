import cv2
import numpy as np
import time
import os
from utils.detection_utils import load_yolo, detect_objects, draw_detections, detect_helmet_violation
from utils.tracking_utils import Tracker

# Paths to model files
MODEL_DIR = "models"
CONFIG_PATH = "yolo.cfg"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov3.weights")
NAMES_PATH = "obj.names"

# Path to input video
# Path to input video - update this to match your actual video file
VIDEO_PATH = os.path.join("data", "BB_6d315b1e-956e-43e8-9d62-7d18efed3dd2.mp4")
# Initialize violation counters
total_objects = 0
current_violations = 0

# Initialize tracker
tracker = Tracker(max_age=20, min_hits=5)

def process_frame(frame, net, output_layers, classes):
    """
    Process a single frame for traffic violation detection
    
    Args:
        frame: Input frame
        net: YOLO network
        output_layers: Output layers of the network
        classes: List of class names
        
    Returns:
        processed_frame: Frame with detections and violations marked
        current_violations: Number of current violations
    """
    global total_objects, current_violations
    
    try:
        # Detect objects
        boxes, confidences, class_ids, indices = detect_objects(frame, net, output_layers, classes)
        
        # Draw detections
        for i in range(len(boxes)):
            if i in indices:
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                label = classes[class_id]
                
                # Set color based on class
                if label == "person":
                    color = (0, 255, 0)  # Green for person
                elif label == "motorbike":
                    color = (0, 0, 255)  # Red for motorbike
                else:
                    color = (255, 0, 0)  # Blue for other objects
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                cv2.putText(frame, f"{label}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Count motorcycles as violations for demonstration
        current_violations = len([i for i in range(len(class_ids)) 
                                if i in indices and classes[class_ids[i]] == "motorbike"])
        
        # Display violation count
        cv2.putText(frame, f"Current Violations: {current_violations}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Total objects: {total_objects}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, current_violations
    
    except Exception as e:
        print(f"Error in process_frame: {e}")
        # Return original frame if there's an error
        return frame, 0
def test_yolo_model():
    """Test if the YOLO model loads correctly"""
    try:
        # Load YOLO network
        net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
        
        # Get output layer names
        layer_names = net.getLayerNames()
        try:
            # OpenCV 4.5.4+
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            # OpenCV <4.5.4
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        # Load class names
        with open(NAMES_PATH, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        print("YOLO model loaded successfully!")
        print(f"Number of classes: {len(classes)}")
        print(f"First few classes: {classes[:5]}")
        print(f"Output layers: {output_layers}")
        return True
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return False


def main():
    """
    Main function to run traffic violation detection
    """
    # Create directories if they don't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Check if model file exists
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Model file not found at {WEIGHTS_PATH}")
        print("Please download the YOLOv3 weights file and place it in the models directory.")
        return
    
    # Check if video file exists
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        print("Please provide a video file in the data directory.")
        return
    
    # Load YOLO model
    print("Loading YOLO model...")
    
    try:
        net, output_layers, classes = load_yolo(CONFIG_PATH, WEIGHTS_PATH, NAMES_PATH)
        
        # Print model information for debugging
        print(f"Loaded {len(classes)} classes")
        print(f"Output layers: {output_layers}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    if not test_yolo_model():
        print("Failed to load YOLO model. Please check the weights file.")
        return
    # Open video file
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create output video writer
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video
    print("Processing video...")
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Process frame
            processed_frame, _ = process_frame(frame, net, output_layers, classes)
            
            # Write frame to output video
            out.write(processed_frame)
            
            # Display frame
            cv2.imshow("Traffic Violation Detection", processed_frame)
            
            # Increment frame count
            frame_count += 1
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            # Continue with next frame
            continue
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    fps_processed = frame_count / processing_time if processing_time > 0 else 0
    
    print(f"Video processing complete.")
    print(f"Total frames processed: {frame_count}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Processing FPS: {fps_processed:.2f}")
    print(f"Total objects detected: {total_objects}")
    print(f"Output saved to: {output_path}")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    # Add this after loading the model
    print(f"Model loaded successfully")
    print(f"Classes: {classes[:10]}...")  # Print first 10 classes

if __name__ == "__main__":
    main()