import cv2
import numpy as np

def load_yolo(config_path, weights_path, names_path):
    """
    Load YOLO model with given configuration
    
    Args:
        config_path: Path to YOLO config file
        weights_path: Path to YOLO weights file
        names_path: Path to object names file
        
    Returns:
        net: YOLO network
        output_layers: Output layers of the network
        classes: List of class names
    """
    # Load YOLO network
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    try:
        # OpenCV 4.5.4+
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        # OpenCV <4.5.4
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # Load class names
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, output_layers, classes

def detect_objects(frame, net, output_layers, classes, conf_threshold=0.5, nms_threshold=0.4):
    """
    Detect objects in a frame using YOLO
    
    Args:
        frame: Input frame
        net: YOLO network
        output_layers: Output layers of the network
        classes: List of class names
        conf_threshold: Confidence threshold
        nms_threshold: Non-maximum suppression threshold
        
    Returns:
        boxes: Bounding boxes
        confidences: Confidence scores
        class_ids: Class IDs
        indices: Indices of boxes after NMS
    """
    height, width, _ = frame.shape
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Set input to the network
    net.setInput(blob)
    
    # Forward pass
    outs = net.forward(output_layers)
    
    # Initialize lists
    boxes = []
    confidences = []
    class_ids = []
    
    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            
            # Fix for index out of bounds error
            if len(scores) == 0:
                continue
                
            try:
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > conf_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
            except IndexError:
                # Skip this detection if there's an index error
                continue
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold) if boxes else []
    
    # Convert indices to the expected format if using older OpenCV versions
    if isinstance(indices, np.ndarray) and len(indices.shape) > 1:
        indices = indices.flatten()
    
    return boxes, confidences, class_ids, indices

def draw_detections(frame, boxes, confidences, class_ids, indices, classes):
    """
    Draw bounding boxes and labels on the frame
    
    Args:
        frame: Input frame
        boxes: Bounding boxes
        confidences: Confidence scores
        class_ids: Class IDs
        indices: Indices of boxes after NMS
        classes: List of class names
        
    Returns:
        frame: Frame with detections drawn
    """
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            # Set color based on class
            if label == "helmet":
                color = (0, 255, 0)  # Green for helmet
            elif label == "no-helmet":
                color = (0, 0, 255)  # Red for no-helmet
            else:
                color = (255, 0, 0)  # Blue for other objects
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def detect_helmet_violation(class_ids, indices, classes):
    """
    Detect helmet violations
    
    Args:
        class_ids: Class IDs
        indices: Indices of boxes after NMS
        classes: List of class names
        
    Returns:
        violations: Number of helmet violations
    """
    violations = 0
    
    # Convert indices to list if it's a numpy array
    if isinstance(indices, np.ndarray):
        indices = indices.flatten()
    
    # Get all detected motorcycles
    motorcycle_indices = [i for i in range(len(class_ids)) if i in indices and classes[class_ids[i]] == "motorbike"]
    
    # For this example, we'll count each motorcycle as a potential violation
    # In a real system, you'd need to check if the rider is wearing a helmet
    violations = len(motorcycle_indices)
    
    return violations