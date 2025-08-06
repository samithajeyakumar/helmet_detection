import cv2
import numpy as np

class Tracker:
    def __init__(self, max_age=10, min_hits=3):
        """
        Initialize tracker
        
        Args:
            max_age: Maximum number of frames to keep a track alive without matching
            min_hits: Minimum number of hits to start tracking
        """
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.min_hits = min_hits
    
    def update(self, detections):
        """
        Update tracks with new detections
        
        Args:
            detections: List of detections [x, y, w, h, class_id, confidence]
            
        Returns:
            active_tracks: Dictionary of active tracks
        """
        # If no tracks yet, initialize with detections
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = {
                    'box': det[:4],
                    'class_id': det[4],
                    'confidence': det[5],
                    'age': 0,
                    'hits': 1,
                    'active': False
                }
                self.next_id += 1
            return self.tracks
        
        # Match detections to existing tracks
        matched_indices = self._match_detections(detections)
        
        # Update matched tracks
        for i, j in matched_indices:
            track_id = list(self.tracks.keys())[i]
            self.tracks[track_id]['box'] = detections[j][:4]
            self.tracks[track_id]['class_id'] = detections[j][4]
            self.tracks[track_id]['confidence'] = detections[j][5]
            self.tracks[track_id]['age'] = 0
            self.tracks[track_id]['hits'] += 1
            
            # Activate track if it has enough hits
            if self.tracks[track_id]['hits'] >= self.min_hits:
                self.tracks[track_id]['active'] = True
        
        # Add unmatched detections as new tracks
        matched_det_indices = [j for _, j in matched_indices]
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                self.tracks[self.next_id] = {
                    'box': det[:4],
                    'class_id': det[4],
                    'confidence': det[5],
                    'age': 0,
                    'hits': 1,
                    'active': False
                }
                self.next_id += 1
        
        # Update age of unmatched tracks and remove old ones
        matched_track_indices = [i for i, _ in matched_indices]
        track_ids = list(self.tracks.keys())
        for i, track_id in enumerate(track_ids):
            if i not in matched_track_indices:
                self.tracks[track_id]['age'] += 1
                
                # Remove track if it's too old
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
        
        # Return active tracks
        return {k: v for k, v in self.tracks.items() if v['active']}
    
    def _match_detections(self, detections):
        """
        Match detections to existing tracks using IoU
        
        Args:
            detections: List of detections [x, y, w, h, class_id, confidence]
            
        Returns:
            matches: List of matched indices (track_idx, detection_idx)
        """
        if not self.tracks or not detections:
            return []
        
        # Calculate IoU between each track and detection
        track_boxes = [track['box'] for track in self.tracks.values()]
        detection_boxes = [det[:4] for det in detections]
        
        iou_matrix = np.zeros((len(track_boxes), len(detection_boxes)))
        for i, track_box in enumerate(track_boxes):
            for j, det_box in enumerate(detection_boxes):
                iou_matrix[i, j] = self._calculate_iou(track_box, det_box)
        
        # Match tracks to detections using greedy algorithm
        matches = []
        while np.max(iou_matrix) > 0.3:  # IoU threshold
            # Find max IoU
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            matches.append((i, j))
            
            # Remove matched track and detection from consideration
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
        
        return matches
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two boxes
        
        Args:
            box1: First box [x, y, w, h]
            box2: Second box [x, y, w, h]
            
        Returns:
            iou: IoU value
        """
        # Convert to [x1, y1, x2, y2] format
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate intersection area
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area
        
        return iou