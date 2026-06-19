import cv2
import numpy as np
import mediapipe as mp

class SpatialGridExtractor:
    def __init__(self):
        # Initialize MediaPipe Face Mesh with refined landmark tracking for iris/pupil
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_grid_features(self, frame):
        """
        Processes a real image frame, extracts facial landmarks, computes EAR,
        and partitions the detected face area into a 3-level, 6-cell spatial matrix.
        """
        if frame is None:
            return None
            
        h, w, _ = frame.shape
        # MediaPipe requires RGB images; OpenCV loads images as BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None  # Return None gracefully if no student face is detected

        landmarks = results.multi_face_landmarks[0].landmark
        
        # 1. Extract bounding box dimensions from absolute coordinate scales
        all_x = [lm.x * w for lm in landmarks]
        all_y = [lm.y * h for lm in landmarks]
        xmin, xmax = int(min(all_x)), int(max(all_x))
        ymin, ymax = int(min(all_y)), int(max(all_y))
        
        face_w = xmax - xmin
        face_h = ymax - ymin
        
        # 2. Divide face vertically into 3 Levels and horizontally into 2 columns (6 cells)
        level_height = face_h // 3
        cell_width = face_w // 2
        
        grid_cells = []
        for row in range(3):
            for col in range(2):
                c_xmin = xmin + (col * cell_width)
                c_xmax = c_xmin + cell_width
                c_ymin = ymin + (row * level_height)
                c_ymax = c_ymin + level_height
                grid_cells.append(((c_xmin, c_ymin), (c_xmax, c_ymax)))
                
        # 3. Geometric Eye Aspect Ratio (EAR) tracking algorithm
        def get_ear(p_upper, p_lower, p_left, p_right):
            v_dist = np.linalg.norm(np.array([landmarks[p_upper].x*w, landmarks[p_upper].y*h]) - 
                                    np.array([landmarks[p_lower].x*w, landmarks[p_lower].y*h]))
            h_dist = np.linalg.norm(np.array([landmarks[p_left].x*w, landmarks[p_left].y*h]) - 
                                    np.array([landmarks[p_right].x*w, landmarks[p_right].y*h]))
            return v_dist / (h_dist + 1e-6)

        # Facial landmark indices matching MediaPipe topology charts
        left_ear = get_ear(159, 145, 33, 133)
        right_ear = get_ear(386, 374, 362, 263)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Standard threshold for blink/closure detection
        is_eye_open = avg_ear > 0.22

        return {
            "bbox": (xmin, ymin, xmax, ymax),
            "grid_cells": grid_cells,
            "is_eye_open": is_eye_open,
            "avg_ear": avg_ear,
            "nose_tip": (int(landmarks[1].x * w), int(landmarks[1].y * h))
        }
