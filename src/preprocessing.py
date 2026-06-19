import cv2
import numpy as np

class SpatialGridExtractor:
    def __init__(self):
        # Load OpenCV built-in cascading classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def extract_grid_features(self, frame):
        """
        Detects face region using Haar Cascades, slices it into the research grid layout,
        and computes baseline eye validation structures.
        """
        if frame is None:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None  # No face detected
            
        # Extract primary face dimensions
        (x, y, w, h) = faces[0]
        xmin, xmax = x, x + w
        ymin, ymax = y, y + h
        
        # Slicing rules: 3 Vertical Levels, 2 Horizontal Columns
        level_height = h // 3
        cell_width = w // 2
        
        grid_cells = []
        for row in range(3):
            for col in range(2):
                c_xmin = xmin + (col * cell_width)
                c_xmax = c_xmin + cell_width
                c_ymin = ymin + (row * level_height)
                c_ymax = c_ymin + level_height
                grid_cells.append(((c_xmin, c_ymin), (c_xmax, c_ymax)))
                
        # Detect eyes within the face bounding box
        face_roi = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
        
        # Calculate mock EAR profile based on real eye area sizes or presence
        is_eye_open = len(eyes) >= 2
        avg_ear = 0.28 if is_eye_open else 0.15
        
        # Use center point of the face box as approximate nose tracking coordinate
        nose_tip = (int(x + w/2), int(y + h/2))

        return {
            "bbox": (xmin, ymin, xmax, ymax),
            "grid_cells": grid_cells,
            "is_eye_open": is_eye_open,
            "avg_ear": avg_ear,
            "nose_tip": nose_tip
        }
