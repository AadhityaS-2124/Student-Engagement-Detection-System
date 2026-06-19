import os
import cv2
import torch
import numpy as np
from model import StudentEngagementCNN
from preprocessing import SpatialGridExtractor

def run_live_inference():
    print("Initializing Real-Time Webcam Inference Engine...")
    
    # 1. Initialize the video capture stream (0 is usually default laptop webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open or access your webcam hardware stream.")
        return
        
    extractor = SpatialGridExtractor()
    
    # 2. Instantiate and load trained custom CNN weights
    model = StudentEngagementCNN(num_classes=3)
    weights_path = "data/processed/engagement_cnn_weights.pt"
    
    if os.path.exists(weights_path):
        print("🔥 Loading customized production weights from training run...")
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    else:
        print("⚠️ Production weights not found. Running with baseline initializations...")
        
    model.eval()
    
    class_labels = {0: "Disengaged", 1: "Normally Engaged", 2: "Highly Engaged"}
    print("\n🎥 Stream Started! Press 'q' on your video window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from device stream.")
            break
            
        # Flip frame horizontally for natural mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # 3. Pass live frame to your OpenCV Spatial Cascade Extractor
        metrics = extractor.extract_grid_features(frame)
        
        if metrics is not None:
            # Draw the face bounding box on screen
            xmin, ymin, xmax, ymax = metrics["bbox"]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
            
            # Reconstruct the 4x4 matrix representation matching your CNN input layers
            matrix = np.zeros((4, 4), dtype=np.float32)
            matrix[0, :] = metrics["avg_ear"]
            matrix[1, :] = 1.0 if metrics["is_eye_open"] else 0.0
            matrix[2, :2] = metrics["nose_tip"][0] / w
            matrix[2, 2:] = metrics["nose_tip"][1] / h
            
            # Format matrix tensor to expected dimensions [Batch=1, Channel=1, H=4, W=4]
            tensor_input = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Run feedforward evaluation inference
            with torch.no_grad():
                outputs = model(tensor_input)
                _, predicted = torch.max(outputs, 1)
                prediction = predicted.item()
            
            # 4. Handle classes and triggers
            status_text = f"Status: {class_labels[prediction]}"
            
            if prediction == 0:
                # Flash a bold red alert if Disengaged status is flagged
                cv2.putText(frame, "WARNING: PAY ATTENTION!", (30, 60), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 3)
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
                
            cv2.putText(frame, status_text, (30, ymin - 15 if ymin > 40 else 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            # If no face is caught in frame, prompt visibility alert
            cv2.putText(frame, "STATUS: No Face Detected", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
        # Display rendering layout window
        cv2.imshow("Student Engagement Detection System Monitor", frame)
        
        # Clear loop execution if 'q' key is intercepted
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam inference loop terminated cleanly.")

if __name__ == "__main__":
    run_live_inference()
