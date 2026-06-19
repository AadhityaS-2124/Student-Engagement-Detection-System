import torch
import numpy as np
from model import StudentEngagementCNN

def run_live_inference():
    print("Loading saved model weights for inference...")
    
    # Initialize the CNN structure and load the saved weights
    model = StudentEngagementCNN(num_classes=3)
    model.load_state_dict(torch.load("data/processed/engagement_cnn_weights.pt"))
    model.eval()
    
    # Class map for human-readable outputs
    engagement_classes = {0: "Disengaged ❌", 1: "Normally Engaged 🟡", 2: "Highly Engaged 🟢"}
    
    print("\n--- Simulating Live Student Tracker Session ---")
    
    # Simulating a stream of 5 frame updates
    for frame_id in range(1, 6):
        # Pass a 1x1x4x4 tensor to match the exact mathematical shape used in train.py
        eval_input = torch.randn(1, 1, 4, 4)
            
        # Run the forward pass through the trained network
        with torch.no_grad():
            output = model(eval_input)
            _, prediction = torch.max(output, 1)
            predicted_class = prediction.item()
            
        print(f"[Frame {frame_id:02d}] Status: {engagement_classes[predicted_class]}")

if __name__ == "__main__":
    run_live_inference()
