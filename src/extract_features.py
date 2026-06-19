import os
import cv2
import numpy as np
from preprocessing import SpatialGridExtractor

def cache_kaggle_features(dataset_path, output_prefix="kaggle"):
    print(f"Starting deep-scan feature extraction on: {dataset_path}")
    extractor = SpatialGridExtractor()
    
    extracted_features = []
    extracted_labels = []
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Recursively crawl every single subfolder in the dataset path
    for root, dirs, files in os.walk(dataset_path):
        # Filter out images in the current folder
        images = [f for f in files if f.lower().endswith(valid_extensions)]
        if not images:
            continue
            
        # Determine the label based on the folder name it lives in
        folder_name = os.path.basename(root).lower()
        if "highly" in folder_name or "good" in folder_name or "engaged" in folder_name and not "not" in folder_name:
            label = 2  # Highly Engaged
        elif "normal" in folder_name or "neutral" in folder_name:
            label = 1  # Normally Engaged
        else:
            label = 0  # Disengaged / Not Engaged fallback
            
        print(f"Found {len(images)} images in folder '{os.path.basename(root)}' -> Processing and mapping to Class {label}...")
        
        # Limit frames per directory path to prevent container memory overflows
        for img_name in images[:300]:
            img_path = os.path.join(root, img_name)
            frame = cv2.imread(img_path)
            
            if frame is None:
                continue
                
            metrics = extractor.extract_grid_features(frame)
            
            if metrics is not None:
                matrix = np.zeros((4, 4), dtype=np.float32)
                matrix[0, :] = metrics["avg_ear"]
                matrix[1, :] = 1.0 if metrics["is_eye_open"] else 0.0
                matrix[2, :2] = metrics["nose_tip"][0] / frame.shape[1]
                matrix[2, 2:] = metrics["nose_tip"][1] / frame.shape[0]
                
                extracted_features.append(matrix)
                extracted_labels.append(label)

    if len(extracted_features) == 0:
        print("\n[ERROR] No faces could be processed. Verify the face/eye cascade classifiers can read the files.")
        return

    X = np.expand_dims(np.array(extracted_features), axis=1) # Shape: [N, 1, 4, 4]
    y = np.array(extracted_labels)
    
    os.makedirs("data/processed", exist_ok=True)
    np.save(f"data/processed/{output_prefix}_X.npy", X)
    np.save(f"data/processed/{output_prefix}_y.npy", y)
    print(f"\n[SUCCESS] Successfully extracted {len(X)} real face grid matrices to data/processed/{output_prefix}_X.npy!")

if __name__ == "__main__":
    sample_download_path = "/home/codespace/.cache/kagglehub/datasets/joyee19/studentengagement/versions/1"
    cache_kaggle_features(sample_download_path, "kaggle")
