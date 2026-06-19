import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model import StudentEngagementCNN

def run_training_pipeline():
    print("Initializing Weighted Production Training Pipeline...")
    
    X_path = "data/processed/kaggle_X.npy"
    y_path = "data/processed/kaggle_y.npy"
    
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        print("[ERROR] Extracted Kaggle features missing. Please run extract_features.py first.")
        return

    print("🔥 Loading authentic behavioral dataset...")
    X_data = torch.tensor(np.load(X_path), dtype=torch.float32)
    y_data = torch.tensor(np.load(y_path), dtype=torch.long)
    
    # Calculate exact class distributions dynamically to solve imbalance
    labels_np = y_data.numpy()
    class_counts = np.bincount(labels_np, minlength=3)
    print(f"Dataset Distribution -> Class 0 (Disengaged): {class_counts[0]} | Class 1: {class_counts[1]} | Class 2 (Engaged): {class_counts[2]}")
    
    # Compute inverse frequency weights: weight = total_samples / (num_classes * class_samples)
    total_samples = len(labels_np)
    weights = []
    for count in class_counts:
        if count > 0:
            weights.append(total_samples / (3.0 * count))
        else:
            weights.append(1.0) # Avoid division by zero for unrepresented Class 1
            
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Calculated Penalization Weights: {class_weights.tolist()}")

    # 80/20 train/validation split
    dataset = TensorDataset(X_data, y_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = StudentEngagementCNN(num_classes=3)
    
    # Inject penalization weights into the Cross Entropy loss engine
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    print(f"Beginning adaptive optimization loops targeting over 93.5% validation accuracy...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_loss = running_loss / len(train_loader.dataset)
        val_acc = (correct / total) * 100 if total > 0 else 0
        print(f"Epoch {epoch+1:02d}/{epochs} -> Loss: {epoch_loss:.4f} | Dynamic Val Accuracy: {val_acc:.2f}%")
        
        if val_acc >= 93.5:
            print(f"\n[SUCCESS] Target threshold reached at {val_acc:.2f}%! Saving real production weights.")
            os.makedirs("data/processed", exist_ok=True)
            torch.save(model.state_dict(), "data/processed/engagement_cnn_weights.pt")
            break

if __name__ == "__main__":
    run_training_pipeline()
