import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. Define a Structured Dataset using the 3-Level / 6-Cell Feature Baseline
class EngagementDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.features = torch.zeros(num_samples, 1, 4, 4)
        self.labels = torch.randint(0, 3, (num_samples,))
        
        for i in range(num_samples):
            label = self.labels[i].item()
            
            if label == 0:    # Disengaged: Lower eye features (EAR), high head tilt offsets
                self.features[i, 0, 0:2, :] = torch.randn(2, 4) * 0.2 - 0.5
                self.features[i, 0, 2:4, :] = torch.randn(2, 4) * 0.2 + 0.8
            elif label == 1:  # Normally Engaged: Standard baseline values
                self.features[i, 0, :, :] = torch.randn(4, 4) * 0.4
            elif label == 2:  # Highly Engaged: High focal tracking, centered positioning
                self.features[i, 0, 0:2, :] = torch.randn(2, 4) * 0.1 + 0.9
                self.features[i, 0, 2:4, :] = torch.randn(2, 4) * 0.1 - 0.1
                
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
        
# 2. Main Training Framework Execution
def run_training_pipeline():
    print("Initializing Training Pipeline inside GitHub Codespace...")
    
    # Setup data loaders for our training and validation cycles
    train_dataset = EngagementDataset(num_samples=1200)
    val_dataset = EngagementDataset(num_samples=300)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Import the CNN model structure from your model file
    from model import StudentEngagementCNN
    model = StudentEngagementCNN(num_classes=3)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    print(f"Beginning optimization loops across {epochs} epochs targeting a 93.5% threshold...")
    
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
            
        # Validation Assessment Phase
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
        val_acc = (correct / total) * 100
        print(f"Epoch {epoch+1:02d}/{epochs} -> Training Loss: {epoch_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")
        
        # Guardrail checkpoint checking for model completion criteria
        if val_acc >= 93.5:
            print(f"Success! Target validation threshold achieved at {val_acc:.2f}% accuracy. Saving model checkpoints.")
            torch.save(model.state_dict(), "data/processed/engagement_cnn_weights.pt")
            break

if __name__ == "__main__":
    run_training_pipeline()