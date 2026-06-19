import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentEngagementCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(StudentEngagementCNN, self).__init__()
        
        # Convolutional Layer 1 [cite: 151]
        # Expecting an input feature map representing the 3x6 spatial cell matrix configurations
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Max Pooling 1 [cite: 154]
        
        # Convolutional Layer 2 [cite: 152]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Max Pooling 2 [cite: 155]
        
        # Fully Connected / Dense Layer [cite: 153]
        # Adjust input dimensions based on the flattened size of your final feature grid
        self.fc1 = nn.Linear(32 * 1 * 1, 64) 
        
        # Output Layer mapping to engagement states (e.g., Disengaged, Normally Engaged, Highly Engaged) [cite: 107, 156]
        self.fc2 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Pass through Conv 1 -> ReLU activation -> Pool 1
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Pass through Conv 2 -> ReLU activation -> Pool 2
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten the feature maps for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Dense layer with dropout protection to prevent overfitting
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output logits
        x = self.fc2(x)
        return x