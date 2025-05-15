import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Custom Dataset for loading LiDAR images and labels
class LidarImageDataset(Dataset):
    def __init__(self, spoofed_images_path, non_spoofed_images_path, transform=None):
        self.images_path = spoofed_images_path + non_spoofed_images_path
        self.transform = transform
        self.data = []

        # Load spoofed images
        for image_file in os.listdir(spoofed_images_path):
            image_path = os.path.join(spoofed_images_path, image_file)
            self.data.append((image_path, 1))  # 1 for spoofed

        # Load non-spoofed images
        for image_file in os.listdir(non_spoofed_images_path):
            image_path = os.path.join(non_spoofed_images_path, image_file)
            self.data.append((image_path, 0))  # 0 for normal

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations with data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset and DataLoader
spoofed_images_path = 'sensor_data/lidar/spoofed'
non_spoofed_images_path = 'sensor_data/lidar/non_spoofed'
dataset = LidarImageDataset(spoofed_images_path=spoofed_images_path, non_spoofed_images_path=non_spoofed_images_path, transform=transform)

# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define a deeper CNN model (ResNet50 for better feature extraction)
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 output classes: normal and spoofed

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Path to save the best model
best_model_path = 'best_lidar_spoofing_model.pth'

# Training function with model saving
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save the model if the validation accuracy is the best seen so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

# Load the best model (for testing)
best_model = models.resnet50(pretrained=True)
best_model.fc = nn.Linear(best_model.fc.in_features, 2)
best_model.load_state_dict(torch.load(best_model_path))
best_model = best_model.to(device)

