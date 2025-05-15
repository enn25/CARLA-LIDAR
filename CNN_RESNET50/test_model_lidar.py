import os
import torch
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define paths
model_path = 'best_lidar_spoofing_model.pth'
spoofed_folder = r'sensor_data\lidar\spoofed'
non_spoofed_folder = r'sensor_data\lidar\non_spoofed'

# Define transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size based on your model's requirements
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the model and set it to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)  # Use ResNet50 if this matches your trained model
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Update output layer to binary classification
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Function to test a LiDAR image
def test_lidar_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB if needed
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        confidence, predicted = torch.max(output, 1)
    return predicted.item(), torch.softmax(output, 1)[0].cpu().numpy()  # Returns predicted label and confidence scores

# Loop through test folders and collect predictions and labels
y_true = []  # Actual labels
y_pred = []  # Predicted labels
confidence_scores = []

# Test spoofed LiDAR images (label = 1)
for img_file in os.listdir(spoofed_folder):
    img_path = os.path.join(spoofed_folder, img_file)
    prediction, confidence = test_lidar_image(img_path)
    y_true.append(1)  # Label for spoofed LiDAR data
    y_pred.append(prediction)
    confidence_scores.append(confidence[1])  # Store spoof confidence score

# Test non-spoofed LiDAR images (label = 0)
for img_file in os.listdir(non_spoofed_folder):
    img_path = os.path.join(non_spoofed_folder, img_file)
    prediction, confidence = test_lidar_image(img_path)
    y_true.append(0)  # Label for non-spoofed LiDAR data
    y_pred.append(prediction)
    confidence_scores.append(confidence[0])  # Store non-spoof confidence score

# Calculate accuracy
accuracy = 100 * np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print(f"Test Accuracy: {accuracy:.2f}%")

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Spoofed", "Spoofed"]).plot(cmap="Blues")
plt.title("Confusion Matrix for LiDAR Spoof Detection")
plt.show()

# Calculate average confidence for spoofed and non-spoofed classes
average_conf_spoofed = np.mean([score for i, score in enumerate(confidence_scores) if y_true[i] == 1])
average_conf_non_spoofed = np.mean([score for i, score in enumerate(confidence_scores) if y_true[i] == 0])
print(f"Average confidence for spoofed images: {average_conf_spoofed:.2f}")
print(f"Average confidence for non-spoofed images: {average_conf_non_spoofed:.2f}")

