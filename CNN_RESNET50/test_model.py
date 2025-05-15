import os
import torch
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix
import numpy as np

# Define paths
model_path = 'best_camera_spoofing_model.pth'  # Ensure this matches the saved model's name
spoofed_folder = '/home/nithin/carla/PythonAPI/examples/cnn/sensor_data/camera/spoofed'
non_spoofed_folder = '/home/nithin/carla/PythonAPI/examples/cnn/sensor_data/camera/non_spoofed'

# Define transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the model and set it to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)  # Change to ResNet50
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Adjust output layer to 2 classes
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Function to test an image
def test_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()  # 1 for spoofed, 0 for non-spoofed

# Loop through test folders and collect predictions and labels
y_true = []  # Actual labels
y_pred = []  # Predicted labels

# Test spoofed images (label = 1)
for img_file in os.listdir(spoofed_folder):
    img_path = os.path.join(spoofed_folder, img_file)
    prediction = test_image(img_path)
    y_true.append(1)  # Label for spoofed images
    y_pred.append(prediction)

# Test non-spoofed images (label = 0)
for img_file in os.listdir(non_spoofed_folder):
    img_path = os.path.join(non_spoofed_folder, img_file)
    prediction = test_image(img_path)
    y_true.append(0)  # Label for non-spoofed images
    y_pred.append(prediction)

# Calculate accuracy
accuracy = 100 * np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print(f"Test Accuracy: {accuracy:.2f}%")

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

