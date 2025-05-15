import carla
import pygame
import numpy as np
import cv2
import random
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from twilio.rest import Client
from collections import deque

# Initialize pygame and CARLA client
pygame.init()
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()

# Twilio setup (replace with your actual Twilio credentials)
TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_PHONE_NUMBER = 'your_twilio_phone_number'
YOUR_PHONE_NUMBER = 'your_phone_number'
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Spawn vehicle blueprint and setup
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

# Function to try spawning a vehicle with collision handling and retries
def try_spawn_vehicle_with_retries(bp, spawn_point, max_retries=10):
    vehicle = None
    retries = 0
    while vehicle is None and retries < max_retries:
        vehicle = world.try_spawn_actor(bp, spawn_point)
        if vehicle is None:
            print(f"Failed to spawn vehicle due to collision at {spawn_point.location}, retrying...")
            retries += 1
            spawn_point.location.y += random.uniform(-2.0, 2.0)
            spawn_point.location.x += random.uniform(-2.0, 2.0)
    if vehicle is None:
        print(f"Failed to spawn vehicle after {max_retries} attempts.")
    return vehicle

# Spawn vehicles
spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)
vehicle = try_spawn_vehicle_with_retries(vehicle_bp, spawn_point)
if vehicle is None:
    print("Failed to spawn the main vehicle due to collision. Exiting...")
    pygame.quit()
    exit()

front_car_distance = 20
front_car_spawn_point = carla.Transform(
    spawn_point.location + spawn_point.get_forward_vector() * front_car_distance,
    spawn_point.rotation
)
front_car = try_spawn_vehicle_with_retries(vehicle_bp, front_car_spawn_point)
if front_car is None:
    print("Failed to spawn the front car. Continuing without it.")

# Set up sensors
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
camera_bp = blueprint_library.find('sensor.camera.rgb')

lidar_bp.set_attribute('range', '100')
lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.5))
camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Set up spoofing and detection states
spoof_mode = 0  # 0: no spoof, 1: LiDAR spoof, 2: Camera spoof, 3: Both spoof
detection_mode = False

# Create pygame window
width, height = 800, 600
window = pygame.display.set_mode((width * 3, height))

# Load YOLOv3 model for object detection
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Advanced neural network for spoof classification
class SpoofClassifier(nn.Module):
    def __init__(self, input_size):
        super(SpoofClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize the spoof classifier
input_size = 20  # Adjust based on your feature extraction
spoof_classifier = SpoofClassifier(input_size)
spoof_classifier.eval()

# Data buffers for feature extraction
lidar_buffer = deque(maxlen=100)
camera_buffer = deque(maxlen=100)

# Functions to handle data spoofing
def spoof_lidar_data(points):
    if spoof_mode in [1, 3]:
        noise = np.random.normal(0, 10, points.shape)
        points += noise
        points = points[points[:, 2] > 1]
    return points

def spoof_camera_image(image):
    if spoof_mode in [2, 3]:
        image = cv2.blur(image, (30, 30))
    return image

# Sensor callbacks
lidar_data = []
def process_lidar_data(point_cloud):
    global lidar_data
    points = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3])
    lidar_data = spoof_lidar_data(points)
    if detection_mode:
        lidar_buffer.append(extract_lidar_features(points))

camera_image = None
camera_image_for_detection = None
def process_camera_image(image):
    global camera_image, camera_image_for_detection
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    array_spoofed = spoof_camera_image(array)
    array_with_edges = detect_objects_and_edges(array_spoofed)
    camera_image = array_with_edges
    camera_image_for_detection = detect_objects(array_spoofed)
    if detection_mode:
        camera_buffer.append(extract_camera_features(array_spoofed))

# Feature extraction functions
def extract_lidar_features(points):
    # Extract meaningful features from LiDAR data
    return np.concatenate([
        np.mean(points, axis=0),
        np.std(points, axis=0),
        np.percentile(points, [25, 50, 75], axis=0).flatten()
    ])

def extract_camera_features(image):
    # Extract meaningful features from camera image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.concatenate([
        np.mean(gray),
        np.std(gray),
        np.percentile(gray, [25, 50, 75]),
        cv2.Laplacian(gray, cv2.CV_64F).var()
    ])

# Object detection and edge detection functions
def detect_objects_and_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(image, 1.0, edges_colored, 1.0, 0)
    return combined

def detect_objects(image):
    image_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(image_tensor)
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    debug_image = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_image, f"{label}: {score:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return debug_image

# Spoof detection function
def detect_spoof():
    if len(lidar_buffer) < 100 or len(camera_buffer) < 100:
        return  # Not enough data for detection

    lidar_features = np.array(lidar_buffer)
    camera_features = np.array(camera_buffer)

    # Combine features
    combined_features = np.concatenate([lidar_features, camera_features], axis=1)
    
    # Use the last 20 frames for prediction
    recent_features = combined_features[-20:].flatten()

    with torch.no_grad():
        classifier_input = torch.tensor(recent_features, dtype=torch.float32)
        spoof_type = torch.argmax(spoof_classifier(classifier_input)).item()

    if spoof_type != 0:
        spoof_types = ["No spoof", "LiDAR spoof", "Camera spoof", "Both spoofed"]
        message = f"Alert: {spoof_types[spoof_type]} detected!"
        print(message)
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=YOUR_PHONE_NUMBER
        )

# Attach callbacks to sensors
lidar_sensor.listen(lambda data: process_lidar_data(data))
camera_sensor.listen(lambda data: process_camera_image(data))

# Pygame event handling
def handle_events():
    global spoof_mode, detection_mode
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                spoof_mode = 1
            elif event.key == pygame.K_2:
                spoof_mode = 2
            elif event.key == pygame.K_3:
                spoof_mode = 3
            elif event.key == pygame.K_0:
                spoof_mode = 0
            elif event.key == pygame.K_4:
                detection_mode = not detection_mode
                print(f"Detection mode {'activated' if detection_mode else 'deactivated'}")

# Dataset class for training
class SpoofDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Training function
def train_spoof_classifier(features, labels, model, epochs=100, batch_size=32):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create datasets and dataloaders
    train_dataset = SpoofDataset(X_train, y_train)
    test_dataset = SpoofDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.4f}")

    return model, scaler

# Main simulation loop
try:
    while True:
        handle_events()

        if detection_mode:
            detect_spoof()

        lidar_surface = np.zeros((height, width, 3), dtype=np.uint8)
        if lidar_data is not None:
            for point in lidar_data:
                scale_factor = 4
                x = int(width / 2 + point[1] * scale_factor)
                y = int(height / 2 - point[0] * scale_factor)
                if 0 <= x < width and 0 <= y < height:
                    lidar_surface[y, x] = (255, 255, 255)

        lidar_surface = cv2.cvtColor(lidar_surface, cv2.COLOR_BGR2RGB)
        lidar_surface = pygame.surfarray.make_surface(lidar_surface)

        if camera_image is not None:
            camera_surface = pygame.surfarray.make_surface(camera_image.swapaxes(0, 1))

        if camera_image_for_detection is not None:
            detection_surface = pygame.surfarray.make_surface(camera_image_for_detection.swapaxes(0, 1))
        else:
            detection_surface = pygame.Surface((width, height))
            detection_surface.fill((0, 0, 0))

        window.blit(lidar_surface, (0, 0))
        if camera_image is not None:
            window.blit(camera_surface, (width, 0))
        if camera_image_for_detection is not None:
            window.blit(detection_surface, (width * 2, 0))

        pygame.display.flip()
        pygame.time.Clock().tick(30)

finally:
    lidar_sensor.destroy()
    camera_sensor.destroy()
    vehicle.destroy()
    if front_car is not None:
        front_car.destroy()
    pygame.quit()

# Example usage of the training function (you would run this separately with collected data)
# collected_features, collected_labels = load_your_data()
