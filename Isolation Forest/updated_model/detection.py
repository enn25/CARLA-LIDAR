import carla
import pygame
import numpy as np
import cv2
import random
import time
import math
import torch
import torch.nn as nn
import torchvision
from collections import deque
from twilio.rest import Client

# Initialize pygame and CARLA client
pygame.init()
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()

# Twilio setup (replace with your Twilio credentials)
TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_PHONE_NUMBER = 'your_twilio_phone_number'
YOUR_PHONE_NUMBER = 'your_phone_number'
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# SpoofClassifier model definition
class SpoofClassifier(nn.Module):
    def __init__(self, input_size):
        super(SpoofClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Function to load the model with flexible input size
def load_model(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        input_size = checkpoint['fc1.weight'].size(1)
        model = SpoofClassifier(input_size)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"Model loaded successfully with input size: {input_size}")
        return model, input_size
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

# Load the trained model
model, input_size = load_model('best_model.pth')
if model is None:
    print("Failed to load the model. Exiting...")
    pygame.quit()
    exit()

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

# Get a list of spawn points and choose one
spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)

# Try spawning the main vehicle with retries
vehicle = try_spawn_vehicle_with_retries(vehicle_bp, spawn_point)

if vehicle is None:
    print("Failed to spawn the main vehicle due to collision. Exiting...")
    pygame.quit()
    exit()

# Set up sensors
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
camera_bp = blueprint_library.find('sensor.camera.rgb')

# LiDAR configuration
lidar_bp.set_attribute('range', '100')
lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

# Camera configuration
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.5))
camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Set up spoofing states
spoof_mode = 0  # 0: no spoof, 1: LiDAR spoof, 2: Camera spoof, 3: Both spoof
detection_mode = False
last_detection_time = 0
detection_interval = 10  # seconds

# Create pygame window (now 3 frames: LiDAR, raw camera, and object detection)
width, height = 800, 600
window = pygame.display.set_mode((width * 3, height))  # Triple window width

# Load YOLOv3 model
yolo_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
yolo_model.eval()

# Function to handle LiDAR data spoofing
def spoof_lidar_data(points):
    if spoof_mode in [1, 3]:
        noise = np.random.normal(0, 10, points.shape)
        points = np.copy(points)  # Create a copy to avoid modifying read-only array
        points += noise
        points = points[points[:, 2] > 1]
    return points

# Function to handle Camera spoofing
def spoof_camera_image(image):
    if spoof_mode in [2, 3]:
        image = cv2.blur(image, (30, 30))
    return image

# LiDAR callback
lidar_data = []
def process_lidar_data(point_cloud):
    global lidar_data
    points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
    lidar_data = spoof_lidar_data(points)

# Camera callback
camera_image = None
camera_image_for_detection = None
def process_camera_image(image):
    global camera_image, camera_image_for_detection
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    array_spoofed = spoof_camera_image(np.copy(array))
    array_with_edges = detect_objects_and_edges(array_spoofed)
    camera_image = array_with_edges
    camera_image_for_detection = detect_objects(array_spoofed)

# Object detection and lane edge detection function
def detect_objects_and_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(image, 1.0, edges_colored, 1.0, 0)
    return combined

# Function to detect objects using YOLOv3
def detect_objects(image):
    image_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        prediction = yolo_model(image_tensor)
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

# Attach the callbacks to sensors
lidar_sensor.listen(lambda data: process_lidar_data(data))
camera_sensor.listen(lambda data: process_camera_image(data))

# Function to extract features for spoofing detection
def extract_features(lidar_points, camera_image):
    lidar_features = np.concatenate([
        np.mean(lidar_points, axis=0),
        np.std(lidar_points, axis=0),
        np.percentile(lidar_points, [25, 50, 75], axis=0).flatten()
    ])
    gray = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
    camera_features = np.concatenate([
        [np.mean(gray)],
        [np.std(gray)],
        np.percentile(gray, [25, 50, 75]),
        [cv2.Laplacian(gray, cv2.CV_64F).var()]
    ])
    return np.concatenate([lidar_features, camera_features])

# Function to detect spoofing
def detect_spoofing(lidar_buffer, camera_buffer):
    combined_features = np.concatenate([lidar_buffer, camera_buffer], axis=1)
    input_tensor = torch.tensor(combined_features.flatten(), dtype=torch.float32).unsqueeze(0)
    
    # Ensure input tensor matches the model's expected input size
    if input_tensor.size(1) != input_size:
        print(f"Warning: Input size mismatch. Expected {input_size}, got {input_tensor.size(1)}.")
        return -1  # Indicate an error
    
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Function to send SMS alert
def send_sms_alert(spoof_type):
    try:
        message = twilio_client.messages.create(
            body=f"ALERT: {spoof_type} spoofing detected!",
            from_=TWILIO_PHONE_NUMBER,
            to=YOUR_PHONE_NUMBER
        )
        print(f"SMS alert sent: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS alert: {str(e)}")

# Pygame event handling for spoof toggling and detection mode
def handle_input():
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
            elif event.key == pygame.K_d:
                detection_mode = not detection_mode
                print(f"Detection mode {'activated' if detection_mode else 'deactivated'}")

# Main loop
try:
    lidar_buffer = deque(maxlen=100)
    camera_buffer = deque(maxlen=100)
    
    while True:
        handle_input()

        # Draw LiDAR data in 2D
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

        # Display Camera feed (raw with edge detection)
        if camera_image is not None:
            camera_surface = pygame.surfarray.make_surface(camera_image.swapaxes(0, 1))

        # Display Camera feed (object detection)
        if camera_image_for_detection is not None:
            detection_surface = pygame.surfarray.make_surface(camera_image_for_detection.swapaxes(0, 1))
        else:
            detection_surface = pygame.Surface((width, height))
            detection_surface.fill((0, 0, 0))

        # Update Pygame window
        window.blit(lidar_surface, (0, 0))
        if camera_image is not None:
            window.blit(camera_surface, (width, 0))
        if camera_image_for_detection is not None:
            window.blit(detection_surface, (width * 2, 0))

        pygame.display.flip()

        # Spoofing detection
        if detection_mode and time.time() - last_detection_time > detection_interval:
            if lidar_data is not None and camera_image is not None:
                features = extract_features(lidar_data, camera_image)
                lidar_buffer.append(features[:15])  # Adjust slice based on your feature extraction
                camera_buffer.append(features[15:])
                
                if len(lidar_buffer) == 100 and len(camera_buffer) == 100:
                    detected_spoof = detect_spoofing(np.array(lidar_buffer), np.array(camera_buffer))
                    if detected_spoof != -1:
                        spoof_types = ["No spoofing", "LiDAR spoofing", "Camera spoofing", "Both LiDAR and Camera spoofing"]
                        print(f"Detected: {spoof_types[detected_spoof]}")
                        
                        if detected_spoof != 0:
                            send_sms_alert(spoof_types[detected_spoof])
                    else:
                        print("Error in spoofing detection. Skipping this detection cycle.")
                    
                    last_detection_time = time.time()

        # Control frame rate
        pygame.time.Clock().tick(30)

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    if 'lidar_sensor' in locals():
        lidar_sensor.destroy()
    if 'camera_sensor' in locals():
        camera_sensor.destroy()
    if 'vehicle' in locals():
        vehicle.destroy()
    pygame.quit()
