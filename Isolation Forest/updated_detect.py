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
from sklearn.ensemble import IsolationForest
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

# Load YOLOv3 model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Initialize Isolation Forest models
lidar_if = IsolationForest(contamination=0.1, random_state=42)
camera_if = IsolationForest(contamination=0.1, random_state=42)

# Improved neural network for classification
class SpoofClassifier(nn.Module):
    def __init__(self):
        super(SpoofClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 4)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

spoof_classifier = SpoofClassifier()

# Load the pre-trained model (replace 'model.pth' with the actual path)
# Uncomment and modify when loading your trained model
# spoof_classifier.load_state_dict(torch.load('model.pth'))
spoof_classifier.eval()

# Data buffers for anomaly detection
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
        lidar_buffer.append(np.mean(points, axis=0))

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
        camera_buffer.append(np.mean(array_spoofed, axis=(0, 1)))

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

# Improved spoof detection function with the new model
def detect_spoof():
    global lidar_if, camera_if
    if len(lidar_buffer) < 100 or len(camera_buffer) < 100:
        return  # Not enough data for detection

    lidar_data = np.array(lidar_buffer)
    camera_data = np.array(camera_buffer)

    # Fit Isolation Forest models if not fitted yet
    if not hasattr(lidar_if, 'offset_'):
        lidar_if.fit(lidar_data)
    if not hasattr(camera_if, 'offset_'):
        camera_if.fit(camera_data)

    lidar_pred = lidar_if.predict(lidar_data)
    camera_pred = camera_if.predict(camera_data)

    lidar_anomaly = np.mean(lidar_pred == -1)
    camera_anomaly = np.mean(camera_pred == -1)

    # Prepare input for the neural network classifier
    classifier_input = torch.tensor([lidar_anomaly, camera_anomaly], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        spoof_probabilities = torch.softmax(spoof_classifier(classifier_input), dim=1)
        spoof_type = torch.argmax(spoof_probabilities).item()
    
    spoof_types = ["No spoof", "LiDAR spoof", "Camera spoof", "Both spoofed"]
    confidence = spoof_probabilities[0, spoof_type].item()

    if spoof_type != 0:
        message = f"Alert: {spoof_types[spoof_type]} detected with {confidence:.2f} confidence!"
        print(message)
        # Uncomment the following lines when you're ready to send SMS alerts
        # twilio_client.messages.create(
        #     body=message,
        #     from_=TWILIO_PHONE_NUMBER,
        #     to=YOUR_PHONE_NUMBER
        # )

# Attach callbacks to sensors
lidar_sensor.listen(lambda data: process_lidar_data(data))
camera_sensor.listen(lambda data: process_camera_image(data))

# Pygame event handling
def handle_events():
    global spoof_mode, detection_mode
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                spoof_mode = 0
            elif event.key == pygame.K_2:
                spoof_mode = 1
            elif event.key == pygame.K_3:
                spoof_mode = 2
            elif event.key == pygame.K_4:
                spoof_mode = 3
            elif event.key == pygame.K_d:
                detection_mode = not detection_mode

# Main loop
while True:
    handle_events()
    window.fill((0, 0, 0))
    if camera_image is not None:
        camera_surface = pygame.surfarray.make_surface(np.rot90(camera_image))
        window.blit(camera_surface, (0, 0))
    pygame.display.flip()

    if detection_mode:
        detect_spoof()

    pygame.time.wait(10)

