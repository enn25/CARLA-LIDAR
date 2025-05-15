import carla
import numpy as np
import cv2
import random
import time
import torch
import torch.nn as nn
from collections import deque

# Define the model architecture
class SpoofingDetectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SpoofingDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize CARLA client
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Load the spoofing detection model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 3000  # Adjust this based on your feature size (30 features * 100 frames)
hidden_size = 128  # Adjust as needed
num_classes = 4  # 0: No spoof, 1: LiDAR, 2: Camera, 3: Both

model = SpoofingDetectionModel(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

def spawn_vehicle(world, spawn_point):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    return vehicle

def setup_sensors(world, vehicle):
    blueprint_library = world.get_blueprint_library()
    
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.5))
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    
    return lidar_sensor, camera_sensor

def extract_lidar_features(points):
    return np.concatenate([
        np.mean(points, axis=0),
        np.std(points, axis=0),
        np.percentile(points, [25, 50, 75], axis=0).flatten()
    ])

def extract_camera_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.concatenate([
        [np.mean(gray)],
        [np.std(gray)],
        np.percentile(gray, [25, 50, 75]),
        [cv2.Laplacian(gray, cv2.CV_64F).var()]
    ])

def spoof_lidar_data(points, intensity):
    points = np.copy(points)
    noise = np.random.normal(0, intensity, points.shape)
    points += noise
    return points[points[:, 2] > 0.5]

def spoof_camera_image(image, intensity):
    return cv2.blur(image, (int(intensity), int(intensity)))

def run_scenario(world, model, device, spoof_mode=0, spoof_intensity=10, scenario_duration=200):
    spawn_points = world.get_map().get_spawn_points()
    vehicle = spawn_vehicle(world, random.choice(spawn_points))
    if vehicle is None:
        print("Failed to spawn vehicle. Skipping scenario.")
        return None

    lidar_sensor, camera_sensor = setup_sensors(world, vehicle)

    lidar_buffer = deque(maxlen=100)
    camera_buffer = deque(maxlen=100)

    lidar_data = None
    camera_data = None

    def lidar_callback(data):
        nonlocal lidar_data
        lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]

    def camera_callback(data):
        nonlocal camera_data
        camera_data = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4))[:, :, :3]

    lidar_sensor.listen(lidar_callback)
    camera_sensor.listen(camera_callback)

    try:
        for _ in range(scenario_duration):
            world.tick()
            if lidar_data is None or camera_data is None:
                continue

            if spoof_mode in [1, 3]:
                lidar_data = spoof_lidar_data(lidar_data, spoof_intensity)
            if spoof_mode in [2, 3]:
                camera_data = spoof_camera_image(camera_data, spoof_intensity)

            lidar_features = extract_lidar_features(lidar_data)
            camera_features = extract_camera_features(camera_data)

            lidar_buffer.append(lidar_features)
            camera_buffer.append(camera_features)

            if len(lidar_buffer) == 100 and len(camera_buffer) == 100:
                combined_features = np.concatenate([np.array(lidar_buffer).flatten(), np.array(camera_buffer).flatten()])
                input_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    prediction = model(input_tensor)
                
                predicted_spoof_mode = torch.argmax(prediction, dim=1).item()
                
                if predicted_spoof_mode != 0:
                    print(f"ALERT: Spoofing detected! Predicted mode: {predicted_spoof_mode}")
                    return predicted_spoof_mode
    
    finally:
        lidar_sensor.stop()
        camera_sensor.stop()
        lidar_sensor.destroy()
        camera_sensor.destroy()
        vehicle.destroy()
    
    return 0  # No spoofing detected

def main():
    num_scenarios = 10
    wait_time = 30  # seconds

    for scenario in range(num_scenarios):
        print(f"Running scenario {scenario + 1}/{num_scenarios}")
        
        spoof_mode = random.choice([0, 1, 2, 3])
        spoof_intensity = random.uniform(5, 20)
        
        result = run_scenario(world, model, device, spoof_mode, spoof_intensity)
        
        if result is None:
            print("Scenario failed. Moving to next scenario.")
            continue
        
        if result == 0:
            print("No spoofing detected. Waiting before re-testing...")
            time.sleep(wait_time)
            
            # Re-run the scenario after waiting
            result = run_scenario(world, model, device, spoof_mode, spoof_intensity)
            
            if result != 0:
                print(f"ALERT: Spoofing detected on re-test! Predicted mode: {result}")
            else:
                print("No spoofing detected on re-test.")
        
        print(f"Scenario {scenario + 1} completed. Actual spoof mode: {spoof_mode}")
        print("--------------------")

    print("All scenarios completed.")

if __name__ == '__main__':
    main()
