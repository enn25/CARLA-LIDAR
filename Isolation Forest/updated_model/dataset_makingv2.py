import carla
import random
import numpy as np
import cv2
import time
from collections import deque

def setup_carla_world():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    return world

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
    # Create a copy of the points array to avoid modification of read-only buffer
    points = np.copy(points)
    noise = np.random.normal(0, intensity, points.shape)
    points += noise
    return points[points[:, 2] > 0.5]  # Remove points below ground level

def spoof_camera_image(image, intensity):
    return cv2.blur(image, (int(intensity), int(intensity)))

def collect_data(world, num_scenarios, frames_per_scenario):
    data = []
    labels = []
    
    for scenario in range(num_scenarios):
        print(f"Scenario {scenario + 1}/{num_scenarios}")
        
        # Spawn vehicle
        spawn_points = world.get_map().get_spawn_points()
        vehicle = spawn_vehicle(world, random.choice(spawn_points))
        if vehicle is None:
            continue
        
        # Setup sensors
        lidar_sensor, camera_sensor = setup_sensors(world, vehicle)
        
        # Initialize data buffers
        lidar_buffer = deque(maxlen=100)
        camera_buffer = deque(maxlen=100)
        
        # Choose spoof mode for this scenario
        spoof_mode = random.choice([0, 1, 2, 3])  # 0: No spoof, 1: LiDAR, 2: Camera, 3: Both
        spoof_intensity = random.uniform(5, 20)
        
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
        
        for _ in range(frames_per_scenario):
            # Wait for sensor data
            while lidar_data is None or camera_data is None:
                world.tick()
            
            # Apply spoofing
            if spoof_mode in [1, 3]:
                lidar_data = spoof_lidar_data(lidar_data, spoof_intensity)
            if spoof_mode in [2, 3]:
                camera_data = spoof_camera_image(camera_data, spoof_intensity)
            
            # Extract features
            lidar_features = extract_lidar_features(lidar_data)
            camera_features = extract_camera_features(camera_data)
            
            # Add to buffers
            lidar_buffer.append(lidar_features)
            camera_buffer.append(camera_features)
            
            # If buffers are full, add to dataset
            if len(lidar_buffer) == 100 and len(camera_buffer) == 100:
                combined_features = np.concatenate([lidar_buffer, camera_buffer], axis=1)
                data.append(combined_features.flatten())
                labels.append(spoof_mode)
        
        # Clean up
        lidar_sensor.stop()
        camera_sensor.stop()
        lidar_sensor.destroy()
        camera_sensor.destroy()
        vehicle.destroy()
    
    return np.array(data), np.array(labels)

def main():
    world = setup_carla_world()
    num_scenarios = 150
    frames_per_scenario = 200
    
    start_time = time.time()
    data, labels = collect_data(world, num_scenarios, frames_per_scenario)
    end_time = time.time()
    
    print(f"Data collection completed in {end_time - start_time:.2f} seconds")
    print(f"Collected {len(data)} samples")
    
    # Save data (you might want to use a more sophisticated method like HDF5 for large datasets)
    np.save('town3_data.npy', data)
    np.save('town3_labels.npy', labels)

if __name__ == '__main__':
    main()

