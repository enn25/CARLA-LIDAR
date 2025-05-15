import carla
import random
import numpy as np
import os
import json
import queue
import time
import matplotlib.pyplot as plt
from datetime import datetime

class CarlaLidarDatasetGenerator:
    def __init__(self):
        try:
            # Connect to CARLA server
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            
            # Check version compatibility
            client_version = self.client.get_client_version()
            server_version = self.client.get_server_version()
            print(f"Client version: {client_version}")
            print(f"Server version: {server_version}")
            
            if client_version != server_version:
                print(f"WARNING: Version mismatch detected! Client: {client_version}, Server: {server_version}")
                response = input("Do you want to continue anyway? (y/n): ")
                if response.lower() != 'y':
                    exit(1)
            
            self.world = self.client.get_world()
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            
            # Setup paths
            self.output_path = 'lidar_dataset'
            self.normal_images_path = os.path.join(self.output_path, 'normal', 'images')
            self.normal_labels_path = os.path.join(self.output_path, 'normal', 'labels')
            self.spoofed_images_path = os.path.join(self.output_path, 'spoofed', 'images')
            self.spoofed_labels_path = os.path.join(self.output_path, 'spoofed', 'labels')
            
            os.makedirs(self.normal_images_path, exist_ok=True)
            os.makedirs(self.normal_labels_path, exist_ok=True)
            os.makedirs(self.spoofed_images_path, exist_ok=True)
            os.makedirs(self.spoofed_labels_path, exist_ok=True)
            
            # Initialize sensors
            self.lidar = None
            self.sensor_queue = queue.Queue()
            
        except Exception as e:
            print(f"Initialization error: {e}")
            raise
    
    def sensor_callback(self, data):
        """Callback function for sensor data"""
        try:
            self.sensor_queue.put(data)
        except Exception as e:
            print(f"Error in sensor callback: {e}")
    
    def setup_lidar_sensor(self, vehicle):
        try:
            # Lidar configuration
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '100.0')
            lidar_bp.set_attribute('rotation_frequency', '10.0')
            lidar_bp.set_attribute('channels', '64')
            lidar_bp.set_attribute('points_per_second', '500000')
            
            # Spawn lidar sensor
            lidar_transform = carla.Transform(carla.Location(z=2.0))  # Position above the vehicle
            self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
            self.lidar.listen(lambda point_cloud: self.sensor_callback(point_cloud))
            
            # Wait for sensor to be ready
            time.sleep(0.5)
        except Exception as e:
            print(f"Lidar setup error: {e}")
            raise
    
    def wait_for_sensor_data(self, timeout=10.0):
        """Wait for sensor data with timeout"""
        try:
            return self.sensor_queue.get(timeout=timeout)
        except queue.Empty:
            raise Exception("Sensor data timeout")

    def apply_spoofing_effects(self, point_cloud):
        """Apply more intense spoofing effects to the LiDAR point cloud"""
        try:
            effect = random.choice(['drop_points', 'add_noise', 'invert_data'])

            if effect == 'drop_points':
                # Drop more points by keeping only 25% of the original data
                mask = np.random.rand(len(point_cloud)) > 0.75  # Keep 25% of points
                return point_cloud[mask]

            elif effect == 'add_noise':
                # Add stronger noise to the LiDAR point cloud data
                noise = np.random.normal(0, 2.0, point_cloud.shape)  # Higher std deviation
                return point_cloud + noise

            elif effect == 'invert_data':
                # Invert and amplify the LiDAR point cloud data
                return -point_cloud * 2  # Multiply to increase the inversion intensity

        except Exception as e:
            print("Error applying spoofing effect {}: {}".format(effect, e))
            return point_cloud

    def save_point_cloud_image(self, filename, point_cloud):
        """Convert point cloud data to an image and save it"""
        plt.figure(figsize=(10, 10))
        plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=1, c=point_cloud[:, 2], cmap='viridis', marker='.')
        plt.axis('equal')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def generate_normal_data(self, num_images):
        """Generate normal (non-spoofed) data"""
        for i in range(num_images):
            vehicle = None
            try:
                # Spawn vehicle at random location
                spawn_points = self.world.get_map().get_spawn_points()
                vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.*'))
                spawn_point = random.choice(spawn_points)
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                
                if vehicle is None:
                    print(f"Failed to spawn vehicle for image {i}, retrying...")
                    continue
                
                self.setup_lidar_sensor(vehicle)
                
                # Add some random movement
                vehicle.set_autopilot(True)
                time.sleep(0.5)  # Wait for vehicle to start moving
                
                # Capture LiDAR data
                self.world.tick()
                point_cloud = self.wait_for_sensor_data()
                
                # Convert point cloud to numpy array
                point_cloud = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
                
                # Save LiDAR image and labels
                filename = f'normal_{i:04d}'
                self.save_point_cloud_image(os.path.join(self.normal_images_path, f'{filename}.png'), point_cloud)
                
                # Generate labels
                labels = {
                    'filename': f'{filename}.png',
                    'is_spoofed': False,
                    'timestamp': datetime.now().isoformat(),
                    'vehicle_location': [spawn_point.location.x, spawn_point.location.y, spawn_point.location.z],
                    'weather': str(self.world.get_weather())
                }
                
                with open(os.path.join(self.normal_labels_path, f'{filename}.json'), 'w') as f:
                    json.dump(labels, f)
                
                print(f"Generated normal LiDAR image {i+1}/{num_images}")
                
            except Exception as e:
                print(f"Error generating normal LiDAR image {i}: {e}")
                continue
            finally:
                if self.lidar:
                    self.lidar.destroy()
                    self.lidar = None
                if vehicle:
                    vehicle.set_autopilot(False)
                    vehicle.destroy()
                while not self.sensor_queue.empty():
                    self.sensor_queue.get()
    
    def generate_spoofed_data(self, num_images):
        """Generate spoofed data with severe visual distortions"""
        for i in range(num_images):
            vehicle = None
            try:
                spawn_points = self.world.get_map().get_spawn_points()
                vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.*'))
                spawn_point = random.choice(spawn_points)
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                
                if vehicle is None:
                    print(f"Failed to spawn vehicle for spoofed image {i}, retrying...")
                    continue
                
                self.setup_lidar_sensor(vehicle)
                
                # Add some random movement
                vehicle.set_autopilot(True)
                time.sleep(0.5)
                
                self.world.tick()
                point_cloud = self.wait_for_sensor_data()
                
                # Convert point cloud to numpy array
                point_cloud = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
                
                # Apply spoofing effects
                effect_name = random.choice(['drop_points', 'add_noise', 'invert_data'])
                spoofed_point_cloud = self.apply_spoofing_effects(point_cloud.copy())
                
                # Save LiDAR images
                filename = f'spoofed_{i:04d}'
                self.save_point_cloud_image(os.path.join(self.spoofed_images_path, f'{filename}.png'), spoofed_point_cloud)
                
                # Save both original and spoofed images for reference
                self.save_point_cloud_image(os.path.join(self.spoofed_images_path, f'{filename}_original.png'), point_cloud)
                
                labels = {
                    'filename': f'{filename}.png',
                    'original_filename': f'{filename}_original.png',
                    'is_spoofed': True,
                    'spoofing_type': effect_name,
                    'timestamp': datetime.now().isoformat(),
                    'vehicle_location': [spawn_point.location.x, spawn_point.location.y, spawn_point.location.z],
                    'weather': str(self.world.get_weather())
                }
                
                with open(os.path.join(self.spoofed_labels_path, f'{filename}.json'), 'w') as f:
                    json.dump(labels, f)
                
                print(f"Generated spoofed LiDAR image {i+1}/{num_images} with effect: {effect_name}")
                
            except Exception as e:
                print(f"Error generating spoofed LiDAR image {i}: {e}")
                continue
            finally:
                if self.lidar:
                    self.lidar.destroy()
                    self.lidar = None
                if vehicle:
                    vehicle.set_autopilot(False)
                    vehicle.destroy()
                while not self.sensor_queue.empty():
                    self.sensor_queue.get()

