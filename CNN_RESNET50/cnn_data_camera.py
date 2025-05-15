import carla
import random
import numpy as np
import os
import json
from datetime import datetime
import cv2
import queue
import time
import sys

class CarlaFrontCameraDatasetGenerator:
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
                    sys.exit(1)
            
            self.world = self.client.get_world()
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            
            # Setup paths
            self.output_path = 'dataset'
            self.images_path = os.path.join(self.output_path, 'images')
            self.labels_path = os.path.join(self.output_path, 'labels')
            os.makedirs(self.images_path, exist_ok=True)
            os.makedirs(self.labels_path, exist_ok=True)
            
            # Initialize sensors
            self.camera = None
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
    
    def setup_front_camera(self, vehicle):
        try:
            # Camera configuration for front view
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '1920')
            camera_bp.set_attribute('image_size_y', '1080')
            camera_bp.set_attribute('fov', '90')  # Wider FOV for front view
            
            # Spawn camera at front of vehicle
            camera_transform = carla.Transform(
                carla.Location(x=2.5, z=1.0),  # Position camera at front
                carla.Rotation(pitch=-15)  # Angle slightly downward
            )
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            self.camera.listen(lambda image: self.sensor_callback(image))
            
            # Wait for camera to be ready
            time.sleep(0.5)
        except Exception as e:
            print(f"Camera setup error: {e}")
            raise
    
    def wait_for_sensor_data(self, timeout=10.0):
        """Wait for sensor data with timeout"""
        try:
            return self.sensor_queue.get(timeout=timeout)
        except queue.Empty:
            raise Exception("Sensor data timeout")

    def apply_spoofing_effects(self, image):
        """Apply severe spoofing effects to the image"""
        try:
            effect = 'extreme_blur'
            
            if effect == 'extreme_blur':
                # Apply extreme Gaussian blur
                kernel_size = random.choice([45, 55, 65])  # Large kernel for severe blur
                return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
            
        except Exception as e:
            print(f"Error applying spoofing effect {effect}: {e}")
            return image
    
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
                
                self.setup_front_camera(vehicle)
                
                # Add some random movement
                vehicle.set_autopilot(True)
                time.sleep(0.5)  # Wait for vehicle to start moving
                
                # Capture image
                self.world.tick()
                image = self.wait_for_sensor_data()
                
                # Convert image to numpy array
                img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                img_array = np.reshape(img_array, (image.height, image.width, 4))
                img_array = img_array[:, :, :3]
                
                # Save image and labels
                filename = f'normal_{i:04d}'
                cv2.imwrite(os.path.join(self.images_path, f'{filename}.jpg'), img_array)
                
                # Generate labels
                labels = {
                    'filename': f'{filename}.jpg',
                    'is_spoofed': False,
                    'timestamp': datetime.now().isoformat(),
                    'vehicle_location': [spawn_point.location.x, spawn_point.location.y, spawn_point.location.z],
                    'camera_fov': 90,
                    'weather': str(self.world.get_weather()),
                    'image_resolution': [image.width, image.height]
                }
                
                with open(os.path.join(self.labels_path, f'{filename}.json'), 'w') as f:
                    json.dump(labels, f)
                
                print(f"Generated normal image {i+1}/{num_images}")
                
            except Exception as e:
                print(f"Error generating normal image {i}: {e}")
                continue
            finally:
                if self.camera:
                    self.camera.destroy()
                    self.camera = None
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
                
                self.setup_front_camera(vehicle)
                
                # Add some random movement
                vehicle.set_autopilot(True)
                time.sleep(0.5)
                
                self.world.tick()
                image = self.wait_for_sensor_data()
                
                # Convert image to numpy array
                img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                img_array = np.reshape(img_array, (image.height, image.width, 4))
                img_array = img_array[:, :, :3]
                
                # Apply spoofing effects
                effect_name ='extreme_blur'
                spoofed_image = self.apply_spoofing_effects(img_array.copy())
                
                # Save image and labels
                filename = f'spoofed_{i:04d}'
                cv2.imwrite(os.path.join(self.images_path, f'{filename}.jpg'), spoofed_image)
                
                # Save both original and spoofed images for reference
                #cv2.imwrite(os.path.join(self.images_path, f'{filename}_original.jpg'), img_array)
                
                labels = {
                    'filename': f'{filename}.jpg',
                    'original_filename': f'{filename}_original.jpg',
                    'is_spoofed': True,
                    'spoofing_effect': effect_name,
                    'timestamp': datetime.now().isoformat(),
                    'vehicle_location': [spawn_point.location.x, spawn_point.location.y, spawn_point.location.z],
                    'camera_fov': 90,
                    'weather': str(self.world.get_weather()),
                    'image_resolution': [image.width, image.height]
                }
                
                with open(os.path.join(self.labels_path, f'{filename}.json'), 'w') as f:
                    json.dump(labels, f)
                
                print(f"Generated spoofed image {i+1}/{num_images}")
                
            except Exception as e:
                print(f"Error generating spoofed image {i}: {e}")
                continue
            finally:
                if self.camera:
                    self.camera.destroy()
                    self.camera = None
                if vehicle:
                    vehicle.set_autopilot(False)
                    vehicle.destroy()
                while not self.sensor_queue.empty():
                    self.sensor_queue.get()

    def run(self, normal_images=100, spoofed_images=100):
        """Main function to generate dataset"""
        try:
            print("Generating normal images...")
            self.generate_normal_data(normal_images)
            print("Normal images generated successfully!")
            
            print("Generating spoofed images...")
            self.generate_spoofed_data(spoofed_images)
            print("Spoofed images generated successfully!")
            
        except Exception as e:
            print(f"Error during dataset generation: {e}")
        finally:
            # Clean up
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            print("Finished dataset generation.")

if __name__ == '__main__':
    generator = CarlaFrontCameraDatasetGenerator()
    generator.run(normal_images=150, spoofed_images=150)

