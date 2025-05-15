import carla
import pygame
import numpy as np
import cv2
import random
import time
import math
import torch
from torchvision import transforms, models
from PIL import Image
from twilio.rest import Client

class CarlaDetectionSystem:
    def __init__(self):
        # Initialize pygame and display
        pygame.init()
        self.width, self.height = 800, 600
        self.window = pygame.display.set_mode((self.width * 2, self.height))
        pygame.display.set_caption("CARLA Detection System")

        # Twilio configuration
        self.twilio_client = Client('token', 'token')
        self.from_number = '#number'
        self.to_number = '#number'
        
        # Alert status tracking
        self.alert_status = {
            'lidar_only': False,
            'camera_only': False,
            'both_sensors': False
        }

        # Initialize CARLA client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # Setup device and models
        self.setup_models()
        
        # Initialize state variables
        self.spoof_mode = 0  # 0: no spoof, 1: LiDAR spoof, 2: Camera spoof, 3: Both spoof
        self.detection_mode = False
        self.last_lidar_detection_time = time.time()
        self.last_camera_detection_time = time.time()
        
        # Initialize sensor data storage
        self.lidar_data = []
        self.camera_image = None
        
        # Spawn vehicle and sensors
        self.setup_vehicle_and_sensors()

    def setup_models(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Transform for both models
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Camera model
        self.camera_model = models.resnet50(pretrained=False)
        self.camera_model.fc = torch.nn.Linear(self.camera_model.fc.in_features, 2)
        self.camera_model.load_state_dict(torch.load('best_camera_spoofing_model.pth', map_location=self.device))
        self.camera_model = self.camera_model.to(self.device)
        self.camera_model.eval()

        # LiDAR model
        self.lidar_model = models.resnet50(pretrained=False)
        self.lidar_model.fc = torch.nn.Linear(self.lidar_model.fc.in_features, 2)
        self.lidar_model.load_state_dict(torch.load('best_lidar_spoofing_model.pth', map_location=self.device))
        self.lidar_model = self.lidar_model.to(self.device)
        self.lidar_model.eval()

    def send_alert(self, alert_type):
        """Send SMS alert based on the type of sensor compromise"""
        if not self.alert_status[alert_type]:  # Only send if not already sent
            messages = {
                'lidar_only': "Your Car's LIDAR sensor has been compromised and the vehicle is relying on the camera sensor only. You can take over the control of the vehicle or contact the brand.",
                'camera_only': "Your Car's CAMERA sensor has been compromised and the vehicle is relying on the lidar sensor only. You can take over the control of the vehicle or contact the brand.",
                'both_sensors': "HIGH EMERGENCY!!! : Your Car's sensor has been compromised. The vehicle is stopping. Please take over the control of the vehicle to continue the journey. The brand has been notified and will contact you shortly regarding the issue."
            }

            try:
                message = self.twilio_client.messages.create(
                    body=messages[alert_type],
                    from_=self.from_number,
                    to=self.to_number
                )
                print(f"Alert sent: {alert_type}")
                self.alert_status[alert_type] = True  # Mark this alert as sent
            except Exception as e:
                print(f"Failed to send alert: {e}")

    def check_sensor_status(self, pred_lidar, pred_camera):
        """Check sensor status and send appropriate alerts"""
        if pred_lidar == 1 and pred_camera == 0:
            self.send_alert('lidar_only')
        elif pred_lidar == 0 and pred_camera == 1:
            self.send_alert('camera_only')
        elif pred_lidar == 1 and pred_camera == 1:
            self.send_alert('both_sensors')
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            self.vehicle_stopped = True
            

    def reset_alert_status(self):
        """Reset alert status when spoofing is disabled"""
        if self.spoof_mode == 0:
            self.alert_status = {
                'lidar_only': False,
                'camera_only': False,
                'both_sensors': False
            }

    def setup_vehicle_and_sensors(self):
        blueprint_library = self.world.get_blueprint_library()
        
        # Spawn vehicle
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.try_spawn_vehicle_with_retries(vehicle_bp, random.choice(spawn_points))
        
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle")
            
        # Enable autopilot
        self.vehicle.set_autopilot(True)
        
        # Set target speed (10 km/h = approximately 2.78 m/s)
        self.world.tick()  # Wait for the world to update
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.global_percentage_speed_difference(72)  # Set to 72% slower than default
        traffic_manager.vehicle_percentage_speed_difference(self.vehicle, 72)

        # Setup sensors
        self.setup_sensors(blueprint_library)

    def try_spawn_vehicle_with_retries(self, bp, spawn_point, max_retries=10):
        vehicle = None
        retries = 0
        while vehicle is None and retries < max_retries:
            vehicle = self.world.try_spawn_actor(bp, spawn_point)
            if vehicle is None:
                retries += 1
                spawn_point.location.y += random.uniform(-2.0, 2.0)
                spawn_point.location.x += random.uniform(-2.0, 2.0)
        return vehicle

    def setup_sensors(self, blueprint_library):
        # LiDAR setup
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100')
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.lidar_sensor.listen(self.process_lidar_data)

        # Camera setup
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.5))
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera_sensor.listen(self.process_camera_data)

    def process_lidar_data(self, point_cloud):
        points = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3])
        self.lidar_data = self.spoof_lidar_data(points)

    def process_camera_data(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
        self.camera_image = self.spoof_camera_image(array)

    def spoof_lidar_data(self, points):
        if self.spoof_mode in [1, 3]:
            noise = np.random.normal(0, 10, points.shape)
            points += noise
            points = points[points[:, 2] > 1]
        return points

    def spoof_camera_image(self, image):
        if self.spoof_mode in [2, 3]:
            image = cv2.blur(image, (30, 30))
        return image

    def generate_lidar_image(self, points, width=224, height=224):
        density_map = np.zeros((height, width), dtype=np.float32)
        height_map = np.zeros((height, width), dtype=np.float32)
        intensity_map = np.zeros((height, width), dtype=np.float32)
        
        if len(points) == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
            
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        x_scale = (width - 1) / (x_max - x_min) if x_max != x_min else 1
        y_scale = (height - 1) / (y_max - y_min) if y_max != y_min else 1
        
        for point in points:
            x = int((point[0] - x_min) * x_scale)
            y = int((point[1] - y_min) * y_scale)
            
            if 0 <= x < width and 0 <= y < height:
                density_map[y, x] += 1
                height_map[y, x] = max(height_map[y, x], (point[2] - z_min) / (z_max - z_min + 1e-6))
                intensity_map[y, x] = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        
        density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
        height_map = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX)
        intensity_map = cv2.normalize(intensity_map, None, 0, 255, cv2.NORM_MINMAX)
        
        return np.stack([density_map, height_map, intensity_map], axis=2).astype(np.uint8)

    def predict_spoofing(self, image, model, is_lidar=False):
        if is_lidar:
            image = Image.fromarray(image)
        else:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = model(image)
            confidence = torch.softmax(output, 1)[0].cpu().numpy()
            _, predicted = torch.max(output, 1)
        return predicted.item(), confidence

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_1:
                    self.spoof_mode = 1
                    print("LiDAR spoof mode enabled")
                elif event.key == pygame.K_2:
                    self.spoof_mode = 2
                    print("Camera spoof mode enabled")
                elif event.key == pygame.K_3:
                    self.spoof_mode = 3
                    print("Both sensors spoof mode enabled")
                elif event.key == pygame.K_0:
                    self.spoof_mode = 0
                    print("Spoof mode disabled")
                elif event.key == pygame.K_d:
                    self.detection_mode = not self.detection_mode
                    print(f"Detection mode {'enabled' if self.detection_mode else 'disabled'}")
        return True

    def update_display(self):
        current_time = time.time()
        pred_lidar = 0
        pred_camera = 0

        # Create empty surfaces
        lidar_surface = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Update LiDAR visualization
        if len(self.lidar_data) > 0:
            scale_factor = 4
            for point in self.lidar_data:
                x = int(self.width / 2 + point[1] * scale_factor)
                y = int(self.height / 2 - point[0] * scale_factor)
                if 0 <= x < self.width and 0 <= y < self.height:
                    lidar_surface[y, x] = (255, 255, 255)

            # LiDAR detection
            if self.detection_mode and current_time - self.last_lidar_detection_time >= 5.0:
                lidar_image = self.generate_lidar_image(self.lidar_data)
                pred_lidar, conf_lidar = self.predict_spoofing(lidar_image, self.lidar_model, is_lidar=True)
                print(f"LiDAR Prediction: {'Spoofed' if pred_lidar == 1 else 'Normal'} "
                      f"(Confidence: {conf_lidar[pred_lidar]:.2f})")
                self.last_lidar_detection_time = current_time

        # Update camera visualization
        camera_surface = None
        if self.camera_image is not None:
            camera_surface = pygame.surfarray.make_surface(self.camera_image.swapaxes(0, 1))
            
            # Camera detection
            if self.detection_mode and current_time - self.last_camera_detection_time >= 5.0:
                pred_camera, conf_camera = self.predict_spoofing(self.camera_image, self.camera_model)
                print(f"Camera Prediction: {'Spoofed' if pred_camera == 1 else 'Normal'} "
                      f"(Confidence: {conf_camera[pred_camera]:.2f})")
                self.last_camera_detection_time = current_time

        # Check sensor status and send alerts if needed
        if self.detection_mode:
            self.check_sensor_status(pred_lidar, pred_camera)
        
        # Reset alert status if spoofing is disabled
        self.reset_alert_status()

        # Update display
        self.window.blit(pygame.surfarray.make_surface(lidar_surface), (0, 0))
        if camera_surface is not None:
            self.window.blit(camera_surface, (self.width, 0))

        pygame.display.flip()

    def cleanup(self):
        self.lidar_sensor.destroy()
        self.camera_sensor.destroy()
        self.vehicle.destroy()
        pygame.quit()

    def run(self):
        try:
            clock = pygame.time.Clock()
            running = True
            while running:
                running = self.handle_events()
                self.update_display()
                clock.tick(30)
        finally:
            self.cleanup()

if __name__ == '__main__':
    try:
        detection_system = CarlaDetectionSystem()
        detection_system.run()
    except Exception as e:
        print(f"Error occurred: {e}")
        pygame.quit()
