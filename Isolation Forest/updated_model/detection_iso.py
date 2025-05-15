import carla
import pygame
import numpy as np
import cv2
import random
import time
from sklearn.ensemble import IsolationForest
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

# Constants for anomaly detection
ANOMALY_THRESHOLD = -0.6     # More strict threshold
DETECTION_WINDOW = 50        # Reduced window for faster detection
TRAINING_PERIOD = 1000       # Increased training samples
MIN_POINTS_THRESHOLD = 100   # Minimum LiDAR points to consider
CONFIDENCE_THRESHOLD = 0.9   # Required confidence for anomaly detection
DETECTION_INTERVAL = 1.0     # Detection check interval in seconds

# Initialize Isolation Forest models with improved parameters
lidar_iforest = IsolationForest(
    n_estimators=200,  # Increased number of trees
    max_samples='auto',
    contamination=0.05,  # Reduced contamination assumption
    max_features=1.0,
    bootstrap=True,
    n_jobs=-1,  # Parallel processing
    random_state=42
)

camera_iforest = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=0.05,
    max_features=1.0,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

class SensorSystem:
    def __init__(self):
        self.lidar_data = None
        self.camera_image = None
        self.lidar_features_buffer = deque(maxlen=DETECTION_WINDOW)
        self.camera_features_buffer = deque(maxlen=DETECTION_WINDOW)
        self.training_data_collected = False
        self.samples_collected = 0
        self.spoof_mode = 0
        self.detection_mode = False
        self.last_detection_time = 0

    def extract_lidar_features(self, points):
        if len(points) < MIN_POINTS_THRESHOLD:
            return np.zeros(20)
        
        try:
            # Basic statistical features
            basic_stats = np.concatenate([
                np.mean(points, axis=0),
                np.std(points, axis=0),
                np.percentile(points, [25, 50, 75], axis=0).flatten()
            ])
            
            # Advanced geometric features
            volume = np.prod(np.max(points, axis=0) - np.min(points, axis=0))
            density = len(points) / volume if volume > 0 else 0
            
            # Distance distribution
            distances = np.linalg.norm(points, axis=1)
            distance_stats = [
                np.mean(distances),
                np.std(distances),
                np.max(distances),
                np.min(distances)
            ]
            
            # Spatial distribution
            covariance = np.cov(points.T)
            eigenvalues = np.linalg.eigvals(covariance)
            eigenvalues = np.sort(np.abs(eigenvalues))
            linearity = (eigenvalues[2] - eigenvalues[1]) / eigenvalues[2]
            planarity = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]
            sphericity = eigenvalues[0] / eigenvalues[2]
            
            return np.concatenate([
                basic_stats,
                [density],
                distance_stats,
                [linearity, planarity, sphericity]
            ])
            
        except Exception as e:
            print(f"Error in LiDAR feature extraction: {str(e)}")
            return np.zeros(20)

    def extract_camera_features(self, image):
        if image.size == 0:
            return np.zeros(15)
        
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Basic statistics in different channels
            gray_stats = [np.mean(gray), np.std(gray)]
            hsv_stats = [np.mean(hsv[:,:,i]) for i in range(3)]
            
            # Edge detection features
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
            
            # Texture features
            glcm = cv2.resize(gray, (32, 32))
            texture_features = [
                np.mean(glcm),
                np.var(glcm),
                np.sum(glcm ** 2),
                -np.sum(glcm * np.log2(glcm + 1e-10))
            ]
            
            # Gradient features
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            gradient_stats = [np.mean(gradient_magnitude), np.std(gradient_magnitude)]
            
            return np.concatenate([
                gray_stats,
                hsv_stats,
                [edge_density],
                texture_features,
                gradient_stats
            ])
            
        except Exception as e:
            print(f"Error in camera feature extraction: {str(e)}")
            return np.zeros(15)

    def detect_anomalies(self, features, model, threshold):
        if len(features) < DETECTION_WINDOW:
            return False, 0
        
        try:
            scores = model.score_samples(features)
            anomaly_scores = scores < threshold
            confidence = np.mean(anomaly_scores)
            
            is_anomaly = (np.mean(scores) < threshold and 
                         confidence > CONFIDENCE_THRESHOLD)
            
            return is_anomaly, confidence
            
        except Exception as e:
            print(f"Error in anomaly detection: {str(e)}")
            return False, 0

    def check_for_anomalies(self):
        if not (len(self.lidar_features_buffer) == DETECTION_WINDOW and 
                len(self.camera_features_buffer) == DETECTION_WINDOW):
            return None
        
        lidar_features = np.array(list(self.lidar_features_buffer))
        camera_features = np.array(list(self.camera_features_buffer))
        
        lidar_anomaly, lidar_confidence = self.detect_anomalies(
            lidar_features, lidar_iforest, ANOMALY_THRESHOLD)
        camera_anomaly, camera_confidence = self.detect_anomalies(
            camera_features, camera_iforest, ANOMALY_THRESHOLD)
        
        return {
            'lidar': (lidar_anomaly, lidar_confidence),
            'camera': (camera_anomaly, camera_confidence)
        }

def setup_carla():
    # Spawn vehicle blueprint and setup
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    
    # Get spawn points and choose one
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    
    # Try spawning vehicle with collision handling
    vehicle = None
    max_retries = 10
    retries = 0
    
    while vehicle is None and retries < max_retries:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            print(f"Failed to spawn vehicle, retrying... ({retries}/{max_retries})")
            retries += 1
            spawn_point.location.y += random.uniform(-2.0, 2.0)
            spawn_point.location.x += random.uniform(-2.0, 2.0)
    
    if vehicle is None:
        raise Exception("Failed to spawn vehicle")
    
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
    
    return vehicle, lidar_sensor, camera_sensor

def send_sms_alert(spoof_type):
    try:
        message = twilio_client.messages.create(
            body=f"ALERT: Potential {spoof_type} detected in autonomous vehicle sensors!",
            from_=TWILIO_PHONE_NUMBER,
            to=YOUR_PHONE_NUMBER
        )
        print(f"SMS alert sent: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS alert: {str(e)}")

def main():
    try:
        # Setup display
        width, height = 800, 600
        window = pygame.display.set_mode((width * 2, height))
        pygame.display.set_caption("CARLA Sensor Anomaly Detection")
        
        # Initialize sensor system
        sensor_system = SensorSystem()
        
        # Setup CARLA
        vehicle, lidar_sensor, camera_sensor = setup_carla()
        
        # Sensor callbacks
        def process_lidar(point_cloud):
            points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
            if sensor_system.spoof_mode in [1, 3]:
                noise = np.random.normal(0, 2, points.shape)
                points = points + noise
            sensor_system.lidar_data = points
            
            if sensor_system.detection_mode:
                features = sensor_system.extract_lidar_features(points)
                sensor_system.lidar_features_buffer.append(features)
        
        def process_camera(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))[:, :, :3]
            if sensor_system.spoof_mode in [2, 3]:
                array = cv2.GaussianBlur(array, (15, 15), 0)
            sensor_system.camera_image = array
            
            if sensor_system.detection_mode:
                features = sensor_system.extract_camera_features(array)
                sensor_system.camera_features_buffer.append(features)
        
        lidar_sensor.listen(process_lidar)
        camera_sensor.listen(process_camera)
        
        # Main game loop
        running = True
        clock = pygame.time.Clock()
        
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        sensor_system.spoof_mode = 1
                        print("LiDAR spoofing activated")
                    elif event.key == pygame.K_2:
                        sensor_system.spoof_mode = 2
                        print("Camera spoofing activated")
                    elif event.key == pygame.K_3:
                        sensor_system.spoof_mode = 3
                        print("Both spoofing activated")
                    elif event.key == pygame.K_0:
                        sensor_system.spoof_mode = 0
                        print("Spoofing deactivated")
                    elif event.key == pygame.K_d:
                        sensor_system.detection_mode = not sensor_system.detection_mode
                        print(f"Detection mode {'activated' if sensor_system.detection_mode else 'deactivated'}")
            
            # Training phase
            if sensor_system.detection_mode and not sensor_system.training_data_collected:
                if sensor_system.samples_collected < TRAINING_PERIOD:
                    if len(sensor_system.lidar_features_buffer) > 0 and len(sensor_system.camera_features_buffer) > 0:
                        sensor_system.samples_collected += 1
                        if sensor_system.samples_collected == TRAINING_PERIOD:
                            print("Training Isolation Forest models...")
                            lidar_iforest.fit(np.array(list(sensor_system.lidar_features_buffer)))
                            camera_iforest.fit(np.array(list(sensor_system.camera_features_buffer)))
                            sensor_system.training_data_collected = True
                            print("Models trained successfully!")
            
            # Detection phase
            current_time = time.time()
            if (sensor_system.training_data_collected and 
                sensor_system.detection_mode and 
                current_time - sensor_system.last_detection_time > DETECTION_INTERVAL):
                
                anomalies = sensor_system.check_for_anomalies()
                if anomalies:
                    lidar_result = anomalies['lidar']
                    camera_result = anomalies['camera']
                    
                    if lidar_result[0] and camera_result[0]:
                        print(f"WARNING: Both sensors showing anomalies! "
                              f"Confidence - LiDAR: {lidar_result[1]:.2f}, "
                              f"Camera: {camera_result[1]:.2f}")
                        send_sms_alert("LiDAR and Camera spoofing")
                    elif lidar_result[0]:
                        print(f"WARNING: LiDAR anomaly detected! "
                              f"Confidence: {lidar_result[1]:.2f}")
                        send_sms_alert("LiDAR spoofing")
                    elif camera_result[0]:
                        print(f"WARNING: Camera anomaly detected! "
                              f"Confidence: {camera_result[1]:.2f}")
                        send_sms_alert("Camera spoofing")
                    
                    sensor_system.last_detection_time = current_time
            
            # Visualization
            if sensor_system.lidar_data is not None:
                lidar_surface = np.zeros((height, width, 3), dtype=np.uint8)
                for point in sensor_system.lidar_data:
                    x = int(width/2 + point[1] * 4)
                    y = int(height/2 - point[0] * 4)
                    if 0 <= x < width and 0 <= y < height:
                        lidar_surface[y, x] = (255, 255, 255)
                
                lidar_surface = pygame.surfarray.make_surface(
                    cv2.cvtColor(lidar_surface, cv2.COLOR_BGR2RGB))
                window.blit(lidar_surface, (0, 0))
            
            if sensor_system.camera_image is not None:
                camera_surface = pygame.surfarray.make_surface(
                    sensor_system.camera_image.swapaxes(0, 1))
                window.blit(camera_surface, (width, 0))
            
            # Add status overlay
            font = pygame.font.Font(None, 36)
            status_texts = [
                f"Spoof Mode: {sensor_system.spoof_mode}",
                f"Detection: {'ON' if sensor_system.detection_mode else 'OFF'}",
                f"Training: {sensor_system.samples_collected}/{TRAINING_PERIOD}",
                f"Model Status: {'Trained' if sensor_system.training_data_collected else 'Training'}"
            ]
            
            for i, text in enumerate(status_texts):
                surface = font.render(text, True, (255, 255, 255))
                window.blit(surface, (10, 10 + i * 30))
            
            pygame.display.flip()
            clock.tick(30)
    
    except Exception as e:
        print(f"An error occurred in main loop: {str(e)}")
        
    finally:
        print("Cleaning up...")
        try:
            lidar_sensor.destroy()
            camera_sensor.destroy()
            vehicle.destroy()
        except:
            pass
        pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting by user request...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        pygame.quit()
