import carla
import random
import numpy as np
import cv2
import time
from collections import deque

class CarlaSensorManager:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.lidar_sensor = None
        self.camera_sensor = None
        self.lidar_data = None
        self.camera_data = None

    def setup_sensors(self):
        blueprint_library = self.world.get_blueprint_library()
        
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100')
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.5))
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        
        self.lidar_sensor.listen(self.lidar_callback)
        self.camera_sensor.listen(self.camera_callback)

    def lidar_callback(self, data):
        self.lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]

    def camera_callback(self, data):
        self.camera_data = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4))[:, :, :3]

    def wait_for_data(self):
        while self.lidar_data is None or self.camera_data is None:
            self.world.tick()

    def clean_up(self):
        if self.lidar_sensor:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
        if self.camera_sensor:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()

class CarlaDataCollector:
    def __init__(self, client, maps, num_scenarios_per_map, frames_per_scenario):
        self.client = client
        self.maps = maps
        self.num_scenarios_per_map = num_scenarios_per_map
        self.frames_per_scenario = frames_per_scenario

    def setup_carla_world(self, map_name):
        self.world = self.client.load_world(map_name)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        return self.world

    def spawn_vehicle(self, spawn_point):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        return vehicle

    @staticmethod
    def extract_lidar_features(points):
        return np.concatenate([
            np.mean(points, axis=0),
            np.std(points, axis=0),
            np.percentile(points, [25, 50, 75], axis=0).flatten()
        ])

    @staticmethod
    def extract_camera_features(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.concatenate([
            [np.mean(gray)],
            [np.std(gray)],
            np.percentile(gray, [25, 50, 75]),
            [cv2.Laplacian(gray, cv2.CV_64F).var()]
        ])

    @staticmethod
    def spoof_lidar_data(points, intensity):
        points = np.copy(points)
        noise = np.random.normal(0, intensity, points.shape)
        points += noise
        return points[points[:, 2] > 0.5]

    @staticmethod
    def spoof_camera_image(image, intensity):
        return cv2.blur(image, (int(intensity), int(intensity)))

    def collect_data_from_map(self, map_name):
        print(f"Collecting data from map: {map_name}")
        self.setup_carla_world(map_name)
        data = []
        labels = []

        for scenario in range(self.num_scenarios_per_map):
            print(f"Scenario {scenario + 1}/{self.num_scenarios_per_map}")
            
            spawn_points = self.world.get_map().get_spawn_points()
            vehicle = self.spawn_vehicle(random.choice(spawn_points))
            if vehicle is None:
                continue
            
            sensor_manager = CarlaSensorManager(self.world, vehicle)
            sensor_manager.setup_sensors()
            
            lidar_buffer = deque(maxlen=100)
            camera_buffer = deque(maxlen=100)
            
            spoof_mode = random.choice([0, 1, 2, 3])
            spoof_intensity = random.uniform(5, 20)
            
            for _ in range(self.frames_per_scenario):
                sensor_manager.wait_for_data()
                
                lidar_data = sensor_manager.lidar_data
                camera_data = sensor_manager.camera_data
                
                if spoof_mode in [1, 3]:
                    lidar_data = self.spoof_lidar_data(lidar_data, spoof_intensity)
                if spoof_mode in [2, 3]:
                    camera_data = self.spoof_camera_image(camera_data, spoof_intensity)
                
                lidar_features = self.extract_lidar_features(lidar_data)
                camera_features = self.extract_camera_features(camera_data)
                
                lidar_buffer.append(lidar_features)
                camera_buffer.append(camera_features)
                
                if len(lidar_buffer) == 100 and len(camera_buffer) == 100:
                    combined_features = np.concatenate([lidar_buffer, camera_buffer], axis=1)
                    data.append(combined_features.flatten())
                    labels.append(spoof_mode)
            
            sensor_manager.clean_up()
            vehicle.destroy()

        return np.array(data), np.array(labels)

    def collect_all_data(self):
        all_data = []
        all_labels = []

        for map_name in self.maps:
            map_data, map_labels = self.collect_data_from_map(map_name)
            all_data.append(map_data)
            all_labels.append(map_labels)

        return np.concatenate(all_data, axis=0), np.concatenate(all_labels, axis=0)

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']
    num_scenarios_per_map = 250
    frames_per_scenario = 200

    collector = CarlaDataCollector(client, maps, num_scenarios_per_map, frames_per_scenario)

    start_time = time.time()
    combined_data, combined_labels = collector.collect_all_data()
    end_time = time.time()

    print(f"Data collection completed in {end_time - start_time:.2f} seconds")
    print(f"Collected {len(combined_data)} samples across {len(maps)} maps")

    np.save('spoof_classifier_data_multi_map.npy', combined_data)
    np.save('spoof_classifier_labels_multi_map.npy', combined_labels)

if __name__ == '__main__':
    main()
