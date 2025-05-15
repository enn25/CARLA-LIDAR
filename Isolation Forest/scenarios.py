import pygame
import numpy as np
import random
import time

# Initialize Pygame
pygame.init()

# Set up display
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Autonomous Vehicle Simulation")

# Global variables
spoof_mode = 0  # 0: No spoofing, 1: LiDAR spoofing, 2: Camera spoofing, 3: Both
data_list = []  # To store collected data
labels_list = []  # To store labels
EXIT_KEY = pygame.K_ESCAPE  # Set ESC key as the exit key

# Initialize LiDAR and camera sensors
class LidarSensor:
    def __init__(self):
        pass

    def get_data(self):
        return [random.uniform(0, 10) for _ in range(360)]  # Simulated LiDAR data

class CameraSensor:
    def __init__(self):
        pass

    def get_image(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Simulated camera image

# Initialize vehicle
class Vehicle:
    def __init__(self):
        self.position = [screen_width // 2, screen_height // 2]
    
    def update(self):
        # Update vehicle position and other parameters here
        pass

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 0, 0), (*self.position, 50, 30))  # Draw vehicle

# Initialize sensor and vehicle
lidar_sensor = LidarSensor()
camera_sensor = CameraSensor()
vehicle = Vehicle()

# Function to handle spoof toggling
def handle_spoof_toggle():
    global spoof_mode
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == EXIT_KEY:  # Check for exit key
                return False  # Indicate that we want to exit
            if event.key == pygame.K_1:
                spoof_mode = 1  # Spoof LiDAR only
            elif event.key == pygame.K_2:
                spoof_mode = 2  # Spoof Camera only
            elif event.key == pygame.K_3:
                spoof_mode = 3  # Spoof both
            elif event.key == pygame.K_0:
                spoof_mode = 0  # Disable spoofing
    return True  # Continue running

# Main loop
try:
    while True:
        # Check for exit
        if not handle_spoof_toggle():
            break  # Exit loop if exit key was pressed

        # Update and draw everything
        vehicle.update()
        screen.fill((0, 0, 0))  # Clear the screen
        vehicle.draw(screen)  # Draw the vehicle

        # Handle LiDAR and camera data collection based on spoof mode
        if spoof_mode == 0:
            lidar_data = lidar_sensor.get_data()
            camera_image = camera_sensor.get_image()
            # Process the data and append to lists
            data_list.append(lidar_data)  # Store LiDAR data
            labels_list.append(0)  # Append label (0: no spoofing)

            # Visualize LiDAR data
            for angle in range(len(lidar_data)):
                distance = lidar_data[angle]
                x = int(vehicle.position[0] + distance * np.cos(np.radians(angle)))
                y = int(vehicle.position[1] + distance * np.sin(np.radians(angle)))
                pygame.draw.line(screen, (0, 255, 0), vehicle.position, (x, y), 1)  # Draw LiDAR line

        elif spoof_mode == 1:
            lidar_data = [random.uniform(0, 10) for _ in range(360)]  # Spoofed LiDAR data
            data_list.append(lidar_data)
            labels_list.append(1)  # Append label (1: LiDAR spoofing)

            # Visualize spoofed LiDAR data
            for angle in range(len(lidar_data)):
                distance = lidar_data[angle]
                x = int(vehicle.position[0] + distance * np.cos(np.radians(angle)))
                y = int(vehicle.position[1] + distance * np.sin(np.radians(angle)))
                pygame.draw.line(screen, (255, 0, 0), vehicle.position, (x, y), 1)  # Draw spoofed LiDAR line

        elif spoof_mode == 2:
            camera_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Spoofed camera image
            data_list.append(camera_image.flatten())  # Flatten and store
            labels_list.append(2)  # Append label (2: Camera spoofing)

            # Display the spoofed camera image
            camera_surface = pygame.surfarray.make_surface(camera_image)
            screen.blit(camera_surface, (0, 0))  # Draw camera image at the top-left corner

        elif spoof_mode == 3:
            lidar_data = [random.uniform(0, 10) for _ in range(360)]  # Spoofed LiDAR data
            camera_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Spoofed camera image
            data_list.append(lidar_data + camera_image.flatten().tolist())  # Combine and store
            labels_list.append(3)  # Append label (3: Both spoofing)

            # Visualize spoofed LiDAR data
            for angle in range(len(lidar_data)):
                distance = lidar_data[angle]
                x = int(vehicle.position[0] + distance * np.cos(np.radians(angle)))
                y = int(vehicle.position[1] + distance * np.sin(np.radians(angle)))
                pygame.draw.line(screen, (255, 0, 0), vehicle.position, (x, y), 1)  # Draw spoofed LiDAR line
            
            # Display the spoofed camera image
            camera_surface = pygame.surfarray.make_surface(camera_image)
            screen.blit(camera_surface, (0, 0))  # Draw camera image at the top-left corner

        pygame.display.flip()  # Update the display
        time.sleep(0.1)  # Control the frame rate

finally:
    # Cleanup and save data when exiting
    pygame.quit()

    # Save collected data and labels
    np.save('test_datav3.npy', data_list)
    np.save('test_labelv3.npy', labels_list)

