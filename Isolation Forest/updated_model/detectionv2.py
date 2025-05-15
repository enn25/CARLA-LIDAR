import carla
import pygame
import numpy as np
import cv2
import random
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
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

# Constants for feature extraction
NUM_HISTOGRAM_BINS = 10
NUM_FEATURES_PER_COORDINATE = NUM_HISTOGRAM_BINS
CAMERA_HISTOGRAM_BINS = 10

# SpoofClassifier model definition with updated architecture
class SpoofClassifier(nn.Module):
    def __init__(self):
        super(SpoofClassifier, self).__init__()
        # Calculate input size based on our feature extraction
        lidar_features = 3 + 3 + 9 + (NUM_HISTOGRAM_BINS * 3)  # mean, std, percentiles, histograms
        camera_features = 1 + 1 + 3 + 1 + 1 + CAMERA_HISTOGRAM_BINS + 1  # mean, std, percentiles, blur, edge density, histogram, edge pixels
        self.input_size = lidar_features + camera_features
        
        self.fc1 = nn.Linear(self.input_size, 128)
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

# Function to load the model
def load_model(model_path):
    try:
        model = SpoofClassifier()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Model loaded successfully with input size: {model.input_size}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Load the trained model
model = load_model('best_model.pth')
if model is None:
    print("Failed to load the model. Exiting...")
    pygame.quit()
    exit()

# [Previous vehicle and sensor setup code remains the same until extract_features function]

# Function to extract features for spoofing detection (Fixed version)
def extract_features(lidar_points, camera_image):
    # LiDAR features
    if len(lidar_points) == 0:
        # Handle empty LiDAR data
        lidar_features = np.zeros(3 + 3 + 9 + (NUM_HISTOGRAM_BINS * 3))
    else:
        mean_features = np.mean(lidar_points, axis=0)
        std_features = np.std(lidar_points, axis=0)
        percentile_features = np.percentile(lidar_points, [25, 50, 75], axis=0).flatten()
        
        # Compute histograms for each coordinate
        hist_x = np.histogram(lidar_points[:, 0], bins=NUM_HISTOGRAM_BINS)[0]
        hist_y = np.histogram(lidar_points[:, 1], bins=NUM_HISTOGRAM_BINS)[0]
        hist_z = np.histogram(lidar_points[:, 2], bins=NUM_HISTOGRAM_BINS)[0]
        
        # Normalize histograms
        hist_x = hist_x / (np.sum(hist_x) + 1e-10)
        hist_y = hist_y / (np.sum(hist_y) + 1e-10)
        hist_z = hist_z / (np.sum(hist_z) + 1e-10)
        
        lidar_features = np.concatenate([
            mean_features,
            std_features,
            percentile_features,
            hist_x,
            hist_y,
            hist_z
        ])

    # Camera features
    gray = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # Basic statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    percentiles = np.percentile(gray, [25, 50, 75])
    
    # Blur detection
    blur_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Edge density
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    # Intensity histogram
    intensity_hist = np.histogram(gray.flatten(), bins=CAMERA_HISTOGRAM_BINS)[0]
    intensity_hist = intensity_hist / (np.sum(intensity_hist) + 1e-10)  # Normalize
    
    # Edge pixels count
    edge_pixels = len(cv2.findNonZero(edges)) if cv2.findNonZero(edges) is not None else 0
    edge_pixels_normalized = edge_pixels / (edges.shape[0] * edges.shape[1])
    
    camera_features = np.concatenate([
        [mean_intensity],
        [std_intensity],
        percentiles,
        [blur_measure],
        [edge_density],
        intensity_hist,
        [edge_pixels_normalized]
    ])
    
    # Combine all features
    all_features = np.concatenate([lidar_features, camera_features])
    return all_features

# Function to detect spoofing (Updated)
def detect_spoofing(features):
    try:
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0][predicted.item()].item()
        return predicted.item(), confidence
    except Exception as e:
        print(f"Error in spoofing detection: {str(e)}")
        return -1, 0

# Main loop update
try:
    features_buffer = []  # Changed from deque to list for easier numpy operations
    running = True
    
    while running:
        running = handle_input()
        
        # [Previous rendering code remains the same]
        
        # Spoofing detection with fixed feature handling
        if detection_mode and time.time() - last_detection_time > detection_interval:
            if lidar_data is not None and camera_image is not None:
                features = extract_features(lidar_data, camera_image)
                
                # Single frame detection instead of buffer
                detected_spoof, confidence = detect_spoofing(features)
                
                if detected_spoof != -1:
                    spoof_types = ["No spoofing", "LiDAR spoofing", "Camera spoofing", "Both LiDAR and Camera spoofing"]
                    print(f"Detected: {spoof_types[detected_spoof]} (Confidence: {confidence:.2f})")
                    
                    if detected_spoof != 0 and confidence > 0.7:  # Added confidence threshold
                        send_sms_alert(spoof_types[detected_spoof])
                
                last_detection_time = time.time()
        
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
