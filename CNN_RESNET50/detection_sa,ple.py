import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pygame
import time

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((640, 480))
font = pygame.font.Font(None, 36)

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_camera_spoofing_model.pth'))
model = model.to(device)
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Detection Function
def detect_spoof(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)
        return prediction.item()

# Pygame Alert Function
def show_alert_in_pygame(is_spoofed):
    screen.fill((0, 0, 0))  # Clear screen with black background
    if is_spoofed == 1:
        text = font.render("Alert: Spoof detected!", True, (255, 0, 0))  # Red text for spoof
    else:
        text = font.render("Status: Normal", True, (0, 255, 0))  # Green text for normal
    screen.blit(text, (50, 220))
    pygame.display.flip()  # Update the screen display

# Real-Time Detection Loop
image_folder = 'lidar_dataset/images/spoofed_or_normal'  # Update folder path as needed
image_files = os.listdir(image_folder)
index = 0

# Main Pygame Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Check if there are images to detect
    if index < len(image_files):
        image_path = os.path.join(image_folder, image_files[index])
        is_spoofed = detect_spoof(image_path)
        show_alert_in_pygame(is_spoofed)
        index += 1
        time.sleep(0.5)  # Delay to simulate real-time processing
    else:
        index = 0  # Reset to start of image list for continuous loop

# Quit Pygame
pygame.quit()

