# LiDAR-Camera Cross-Validation System for Autonomous Vehicle Spoofing Defense

This project implements a robust security framework that detects, mitigates, and corrects sensor spoofing attacks in autonomous vehicles by leveraging cross-validation between LiDAR and camera data.

## Disclaimer

**Important:** The filenames and exact script parameters mentioned in this README may differ from the actual files in the repository due to ongoing updates and improvements. Please review all scripts thoroughly to understand their functionality and required parameters before use.

## Overview

Autonomous vehicles rely heavily on LiDAR and camera sensors for environmental perception. This project addresses the critical security vulnerability of sensor spoofing attacks by implementing:

1. Cross-sensor validation between LiDAR and camera data
2. Two detection approaches:
   - Isolation Forest for anomaly detection
   - ResNet-50 deep learning model for advanced spoof detection
3. Real-time detection and correction mechanisms in CARLA simulator

## Installation Instructions

### Prerequisites

- Ubuntu 20.04 LTS or compatible version
- Python 3.7+
- CUDA-compatible GPU (for CNN approach)
- 50+ GB disk space

### Step 1: Install CARLA Simulator

1. Download and install the CARLA simulator from the [official CARLA website](https://carla.org/download.html)
2. Follow the installation instructions for your specific Ubuntu version
3. Verify installation by running the simulator:
   ```
   ./CarlaUE4.sh
   ```

### Step 2: Set Up Project Environment

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/lidar-camera-spoofing-defense.git
   cd lidar-camera-spoofing-defense
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage Guide

### Data Collection

1. Start the CARLA simulator:
   ```
   ./CarlaUE4.sh
   ```

2. Run the automatic control script to move the vehicle in the simulator:
   ```
   python auto_control.py
   ```

3. Run the data collection script for each town in the simulator:
   ```
   python data_collection.py --output_dir ./dataset --town Town01
   python data_collection.py --output_dir ./dataset --town Town02
   python data_collection.py --output_dir ./dataset --town Town03
   ```
   This will collect synchronized LiDAR and camera data with appropriate labels for model training.

4. For the Isolation Forest approach, collect data from multiple towns to decrease overfitting/underfitting:
   ```
   python merge_datasets.py --input_dirs ./dataset/Town01 ./dataset/Town02 ./dataset/Town03 --output_dir ./dataset/merged
   ```

> **Note**: The collected dataset exceeds GitHub's 25MB file size limit and is not included in this repository. Please generate your own dataset using the provided scripts.

### Model Training

#### Isolation Forest Approach

```
python train_isolation_forest.py --data_path ./dataset/merged --output_model ./models/isolation_forest.pkl
```

#### ResNet-50 CNN Approach

```
python train_resnet.py --data_path ./dataset --epochs 50 --batch_size 32 --output_model ./models/resnet50_model.h5
```

### Testing and Visualization

1. Start the CARLA simulator

2. Run the test script to evaluate the trained models:
   ```
   python test_model.py --model_path ./models/resnet50_model.h5 --model_type resnet
   ```
   or
   ```
   python test_model.py --model_path ./models/isolation_forest.pkl --model_type isolation
   ```

3. For real-time visualization within the simulator:
   ```
   python visualize_results.py --model_path ./models/resnet50_model.h5
   ```

## Project Structure

```
.
├── auto_control.py          # Script for automatic vehicle control in CARLA
├── data_collection.py       # Data collection script for training datasets
├── merge_datasets.py        # Script to merge data from multiple towns (for Isolation Forest)
├── models/                  # Directory for trained models (not included in repo)
├── requirements.txt         # Python dependencies
├── test_model.py            # Script for testing trained models
├── train_isolation_forest.py # Training script for Isolation Forest approach
├── train_resnet.py          # Training script for ResNet-50 approach
└── visualize_results.py     # Visualization script for real-time results
```

## Acknowledgments

- CARLA Simulator team for providing the platform for autonomous vehicle research
- Contributors and researchers in the field of sensor security for autonomous systems
