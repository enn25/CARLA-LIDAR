import os
import numpy as np

def merge_datasets(base_path):
    merged_data = []
    merged_labels = []

    for i in range(1, 5):  # For 4 towns
        town_folder = f'town{i}'
        data_path = os.path.join(base_path, town_folder, f'data{i}.npy')
        label_path = os.path.join(base_path, town_folder, f'label{i}.npy')
        
        # Load data and labels
        data = np.load(data_path)
        labels = np.load(label_path)
        
        # Add a column to identify the town
        town_column = np.full((data.shape[0], 1), i)
        data_with_town = np.hstack((data, town_column))
        labels_with_town = np.hstack((labels.reshape(-1, 1), town_column))
        
        merged_data.append(data_with_town)
        merged_labels.append(labels_with_town)
    
    # Concatenate all arrays
    final_data = np.vstack(merged_data)
    final_labels = np.vstack(merged_labels)
    
    # Save merged datasets
    np.save(os.path.join(base_path, 'merged_data.npy'), final_data)
    np.save(os.path.join(base_path, 'merged_labels.npy'), final_labels)
    
    print("Datasets merged successfully!")

# Example usage
base_path = '~/carla/PythonAPI/examples/chatg/updated model'
merge_datasets(base_path)
