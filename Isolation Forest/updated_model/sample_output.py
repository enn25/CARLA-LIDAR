import numpy as np

# Load the data and labels from the saved files
data = np.load('merged_data.npy')
labels = np.load('merged_labels.npy')

# Unique labels present in the dataset
unique_labels = np.unique(labels)

# Create a dictionary to hold filtered data
filtered_data = {label: [] for label in unique_labels}

# Collect up to 2 samples for each label
for i in range(len(labels)):
    label = labels[i]
    if len(filtered_data[label]) < 2:  # Keep only 2 samples for each label
        filtered_data[label].append(data[i])

# Display the filtered samples
for label, samples in filtered_data.items():
    print(f"Label {label}:")
    for idx, sample in enumerate(samples):
        print(f" Sample {idx + 1}: {sample}")

