import numpy as np
from sklearn.model_selection import train_test_split

# Load your dataset (replace with actual paths)
data = np.load('spoof_classifier_data.npy')
labels = np.load('spoof_classifier_labels.npy')

# Split the data into a training set (80%) and a testing set (20%)
# Stratify ensures that the class distribution in both train and test sets is similar
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Print the shapes of the splits to verify
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

# Save the test data to test separately (optional)
np.save('test_data.npy', X_test)
np.save('test_labels.npy', y_test)

# Now you can use the training set to train your model and the test set to evaluate it

