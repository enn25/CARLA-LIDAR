import torch
import torch.nn as nn
import numpy as np

# Define the SpoofClassifier model (make sure this matches the architecture used during training)
class SpoofClassifier(nn.Module):
    def __init__(self, input_size):
        super(SpoofClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
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

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

# Load the test data
test_data = np.load('test_datav4.npy')
test_labels = np.load('test_labelsv4.npy')

# Convert numpy arrays to PyTorch tensors
X_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)

# Get the input size from the test data
input_size = X_test.shape[1]

# Initialize the model
model = SpoofClassifier(input_size)

# Load the saved model state
model.load_state_dict(torch.load('best_model.pth'))

# Set the model to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)

# Calculate accuracy
accuracy = calculate_accuracy(test_outputs, y_test)

# Calculate loss
criterion = nn.CrossEntropyLoss()
test_loss = criterion(test_outputs, y_test)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# If you want to see predictions for individual samples:
for i in range(min(4646, len(X_test))):  # Print first 10 predictions
    print(f"Sample {i+1}: True Label: {y_test[i].item()}, Predicted: {predicted[i].item()}")

# If you want to calculate per-class accuracy:
class_correct = list(0. for i in range(4))
class_total = list(0. for i in range(4))
with torch.no_grad():
    for i, (images, labels) in enumerate(zip(X_test, y_test)):
        outputs = model(images.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        class_correct[labels] += c.item()
        class_total[labels] += 1

for i in range(4):
    print(f'Accuracy of class {i} : {100 * class_correct[i] / class_total[i]:.2f}%')
