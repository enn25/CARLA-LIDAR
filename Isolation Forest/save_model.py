import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
data = np.load('spoof_classifier_data.npy')
labels = np.load('spoof_classifier_labels.npy')

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Get the input size from the data
input_size = X_train.shape[1]

# Define the SpoofClassifier model
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

# Initialize model, loss function, and optimizer
spoof_classifier = SpoofClassifier(input_size)
optimizer = torch.optim.Adam(spoof_classifier.parameters(), lr=0.0001, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

# Training loop
num_epochs = 100
batch_size = 64
best_val_loss = float('inf')
patience = 10
epochs_without_improvement = 0

for epoch in range(num_epochs):
    spoof_classifier.train()
    train_loss = 0.0
    train_accuracy = 0.0
    batches = 0
    
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = spoof_classifier(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_accuracy += calculate_accuracy(outputs, batch_y)
        batches += 1
    
    avg_train_loss = train_loss / batches
    avg_train_accuracy = train_accuracy / batches
    
    # Validation
    spoof_classifier.eval()
    with torch.no_grad():
        val_outputs = spoof_classifier(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_accuracy = calculate_accuracy(val_outputs, y_val)
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(spoof_classifier.state_dict(), 'best_model.pth')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after epoch {epoch + 1}')
            break

# Load the best model
spoof_classifier.load_state_dict(torch.load('best_model.pth'))

# Final evaluation on test set
spoof_classifier.eval()
with torch.no_grad():
    test_outputs = spoof_classifier(X_test)
    test_loss = criterion(test_outputs, y_test)
    test_accuracy = calculate_accuracy(test_outputs, y_test)

print(f'\nFinal Test Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
