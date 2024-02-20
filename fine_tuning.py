import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR100
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(
            in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*10*10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)

        return output

# Load pre-trained model
# Load the model's state dictionary
model = Network()
model.load_state_dict(torch.load('model/model.pth'))

# Access the fc1 layer attribute
num_ftrs = model.fc1.in_features
# Modify last layer for 100 classes in CIFAR-100
model.fc1 = nn.Linear(num_ftrs, 100)

# Define CIFAR-100 dataset and data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-100 dataset
full_train_dataset = CIFAR100(
    root='./dataset2', train=True, download=True, transform=transform)

# Split the dataset into training and validation sets (80% training, 20% validation)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, _ = random_split(full_train_dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load test dataset
test_dataset = CIFAR100(root='./dataset2', train=False,
                        download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Function to save the model


def saveModel():
    path = "./model/model_2.pth"
    torch.save(model.state_dict(), path)
    
def fine_tuning(num_epochs):
    # Fine-tuning loop
    for epoch in range(num_epochs):
        # Training phase
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}")

        # Validation phase (compute test accuracy)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    fine_tuning(10)
    saveModel()