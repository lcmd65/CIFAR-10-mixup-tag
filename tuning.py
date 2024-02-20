import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from optuna import create_study, Trial
import torch.nn.functional as F

# Define your model architecture here (replace with your actual model)


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


def objective_function(trial: Trial):
    # Suggest learning rate using Optuna's loguniform distribution
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    # Define your dataset and transformations
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=data_transform)
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=data_transform)

    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define your model, optimizer, and loss function
    model = Network()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        running_loss = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy on training set
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_accuracy = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)

        # Calculate and print validation accuracy after each epoch
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            val_accuracy = 100 * correct / total
            print(
                f"Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")

    # Return the final validation accuracy
    return val_accuracy

    
if __name__ == "__main__":
    # Create an Optuna study and optimize
    study = create_study(direction="maximize")  # Maximize accuracy
    study.optimize(objective_function, n_trials=5)

    # Get the best trial and its learning rate
    best_trial = study.best_trial
    best_learning_rate = best_trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    # Print results
    print(f"Best learning rate: {best_learning_rate}")
    print(f"Best validation accuracy: {best_trial.value:.2f}")