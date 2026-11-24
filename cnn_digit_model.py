
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Reproducibility
torch.manual_seed(42)

# 2. Select computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 3. Prepare MNIST dataset
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 4. Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

# 5. Define the CNN architecture
class CNNClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = CNNClassifier().to(device)

print("CNN model created successfully.")

# 6. Configure training
loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

# 7. Train the CNN
for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = loss_function(predictions, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Training Loss: {average_loss:.4f}"
    )

# 8. Evaluate the CNN
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)

        predicted_labels = predictions.argmax(dim=1)

        total += labels.size(0)

        correct += (predicted_labels == labels).sum().item()

accuracy = 100 * correct / total

print("\nCNN Evaluation Results")
print("----------------------")
print(f"Correct predictions: {correct}")
print(f"Total test images: {total}")
print(f"Test accuracy: {accuracy:.2f}%")

# 9. Save the trained CNN
torch.save(
    model.state_dict(),
    "cnn_mnist_model.pth"
)

print("\nCNN model saved as cnn_mnist_model.pth")
print("Training and evaluation completed.")
