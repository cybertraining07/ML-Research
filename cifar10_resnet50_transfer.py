
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

# ==========================================
# CIFAR-10 RESNET-50 TRANSFER LEARNING
# Experiment 1: Feature Extraction
# ==========================================

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

device = torch.device("cpu")
print("Using device:", device)

# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Training transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Testing transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load CIFAR-10 datasets
print("\nLoading CIFAR-10 dataset...")

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

# Load pretrained ResNet-50
print("\nLoading pretrained ResNet-50 weights...")

weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Freeze pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace final layer for CIFAR-10
number_of_features = model.fc.in_features
model.fc = nn.Linear(number_of_features, 10)

model = model.to(device)

print("ResNet-50 model loaded successfully.")
print("Pretrained feature layers frozen.")
print("New classification layer: 10 classes")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

# Training
print("\nStarting transfer learning...\n")

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Training Accuracy: {epoch_accuracy:.2f}%"
    )

# Evaluation
print("\nEvaluating model on CIFAR-10 test data...")

model.eval()
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print("\n========== FINAL RESULTS ==========")
print("Model: ResNet-50 Transfer Learning")
print("Experiment: Frozen Feature Extraction")
print("Test images:", total)
print(f"Test accuracy: {test_accuracy:.2f}%")

# Save model
model_path = "cifar10_resnet50_feature_extractor.pth"

torch.save(model.state_dict(), model_path)

print("\nModel saved successfully:", model_path)
print("Transfer learning experiment completed.")
