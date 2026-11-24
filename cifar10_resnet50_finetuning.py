
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

# ==========================================
# CIFAR-10 RESNET-50 FINE-TUNING
# Experiment 2
# ==========================================

BATCH_SIZE = 32
EPOCHS = 2

device = torch.device("cpu")
print("Using device:", device)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

print("\nLoading CIFAR-10 datasets...")

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

# Load the trained Experiment 1 model
print("\nLoading previously trained ResNet-50...")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)

checkpoint = torch.load(
    "cifar10_resnet50_feature_extractor.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)

# Freeze all parameters initially
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze the final residual block
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# Unfreeze the classifier
for parameter in model.fc.parameters():
    parameter.requires_grad = True

model = model.to(device)

print("ResNet-50 checkpoint loaded.")
print("Layer4 and classification layer are trainable.")
print("Earlier feature layers remain frozen.")

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 0.0001},
    {"params": model.fc.parameters(), "lr": 0.0005}
])

# Fine-tuning
print("\nStarting ResNet-50 fine-tuning...\n")

for epoch in range(EPOCHS):

    model.train()

    # Keep frozen feature layers and their BatchNorm
    # statistics fixed during fine-tuning.
    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

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
    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Training Accuracy: {accuracy:.2f}%"
    )

# Evaluation
print("\nEvaluating fine-tuned ResNet-50...")

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

print("\n========== FINE-TUNING RESULTS ==========")
print("Model: ResNet-50")
print("Experiment: Partial Fine-Tuning")
print("Trainable layers: Layer4 and FC")
print("Test images:", total)
print(f"Test accuracy: {test_accuracy:.2f}%")

torch.save(
    model.state_dict(),
    "cifar10_resnet50_finetuned.pth"
)

print("\nModel saved: cifar10_resnet50_finetuned.pth")
print("Fine-tuning experiment completed.")
