
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Reproducibility
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load CIFAR-10
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
    batch_size=128,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

# Create ResNet-18 adapted for CIFAR-10
model = models.resnet18(weights=None)

# Replace ImageNet's large initial convolution
model.conv1 = nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)

# Remove the initial max pooling
model.maxpool = nn.Identity()

# Change output layer to 10 classes
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)

print("ResNet-18 model created successfully.")

# Training configuration
loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0005
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10
)

epochs = 10

# Training
for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Training Loss: {average_loss:.4f}"
    )

# Evaluation
model.eval()

confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Build confusion matrix
        for actual, predicted in zip(
            labels.cpu(), predictions.cpu()
        ):
            confusion_matrix[actual, predicted] += 1

accuracy = 100 * correct / total

# Calculate precision, recall and F1-score
class_precision = []
class_recall = []
class_f1 = []

for i in range(10):

    true_positive = confusion_matrix[i, i].item()
    false_positive = confusion_matrix[:, i].sum().item() - true_positive
    false_negative = confusion_matrix[i, :].sum().item() - true_positive

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0 else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    class_precision.append(precision)
    class_recall.append(recall)
    class_f1.append(f1)

print("\nResNet-18 Evaluation Results")
print("----------------------------")
print(f"Correct predictions: {correct}")
print(f"Total test images: {total}")
print(f"Test accuracy: {accuracy:.2f}%")

print("\nPer-class metrics:")

for i in range(10):
    print(
        f"{classes[i]} | "
        f"Precision: {class_precision[i] * 100:.2f}% | "
        f"Recall: {class_recall[i] * 100:.2f}% | "
        f"F1: {class_f1[i] * 100:.2f}%"
    )

print("\nMacro-average metrics:")
print(f"Precision: {sum(class_precision) / 10 * 100:.2f}%")
print(f"Recall: {sum(class_recall) / 10 * 100:.2f}%")
print(f"F1-score: {sum(class_f1) / 10 * 100:.2f}%")

print("\nConfusion Matrix")
print("Rows = Actual classes; Columns = Predicted classes")
print("Class order:", classes)
print(confusion_matrix.numpy())

# Save model
torch.save(model.state_dict(), "cifar10_resnet18.pth")

print("\nModel saved as cifar10_resnet18.pth")
print("Training and evaluation completed.")
