
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import numpy as np

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

# Test data transformation
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load CIFAR-10 test dataset
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

# Recreate the trained ResNet-18 architecture
model = models.resnet18(weights=None)

model.conv1 = nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)

model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)

# Load saved model weights
model.load_state_dict(
    torch.load(
        "cifar10_resnet18.pth",
        map_location=device
    )
)

model.eval()

print("ResNet-18 model loaded successfully.")

# Collect predictions
all_predictions = []
all_labels = []
all_images = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

        all_images.extend(images.cpu())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Overall accuracy
accuracy = 100 * np.mean(all_predictions == all_labels)

print("\nResearch Evaluation Results")
print("---------------------------")
print(f"Test images: {len(all_labels)}")
print(f"Correct predictions: {(all_predictions == all_labels).sum()}")
print(f"Accuracy: {accuracy:.2f}%")

# Classification report
report = classification_report(
    all_labels,
    all_predictions,
    target_names=classes,
    digits=4,
    zero_division=0
)

print("\nClassification Report")
print("---------------------")
print(report)

with open("classification_report.txt", "w") as file:
    file.write("CIFAR-10 ResNet-18 Classification Report\n\n")
    file.write(report)
    file.write(f"\nOverall accuracy: {accuracy:.2f}%\n")

# Confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(10))
)

fig, ax = plt.subplots(figsize=(11, 9))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    colorbar=False
)

plt.title("CIFAR-10 ResNet-18 Confusion Matrix")
plt.tight_layout()
plt.savefig("cifar10_confusion_matrix.png", dpi=200)
plt.close()

print("Confusion matrix saved.")

# Display sample correct and incorrect predictions
correct_indices = np.where(all_predictions == all_labels)[0]
incorrect_indices = np.where(all_predictions != all_labels)[0]

selected_correct = correct_indices[:5]
selected_incorrect = incorrect_indices[:5]

selected_indices = list(selected_correct) + list(selected_incorrect)

fig, axes = plt.subplots(2, 5, figsize=(15, 7))

for position, index in enumerate(selected_indices):

    row = position // 5
    col = position % 5

    image = all_images[index]

    # Reverse normalization for display
    image = image * torch.tensor(std).view(3, 1, 1)
    image = image + torch.tensor(mean).view(3, 1, 1)

    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()

    actual = classes[all_labels[index]]
    predicted = classes[all_predictions[index]]

    axes[row, col].imshow(image)

    if all_predictions[index] == all_labels[index]:
        title_color = "green"
    else:
        title_color = "red"

    axes[row, col].set_title(
        f"Actual: {actual}\nPredicted: {predicted}",
        color=title_color,
        fontsize=9
    )

    axes[row, col].axis("off")

fig.suptitle(
    "Correct Predictions (Top Row) and Incorrect Predictions (Bottom Row)",
    fontsize=13
)

plt.tight_layout()
plt.savefig("cifar10_prediction_examples.png", dpi=200)
plt.close()

print("Prediction examples saved.")

# Compare original CNN and ResNet-18
model_names = ["Original CNN", "ResNet-18"]
accuracies = [72.40, accuracy]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(model_names, accuracies)

ax.set_ylabel("Test Accuracy (%)")
ax.set_title("CIFAR-10 Model Accuracy Comparison")
ax.set_ylim(0, 100)

for bar, score in zip(bars, accuracies):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{score:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("cifar10_model_comparison.png", dpi=200)
plt.close()

print("Model comparison chart saved.")

print("\nGenerated research files:")
print("1. classification_report.txt")
print("2. cifar10_confusion_matrix.png")
print("3. cifar10_prediction_examples.png")
print("4. cifar10_model_comparison.png")

print("\nResearch evaluation completed successfully.")
