
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("CIFAR-10 ADVANCED MODEL EVALUATION")
print("Using device:", device)
print("=" * 60)

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 64
DATA_DIR = "./data"


# --------------------------------------------------
# 2. Define the original CIFAR-10 CNN
# --------------------------------------------------

class CIFAR10CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --------------------------------------------------
# 3. Define the original CIFAR-10 ResNet-18
# --------------------------------------------------

def create_resnet18():

    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


# --------------------------------------------------
# 4. Define the fine-tuned ResNet-50
# --------------------------------------------------

def create_resnet50():

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


# --------------------------------------------------
# 5. Load saved model checkpoints
# --------------------------------------------------

def load_checkpoint(model, checkpoint_path):

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print("Loaded:", checkpoint_path)

    return model


# --------------------------------------------------
# 6. Prepare CIFAR-10 test datasets
# --------------------------------------------------

cifar_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

resnet50_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

cnn_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=cifar_test_transform
)

resnet18_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=cifar_test_transform
)

resnet50_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=resnet50_test_transform
)

cnn_loader = DataLoader(
    cnn_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

resnet18_loader = DataLoader(
    resnet18_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

resnet50_loader = DataLoader(
    resnet50_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nCIFAR-10 test datasets prepared.")
print("Number of test images:", len(cnn_dataset))


# --------------------------------------------------
# 7. Load all three trained models
# --------------------------------------------------

print("\nLoading trained models...")

cnn_model = load_checkpoint(
    CIFAR10CNN(),
    "cifar10_model.pth"
)

resnet18_model = load_checkpoint(
    create_resnet18(),
    "cifar10_resnet18.pth"
)

resnet50_model = load_checkpoint(
    create_resnet50(),
    "cifar10_resnet50_finetuned.pth"
)

models_to_evaluate = [
    ("CNN", cnn_model, cnn_loader),
    ("ResNet-18", resnet18_model, resnet18_loader),
    ("ResNet-50 Fine-tuned", resnet50_model, resnet50_loader)
]


# --------------------------------------------------
# 8. Evaluation function
# --------------------------------------------------

def evaluate_model(model, loader, model_name):

    all_predictions = []
    all_labels = []

    print(f"\nEvaluating {model_name}...")

    model.eval()

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    accuracy = accuracy_score(y_true, y_pred)

    print(
        f"{model_name} accuracy: {accuracy * 100:.2f}%"
    )

    return y_true, y_pred, accuracy


# --------------------------------------------------
# 9. Evaluate all three models
# --------------------------------------------------

results = {}

for name, model, loader in models_to_evaluate:

    y_true, y_pred, accuracy = evaluate_model(
        model,
        loader,
        name
    )

    results[name] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred
        )
    }


# --------------------------------------------------
# 10. Generate classification reports
# --------------------------------------------------

print("\nGenerating classification reports...")

with open(
    "cifar10_advanced_classification_reports.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("CIFAR-10 ADVANCED CLASSIFICATION REPORTS\n")
    file.write("=" * 65 + "\n")

    for name, result in results.items():

        report = classification_report(
            result["y_true"],
            result["y_pred"],
            target_names=classes,
            digits=4,
            zero_division=0
        )

        file.write(f"\nMODEL: {name}\n")
        file.write("-" * 65 + "\n")

        file.write(
            f"Accuracy: {result['accuracy'] * 100:.2f}%\n\n"
        )

        file.write(report)
        file.write("\n\n")

print("Classification reports saved.")


# --------------------------------------------------
# 11. Generate three confusion matrices
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, (name, result) in zip(axes, results.items()):

    matrix = result["confusion_matrix"]

    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(name)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    for i in range(10):
        for j in range(10):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=6
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()

plt.savefig(
    "cifar10_three_model_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrices saved.")


# --------------------------------------------------
# 12. Calculate per-class accuracy
# --------------------------------------------------

per_class_results = {}

for name, result in results.items():

    matrix = result["confusion_matrix"]

    class_accuracy = (
        np.diag(matrix) / matrix.sum(axis=1)
    ) * 100

    per_class_results[name] = class_accuracy


# Generate grouped per-class accuracy chart

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))

for index, (name, accuracy) in enumerate(
    per_class_results.items()
):

    ax.bar(
        x + (index - 1) * width,
        accuracy,
        width,
        label=name
    )

ax.set_title("CIFAR-10 Per-Class Accuracy Comparison")
ax.set_xlabel("CIFAR-10 Classes")
ax.set_ylabel("Accuracy (%)")

ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=35, ha="right")

ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "cifar10_three_model_per_class_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Per-class accuracy chart saved.")


# --------------------------------------------------
# 13. Generate overall accuracy comparison
# --------------------------------------------------

model_names = list(results.keys())

accuracies = [
    results[name]["accuracy"] * 100
    for name in model_names
]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(model_names, accuracies)

ax.set_title("CIFAR-10 Model Accuracy Comparison")
ax.set_ylabel("Test Accuracy (%)")
ax.set_ylim(0, 100)

ax.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, accuracies):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "cifar10_three_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Overall accuracy chart saved.")


# --------------------------------------------------
# 14. Generate prediction examples
# --------------------------------------------------

# Select the same 12 test images for all models.
rng = np.random.default_rng(42)

sample_indices = rng.choice(
    len(cnn_dataset),
    size=12,
    replace=False
)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

raw_images = cnn_dataset.data
raw_labels = cnn_dataset.targets

for position, (ax, sample_index) in enumerate(
    zip(axes.flat, sample_indices)
):

    image = raw_images[sample_index]
    actual_label = raw_labels[sample_index]

    ax.imshow(image)

    prediction_text = []

    for name, result in results.items():

        prediction = result["y_pred"][sample_index]

        status = (
            "Correct"
            if prediction == actual_label
            else "Incorrect"
        )

        prediction_text.append(
            f"{name}: {classes[prediction]} ({status})"
        )

    ax.set_title(
        f"Actual: {classes[actual_label]}\n"
        + "\n".join(prediction_text),
        fontsize=8
    )

    ax.axis("off")

plt.suptitle(
    "Prediction Examples: CNN vs ResNet-18 vs Fine-tuned ResNet-50",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "cifar10_three_model_prediction_examples.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Prediction examples saved.")


# --------------------------------------------------
# 15. Generate consolidated research summary
# --------------------------------------------------

summary_path = "cifar10_advanced_research_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("CIFAR-10 ADVANCED MODEL EVALUATION\n")
    file.write("=" * 65 + "\n\n")

    file.write("DATASET INFORMATION\n")
    file.write("-" * 65 + "\n")
    file.write("Dataset: CIFAR-10\n")
    file.write("Test images: 10,000\n")
    file.write("Number of classes: 10\n")
    file.write("Evaluation device: " + str(device) + "\n\n")

    file.write("OVERALL MODEL PERFORMANCE\n")
    file.write("-" * 65 + "\n")

    for name, result in results.items():

        file.write(
            f"{name}: {result['accuracy'] * 100:.2f}%\n"
        )

    file.write("\nPER-CLASS PERFORMANCE\n")
    file.write("-" * 65 + "\n")

    for name, class_accuracy in per_class_results.items():

        best_index = int(np.argmax(class_accuracy))
        worst_index = int(np.argmin(class_accuracy))

        file.write(f"\n{name}\n")

        file.write(
            f"Best class: {classes[best_index]} "
            f"({class_accuracy[best_index]:.2f}%)\n"
        )

        file.write(
            f"Weakest class: {classes[worst_index]} "
            f"({class_accuracy[worst_index]:.2f}%)\n"
        )

        for class_name, value in zip(classes, class_accuracy):

            file.write(
                f"{class_name}: {value:.2f}%\n"
            )

    best_model = max(
        results,
        key=lambda name: results[name]["accuracy"]
    )

    worst_model = min(
        results,
        key=lambda name: results[name]["accuracy"]
    )

    improvement = (
        results[best_model]["accuracy"]
        - results[worst_model]["accuracy"]
    ) * 100

    file.write("\nMODEL COMPARISON\n")
    file.write("-" * 65 + "\n")

    file.write(f"Highest accuracy: {best_model}\n")
    file.write(f"Lowest accuracy: {worst_model}\n")

    file.write(
        f"Accuracy difference: {improvement:.2f} percentage points\n"
    )

    file.write("\nMETHODOLOGICAL NOTE\n")
    file.write("-" * 65 + "\n")

    file.write(
        "The CNN and ResNet-18 were trained from scratch using "
        "32x32 CIFAR-10 images. ResNet-50 used ImageNet "
        "pretraining, 224x224 image resizing, and partial "
        "fine-tuning. Therefore, this comparison evaluates "
        "the complete training pipelines and does not isolate "
        "architecture as the only experimental variable.\n"
    )

print("Research summary saved.")


# --------------------------------------------------
# 16. Final results
# --------------------------------------------------

print("\n" + "=" * 60)
print("ADVANCED EVALUATION COMPLETED")
print("=" * 60)

for name, result in results.items():

    print(
        f"{name}: {result['accuracy'] * 100:.2f}%"
    )

print("\nGenerated research files:")
print("1. cifar10_advanced_classification_reports.txt")
print("2. cifar10_three_model_confusion_matrices.png")
print("3. cifar10_three_model_per_class_accuracy.png")
print("4. cifar10_three_model_comparison.png")
print("5. cifar10_three_model_prediction_examples.png")
print("6. cifar10_advanced_research_summary.txt")

print("\nEvaluation completed successfully.")
