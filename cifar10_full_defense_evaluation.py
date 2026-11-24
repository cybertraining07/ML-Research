
import os
import csv
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

torch.manual_seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPSILON = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 10
BATCH_SIZE = 128
PGD_IMAGES_PER_CLASS = 200

OUTPUT_DIR = "full_defense_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Device:", device)
print("Clean and FGSM evaluation: full 10,000-image test set")
print("PGD evaluation: 200 images per class")

# --------------------------------------------------
# 2. CIFAR-10 normalization
# --------------------------------------------------

mean = torch.tensor(
    [0.4914, 0.4822, 0.4465]
).view(1, 3, 1, 1).to(device)

std = torch.tensor(
    [0.2470, 0.2435, 0.2616]
).view(1, 3, 1, 1).to(device)


def normalize(images):
    return (images - mean) / std


# --------------------------------------------------
# 3. CNN architecture
# --------------------------------------------------

class CIFAR10CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
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
# 4. Model checkpoints
# --------------------------------------------------

checkpoints = {
    "Baseline CNN":
        "cifar10_model.pth",

    "FGSM Defense":
        "adversarial_defense_results/cifar10_fgsm_defended.pth",

    "PGD Defense":
        "adversarial_defense_results/cifar10_pgd_defended.pth",

    "Mixed FGSM Defense":
        "mixed_adversarial_results/cifar10_mixed_fgsm_defended.pth"
}


def load_checkpoint(model, path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    state = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    # Support either a direct state dictionary
    # or a dictionary containing model_state_dict.
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    return model


# --------------------------------------------------
# 5. Load CIFAR-10 test data
# --------------------------------------------------

test_transform = transforms.ToTensor()

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

full_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Select exactly 200 images from each class.
class_indices = {i: [] for i in range(10)}

for index, label in enumerate(test_dataset.targets):
    if len(class_indices[label]) < PGD_IMAGES_PER_CLASS:
        class_indices[label].append(index)

pgd_indices = []

for label in range(10):
    pgd_indices.extend(class_indices[label])

pgd_dataset = Subset(test_dataset, pgd_indices)

pgd_loader = DataLoader(
    pgd_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Full test images:", len(test_dataset))
print("Balanced PGD images:", len(pgd_dataset))


# --------------------------------------------------
# 6. Clean evaluation
# --------------------------------------------------

def evaluate_clean(model, loader):

    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(normalize(images))
            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100 * correct / total


# --------------------------------------------------
# 7. FGSM evaluation
# --------------------------------------------------


def evaluate_pgd(model, loader):

    total = 0
    clean_correct = 0
    adversarial_correct = 0
    successful_attacks = 0

    model.eval()

    torch.manual_seed(123)

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        original_images = images.detach()

        with torch.no_grad():

            clean_outputs = model(normalize(original_images))
            clean_predictions = clean_outputs.argmax(dim=1)

            initially_correct = clean_predictions == labels
            clean_correct += initially_correct.sum().item()

        adversarial_images = (
            original_images
            + torch.empty_like(original_images).uniform_(
                -EPSILON, EPSILON
            )
        ).clamp(0, 1).detach()

        for step in range(PGD_STEPS):

            adversarial_images.requires_grad_(True)

            outputs = model(normalize(adversarial_images))
            loss = nn.CrossEntropyLoss()(outputs, labels)

            model.zero_grad()
            loss.backward()

            with torch.no_grad():

                adversarial_images = (
                    adversarial_images
                    + PGD_ALPHA * adversarial_images.grad.sign()
                )

                perturbation = (
                    adversarial_images - original_images
                ).clamp(-EPSILON, EPSILON)

                adversarial_images = (
                    original_images + perturbation
                ).clamp(0, 1).detach()

        with torch.no_grad():

            final_outputs = model(normalize(adversarial_images))
            predictions = final_outputs.argmax(dim=1)

            adversarial_correct += (
                predictions == labels
            ).sum().item()

            successful_attacks += (
                initially_correct & (predictions != labels)
            ).sum().item()

        total += labels.size(0)

    accuracy = 100 * adversarial_correct / total

    attack_success = (
        100 * successful_attacks / clean_correct
        if clean_correct else 0.0
    )

    return accuracy, attack_success

# --------------------------------------------------
# 8. PGD evaluation
# --------------------------------------------------


def evaluate_fgsm(model, loader):

    total = 0
    clean_correct = 0
    adversarial_correct = 0
    successful_attacks = 0

    model.eval()

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)
        images.requires_grad_(True)

        outputs = model(normalize(images))
        clean_predictions = outputs.argmax(dim=1)

        initially_correct = clean_predictions == labels
        clean_correct += initially_correct.sum().item()

        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()

        adversarial_images = (
            images + EPSILON * images.grad.sign()
        ).clamp(0, 1).detach()

        with torch.no_grad():

            adv_outputs = model(normalize(adversarial_images))
            adv_predictions = adv_outputs.argmax(dim=1)

            adversarial_correct += (
                adv_predictions == labels
            ).sum().item()

            successful_attacks += (
                initially_correct & (adv_predictions != labels)
            ).sum().item()

        total += labels.size(0)

    accuracy = 100 * adversarial_correct / total

    attack_success = (
        100 * successful_attacks / clean_correct
        if clean_correct else 0.0
    )

    return accuracy, attack_success

# --------------------------------------------------
# 9. Evaluate all four models
# --------------------------------------------------

results = []

for model_name, checkpoint_path in checkpoints.items():

    print("\n" + "=" * 55)
    print("Evaluating:", model_name)
    print("=" * 55)

    model = CIFAR10CNN().to(device)
    model = load_checkpoint(model, checkpoint_path)

    print("Checkpoint loaded successfully.")

    print("Running clean evaluation...")
    clean_accuracy = evaluate_clean(model, full_loader)

    print("Running FGSM evaluation...")
    fgsm_accuracy, fgsm_asr = evaluate_fgsm(
        model, full_loader
    )

    print("Running PGD evaluation...")
    pgd_accuracy, pgd_asr = evaluate_pgd(
        model, pgd_loader
    )

    results.append({
        "Model": model_name,
        "Clean Accuracy (%)": round(clean_accuracy, 2),
        "FGSM Accuracy (%)": round(fgsm_accuracy, 2),
        "FGSM Attack Success Rate (%)": round(fgsm_asr, 2),
        "PGD Accuracy (%)": round(pgd_accuracy, 2),
        "PGD Attack Success Rate (%)": round(pgd_asr, 2)
    })

    print("Clean accuracy:", round(clean_accuracy, 2), "%")
    print("FGSM accuracy:", round(fgsm_accuracy, 2), "%")
    print("FGSM attack success:", round(fgsm_asr, 2), "%")
    print("PGD accuracy:", round(pgd_accuracy, 2), "%")
    print("PGD attack success:", round(pgd_asr, 2), "%")

    del model


# --------------------------------------------------
# 10. Save CSV results
# --------------------------------------------------

csv_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_full_defense_results.csv"
)

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# 11. Generate comparison chart
# --------------------------------------------------

models = [row["Model"] for row in results]

clean_values = [
    row["Clean Accuracy (%)"] for row in results
]

fgsm_values = [
    row["FGSM Accuracy (%)"] for row in results
]

pgd_values = [
    row["PGD Accuracy (%)"] for row in results
]

x = range(len(models))
width = 0.25

plt.figure(figsize=(12, 7))

plt.bar(
    [i - width for i in x],
    clean_values,
    width,
    label="Clean"
)

plt.bar(
    list(x),
    fgsm_values,
    width,
    label="FGSM"
)

plt.bar(
    [i + width for i in x],
    pgd_values,
    width,
    label="PGD"
)

plt.xticks(list(x), models, rotation=15)
plt.ylabel("Accuracy (%)")
plt.title("CIFAR-10 Adversarial Defense Comparison")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

chart_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_full_defense_comparison.png"
)

plt.savefig(chart_path, dpi=300)
plt.close()


# --------------------------------------------------
# 12. Generate research report
# --------------------------------------------------

report_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_full_defense_report.txt"
)

with open(report_path, "w") as file:

    file.write(
        "CIFAR-10 ADVERSARIAL DEFENSE EVALUATION\n"
    )

    file.write("=" * 55 + "\n\n")

    file.write("Evaluation configuration:\n")
    file.write("Dataset: CIFAR-10 test set\n")
    file.write("Clean evaluation images: 10,000\n")
    file.write("FGSM evaluation images: 10,000\n")
    file.write("PGD evaluation images: 2,000\n")
    file.write("PGD sampling: 200 images per class\n")
    file.write(f"FGSM epsilon: {EPSILON:.6f}\n")
    file.write(f"PGD epsilon: {EPSILON:.6f}\n")
    file.write(f"PGD step size: {PGD_ALPHA:.6f}\n")
    file.write(f"PGD steps: {PGD_STEPS}\n\n")

    file.write("Results:\n\n")

    for row in results:

        for key, value in row.items():
            file.write(f"{key}: {value}\n")

        file.write("\n")

    file.write(
        "Limitations:\n"
        "- PGD results use a balanced 2,000-image subset.\n"
        "- Clean and FGSM results use the full 10,000-image test set.\n"
        "- Results represent the tested attacks and settings only.\n"
        "- Empirical robustness does not establish universal robustness.\n"
        "- The evaluation uses one training run per model.\n"
    )


# --------------------------------------------------
# 13. Final summary
# --------------------------------------------------

print("\n" + "=" * 55)
print("FULL DEFENSE EVALUATION COMPLETED")
print("=" * 55)

for row in results:
    print(row)

print("\nGenerated files:")
print(csv_path)
print(chart_path)
print(report_path)
