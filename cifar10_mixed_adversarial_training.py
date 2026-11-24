
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torch.optim import Adam

# ==========================================
# 1. Configuration
# ==========================================

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
TRAINING_SAMPLES = 10000

EPSILON = 8 / 255
LEARNING_RATE = 0.0002

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = "mixed_adversarial_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASELINE_CHECKPOINT = "cifar10_model.pth"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MEAN = torch.tensor(
    [0.4914, 0.4822, 0.4465],
    device=DEVICE
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.2470, 0.2435, 0.2616],
    device=DEVICE
).view(1, 3, 1, 1)


# ==========================================
# 2. Original CNN architecture
# ==========================================

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
        return self.classifier(self.features(x))


def get_logits(model, images):
    normalized = (images - MEAN) / STD
    return model(normalized)


# ==========================================
# 3. FGSM attack
# ==========================================

def fgsm_attack(model, images, labels):

    original = images.detach()
    adversarial = original.clone().requires_grad_(True)

    logits = get_logits(model, adversarial)
    loss = F.cross_entropy(logits, labels)

    gradient = torch.autograd.grad(
        loss, adversarial
    )[0]

    adversarial = (
        original + EPSILON * gradient.sign()
    ).clamp(0, 1)

    return adversarial.detach()


# ==========================================
# 4. Load datasets
# ==========================================

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

generator = torch.Generator().manual_seed(SEED)

train_indices = torch.randperm(
    len(train_dataset),
    generator=generator
)[:TRAINING_SAMPLES].tolist()

train_subset = Subset(
    train_dataset,
    train_indices
)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

targets = np.array(test_dataset.targets)

test_indices = []

for class_id in range(10):
    class_indices = np.where(
        targets == class_id
    )[0]

    test_indices.extend(
        class_indices[:20].tolist()
    )

test_subset = Subset(
    test_dataset,
    test_indices
)

test_loader = DataLoader(
    test_subset,
    batch_size=50,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 5. Load original baseline model
# ==========================================

model = CIFAR10CNN().to(DEVICE)

checkpoint = torch.load(
    BASELINE_CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 6. Mixed adversarial training
# ==========================================

history = []

print("Device:", DEVICE)
print("Training samples:", TRAINING_SAMPLES)
print("Epochs:", EPOCHS)
print("Clean/adversarial ratio: 50/50")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    total_clean_correct = 0
    total_adv_correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Generate adversarial images without
        # changing model weights.
        model.eval()

        adversarial_images = fgsm_attack(
            model, images, labels
        )

        # Return to training mode.
        model.train()

        optimizer.zero_grad()

        # Clean predictions.
        clean_logits = get_logits(
            model, images
        )

        # Adversarial predictions.
        adv_logits = get_logits(
            model, adversarial_images
        )

        clean_loss = F.cross_entropy(
            clean_logits, labels
        )

        adversarial_loss = F.cross_entropy(
            adv_logits, labels
        )

        # Equal contribution from both losses.
        loss = (
            0.5 * clean_loss
            + 0.5 * adversarial_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        total_clean_correct += (
            clean_logits.argmax(1) == labels
        ).sum().item()

        total_adv_correct += (
            adv_logits.argmax(1) == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = total_loss / total
    clean_accuracy = 100 * total_clean_correct / total
    adversarial_accuracy = 100 * total_adv_correct / total

    history.append({
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Clean training accuracy: {clean_accuracy:.2f}% | "
        f"Adversarial training accuracy: "
        f"{adversarial_accuracy:.2f}%"
    )


# ==========================================
# 7. Save defended model
# ==========================================

checkpoint_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_mixed_fgsm_defended.pth"
)

torch.save(
    model.state_dict(),
    checkpoint_path
)


# ==========================================
# 8. Evaluate clean, FGSM and PGD
# ==========================================

def pgd_attack(
    model,
    images,
    labels,
    steps=10,
    alpha=2 / 255
):

    original = images.detach()

    adversarial = (
        original
        + torch.empty_like(original).uniform_(
            -EPSILON, EPSILON
        )
    ).clamp(0, 1)

    for _ in range(steps):

        adversarial = (
            adversarial.detach().requires_grad_(True)
        )

        logits = get_logits(model, adversarial)

        loss = F.cross_entropy(
            logits, labels
        )

        gradient = torch.autograd.grad(
            loss, adversarial
        )[0]

        adversarial = (
            adversarial.detach()
            + alpha * gradient.sign()
        )

        perturbation = torch.clamp(
            adversarial - original,
            -EPSILON,
            EPSILON
        )

        adversarial = (
            original + perturbation
        ).clamp(0, 1)

    return adversarial.detach()


def evaluate():

    model.eval()

    totals = {
        "Clean": 0,
        "FGSM": 0,
        "PGD": 0
    }

    initially_correct = 0
    attack_success = {
        "FGSM": 0,
        "PGD": 0
    }

    total = 0

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            clean_predictions = get_logits(
                model, images
            ).argmax(1)

        clean_mask = clean_predictions == labels

        initially_correct += clean_mask.sum().item()
        totals["Clean"] += clean_mask.sum().item()

        fgsm_images = fgsm_attack(
            model, images, labels
        )

        with torch.no_grad():
            fgsm_predictions = get_logits(
                model, fgsm_images
            ).argmax(1)

        fgsm_mask = fgsm_predictions == labels

        totals["FGSM"] += fgsm_mask.sum().item()

        attack_success["FGSM"] += (
            clean_mask & (~fgsm_mask)
        ).sum().item()

        pgd_images = pgd_attack(
            model, images, labels
        )

        with torch.no_grad():
            pgd_predictions = get_logits(
                model, pgd_images
            ).argmax(1)

        pgd_mask = pgd_predictions == labels

        totals["PGD"] += pgd_mask.sum().item()

        attack_success["PGD"] += (
            clean_mask & (~pgd_mask)
        ).sum().item()

        total += labels.size(0)

    results = []

    for attack in ["Clean", "FGSM", "PGD"]:

        accuracy = 100 * totals[attack] / total

        asr = (
            100 * attack_success[attack] / initially_correct
            if attack != "Clean" and initially_correct
            else 0
        )

        results.append({
            "Model": "Mixed FGSM Defended CNN",
            "Attack": attack,
            "Clean Accuracy (%)": (
                100 * totals["Clean"] / total
            ),
            "Adversarial Accuracy (%)": accuracy,
            "Attack Success Rate (%)": asr
        })

    return results


results = evaluate()

print("\nMIXED DEFENSE EVALUATION")

for row in results:
    print(row)


# ==========================================
# 9. Save CSV results
# ==========================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_mixed_adversarial_results.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


# ==========================================
# 10. Save training history
# ==========================================

history_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_mixed_training_history.txt"
)

with open(
    history_path,
    "w",
    encoding="utf-8"
) as file:

    for item in history:

        file.write(
            f"Epoch {item['epoch']}: "
            f"Loss={item['loss']:.4f}, "
            f"Clean accuracy={item['clean_accuracy']:.2f}%, "
            f"Adversarial accuracy="
            f"{item['adversarial_accuracy']:.2f}%\n"
        )


# ==========================================
# 11. Save research report
# ==========================================

report_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_mixed_adversarial_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "CIFAR-10 MIXED ADVERSARIAL TRAINING\n"
    )

    file.write("=" * 50 + "\n\n")

    file.write(f"Training subset: {TRAINING_SAMPLES}\n")
    file.write(f"Test sample: {len(test_subset)}\n")
    file.write(f"Epochs: {EPOCHS}\n")
    file.write(f"FGSM epsilon: {EPSILON:.6f}\n")
    file.write("Loss: 0.5 clean + 0.5 adversarial\n\n")

    for row in results:

        file.write(
            f"{row['Attack']}: "
            f"Accuracy="
            f"{row['Adversarial Accuracy (%)']:.2f}%, "
            f"ASR="
            f"{row['Attack Success Rate (%)']:.2f}%\n"
        )

    file.write(
        "\nLIMITATIONS\n"
        "This is a preliminary experiment using a 10,000-image "
        "training subset and a balanced 200-image test sample. "
        "Only FGSM adversarial training was performed. "
        "Additional epochs, full test-set evaluation, "
        "and stronger adaptive attacks are needed.\n"
    )


# ==========================================
# 12. Create training chart
# ==========================================

epochs = [item["epoch"] for item in history]

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    [item["clean_accuracy"] for item in history],
    marker="o",
    label="Clean training accuracy"
)

plt.plot(
    epochs,
    [item["adversarial_accuracy"] for item in history],
    marker="o",
    label="Adversarial training accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Mixed Adversarial Training Progress")
plt.legend()
plt.grid(True)
plt.tight_layout()

chart_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_mixed_training_accuracy.png"
)

plt.savefig(chart_path, dpi=200)
plt.close()


print("\nMixed adversarial training completed.")
print("\nGenerated files:")
print(csv_path)
print(history_path)
print(report_path)
print(chart_path)
print(checkpoint_path)
