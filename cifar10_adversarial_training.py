
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

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 128
EPOCHS = 2
TRAINING_SAMPLES = 10000

EPSILON = 8 / 255
PGD_ALPHA = 2 / 255
PGD_TRAIN_STEPS = 3
PGD_TEST_STEPS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "adversarial_defense_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

BASELINE_CHECKPOINT = "cifar10_model.pth"

MEAN = torch.tensor(
    [0.4914, 0.4822, 0.4465],
    device=DEVICE
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.2470, 0.2435, 0.2616],
    device=DEVICE
).view(1, 3, 1, 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


# --------------------------------------------------
# 2. Custom CNN architecture
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
# 3. Normalize raw pixel images
# --------------------------------------------------

def get_logits(model, raw_images):
    normalized = (raw_images - MEAN) / STD
    return model(normalized)


# --------------------------------------------------
# 4. FGSM attack for training and evaluation
# --------------------------------------------------

def fgsm_attack(model, images, labels, epsilon):

    images = images.detach()

    adversarial = images.clone().requires_grad_(True)

    logits = get_logits(model, adversarial)
    loss = F.cross_entropy(logits, labels)

    gradient = torch.autograd.grad(loss, adversarial)[0]

    adversarial = (
        images + epsilon * gradient.sign()
    ).clamp(0, 1)

    return adversarial.detach()


# --------------------------------------------------
# 5. PGD attack for training and evaluation
# --------------------------------------------------

def pgd_attack(
    model,
    images,
    labels,
    epsilon,
    alpha,
    steps,
    random_start=True
):

    original = images.detach()

    if random_start:
        adversarial = original + torch.empty_like(original).uniform_(
            -epsilon, epsilon
        )
        adversarial = adversarial.clamp(0, 1)
    else:
        adversarial = original.clone()

    for _ in range(steps):

        adversarial = adversarial.detach().requires_grad_(True)

        logits = get_logits(model, adversarial)
        loss = F.cross_entropy(logits, labels)

        gradient = torch.autograd.grad(loss, adversarial)[0]

        adversarial = adversarial.detach() + alpha * gradient.sign()

        perturbation = torch.clamp(
            adversarial - original,
            min=-epsilon,
            max=epsilon
        )

        adversarial = (original + perturbation).clamp(0, 1)

    return adversarial.detach()


# --------------------------------------------------
# 6. Load CIFAR-10 training and test data
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.ToTensor()

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
    transform=test_transform
)

generator = torch.Generator().manual_seed(SEED)

selected_train_indices = torch.randperm(
    len(train_dataset),
    generator=generator
)[:TRAINING_SAMPLES].tolist()

train_subset = Subset(train_dataset, selected_train_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# Select 20 images from each CIFAR-10 class.
targets = np.array(test_dataset.targets)

test_indices = []

for class_id in range(10):
    class_indices = np.where(targets == class_id)[0]
    test_indices.extend(class_indices[:20].tolist())

test_subset = Subset(test_dataset, test_indices)

test_loader = DataLoader(
    test_subset,
    batch_size=50,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 7. Load a fresh copy of the baseline CNN
# --------------------------------------------------

def load_baseline_model():

    model = CIFAR10CNN().to(DEVICE)

    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    return model


# --------------------------------------------------
# 8. Adversarial training
# --------------------------------------------------

def train_defense(model, defense_type):

    optimizer = Adam(model.parameters(), lr=0.0002)

    history = []

    print(f"\nStarting {defense_type} adversarial training")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_images = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Generate adversarial examples without
            # updating model weights during attack generation.
            model.eval()

            if defense_type == "FGSM":
                adversarial = fgsm_attack(
                    model, images, labels, EPSILON
                )

            elif defense_type == "PGD":
                adversarial = pgd_attack(
                    model,
                    images,
                    labels,
                    EPSILON,
                    PGD_ALPHA,
                    PGD_TRAIN_STEPS,
                    random_start=True
                )

            else:
                raise ValueError("Unknown defense type")

            # Update the model using adversarial examples.
            model.train()
            optimizer.zero_grad()

            logits = get_logits(model, adversarial)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_images += labels.size(0)

        average_loss = total_loss / total_images
        training_accuracy = 100 * total_correct / total_images

        history.append({
            "epoch": epoch + 1,
            "loss": average_loss,
            "accuracy": training_accuracy
        })

        print(
            f"{defense_type} | "
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {training_accuracy:.2f}%"
        )

    checkpoint_path = os.path.join(
        RESULTS_DIR,
        f"cifar10_{defense_type.lower()}_defended.pth"
    )

    torch.save(model.state_dict(), checkpoint_path)

    return history, checkpoint_path


# --------------------------------------------------
# 9. Evaluate clean and adversarial accuracy
# --------------------------------------------------

def evaluate_model(model, model_name):

    model.eval()

    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0
    attack_success_fgsm = 0
    attack_success_pgd = 0
    initially_correct = 0
    total = 0

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            clean_logits = get_logits(model, images)
            clean_predictions = clean_logits.argmax(dim=1)

        clean_mask = clean_predictions == labels

        clean_correct += clean_mask.sum().item()
        initially_correct += clean_mask.sum().item()
        total += labels.size(0)

        # FGSM evaluation
        fgsm_images = fgsm_attack(
            model, images, labels, EPSILON
        )

        with torch.no_grad():
            fgsm_predictions = get_logits(
                model, fgsm_images
            ).argmax(dim=1)

        fgsm_mask = fgsm_predictions == labels

        fgsm_correct += fgsm_mask.sum().item()

        attack_success_fgsm += (
            clean_mask & (~fgsm_mask)
        ).sum().item()

        # PGD evaluation
        pgd_images = pgd_attack(
            model,
            images,
            labels,
            EPSILON,
            PGD_ALPHA,
            PGD_TEST_STEPS,
            random_start=True
        )

        with torch.no_grad():
            pgd_predictions = get_logits(
                model, pgd_images
            ).argmax(dim=1)

        pgd_mask = pgd_predictions == labels

        pgd_correct += pgd_mask.sum().item()

        attack_success_pgd += (
            clean_mask & (~pgd_mask)
        ).sum().item()

    results = [
        {
            "Model": model_name,
            "Attack": "Clean",
            "Epsilon": 0,
            "Clean Accuracy (%)": 100 * clean_correct / total,
            "Adversarial Accuracy (%)": 100 * clean_correct / total,
            "Attack Success Rate (%)": 0
        },
        {
            "Model": model_name,
            "Attack": "FGSM",
            "Epsilon": EPSILON,
            "Clean Accuracy (%)": 100 * clean_correct / total,
            "Adversarial Accuracy (%)": 100 * fgsm_correct / total,
            "Attack Success Rate (%)": (
                100 * attack_success_fgsm / initially_correct
                if initially_correct else 0
            )
        },
        {
            "Model": model_name,
            "Attack": "PGD",
            "Epsilon": EPSILON,
            "Clean Accuracy (%)": 100 * clean_correct / total,
            "Adversarial Accuracy (%)": 100 * pgd_correct / total,
            "Attack Success Rate (%)": (
                100 * attack_success_pgd / initially_correct
                if initially_correct else 0
            )
        }
    ]

    return results


# --------------------------------------------------
# 10. Run both defense experiments
# --------------------------------------------------

print("Device:", DEVICE)
print("Training images:", TRAINING_SAMPLES)
print("Evaluation images:", len(test_subset))
print("Epsilon:", EPSILON)

all_histories = {}
all_checkpoints = {}

for defense in ["FGSM", "PGD"]:

    model = load_baseline_model()

    history, checkpoint_path = train_defense(
        model, defense
    )

    all_histories[defense] = history
    all_checkpoints[defense] = checkpoint_path


# --------------------------------------------------
# 11. Evaluate baseline and defended models
# --------------------------------------------------

evaluation_models = {
    "Baseline CNN": load_baseline_model()
}

for defense in ["FGSM", "PGD"]:

    model = CIFAR10CNN().to(DEVICE)

    checkpoint = torch.load(
        all_checkpoints[defense],
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    evaluation_models[f"{defense} Defended CNN"] = model


all_results = []

for model_name, model in evaluation_models.items():

    print(f"\nEvaluating {model_name}")

    model_results = evaluate_model(model, model_name)

    all_results.extend(model_results)

    for result in model_results:
        print(result)


# --------------------------------------------------
# 12. Save CSV results
# --------------------------------------------------

csv_path = os.path.join(
    RESULTS_DIR,
    "cifar10_adversarial_defense_results.csv"
)

fieldnames = list(all_results[0].keys())

with open(csv_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)


# --------------------------------------------------
# 13. Save training history
# --------------------------------------------------

history_path = os.path.join(
    RESULTS_DIR,
    "cifar10_adversarial_training_history.txt"
)

with open(history_path, "w", encoding="utf-8") as file:

    for defense, history in all_histories.items():

        file.write(f"\n{defense} DEFENSE TRAINING\n")

        for item in history:
            file.write(
                f"Epoch {item['epoch']}: "
                f"Loss={item['loss']:.4f}, "
                f"Accuracy={item['accuracy']:.2f}%\n"
            )


# --------------------------------------------------
# 14. Create accuracy comparison chart
# --------------------------------------------------

model_names = list(evaluation_models.keys())

attacks = ["Clean", "FGSM", "PGD"]

x = np.arange(len(model_names))
width = 0.25

plt.figure(figsize=(11, 6))

for index, attack in enumerate(attacks):

    values = []

    for model_name in model_names:

        matching = [
            row for row in all_results
            if row["Model"] == model_name
            and row["Attack"] == attack
        ]

        values.append(
            matching[0]["Adversarial Accuracy (%)"]
        )

    plt.bar(
        x + (index - 1) * width,
        values,
        width,
        label=attack
    )

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Clean and Adversarial Accuracy: Defense Comparison")
plt.xticks(x, model_names, rotation=10)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

chart_path = os.path.join(
    RESULTS_DIR,
    "cifar10_adversarial_defense_comparison.png"
)

plt.savefig(chart_path, dpi=200)
plt.close()


# --------------------------------------------------
# 15. Generate research report
# --------------------------------------------------

report_path = os.path.join(
    RESULTS_DIR,
    "cifar10_adversarial_defense_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("CIFAR-10 ADVERSARIAL DEFENSE RESEARCH\n")
    file.write("=" * 55 + "\n\n")

    file.write(f"Device: {DEVICE}\n")
    file.write(f"Training sample size: {TRAINING_SAMPLES}\n")
    file.write(f"Evaluation sample size: {len(test_subset)}\n")
    file.write(f"Training epochs: {EPOCHS}\n")
    file.write(f"FGSM epsilon: {EPSILON:.6f}\n")
    file.write(f"PGD training steps: {PGD_TRAIN_STEPS}\n")
    file.write(f"PGD evaluation steps: {PGD_TEST_STEPS}\n\n")

    file.write("EXPERIMENT RESULTS\n")
    file.write("-" * 55 + "\n")

    for row in all_results:

        file.write(
            f"\nModel: {row['Model']}\n"
            f"Attack: {row['Attack']}\n"
            f"Clean Accuracy: {row['Clean Accuracy (%)']:.2f}%\n"
            f"Adversarial Accuracy: "
            f"{row['Adversarial Accuracy (%)']:.2f}%\n"
            f"Attack Success Rate: "
            f"{row['Attack Success Rate (%)']:.2f}%\n"
        )

    file.write("\nRESEARCH LIMITATIONS\n")
    file.write("-" * 55 + "\n")
    file.write(
        "The experiment uses a balanced 200-image CIFAR-10 "
        "test sample and a randomly selected 10,000-image "
        "training subset. Training is limited to two epochs. "
        "Results are preliminary and do not establish "
        "general adversarial robustness. The same attack "
        "settings are used for model comparison, but stronger "
        "attacks and additional evaluation are needed.\n"
    )


print("\nAdversarial defense experiments completed.")

print("\nGenerated files:")
print(csv_path)
print(history_path)
print(chart_path)
print(report_path)

for path in all_checkpoints.values():
    print(path)
