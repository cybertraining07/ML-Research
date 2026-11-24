
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import datasets, models
from torch.utils.data import DataLoader, Subset

# ============================================================
# CIFAR-10 ADVERSARIAL ROBUSTNESS RESEARCH
# Project 9: FGSM and PGD attacks
# ============================================================

SEED = 42
NUM_SAMPLES_PER_CLASS = 20
BATCH_SIZE = 16

FGSM_EPSILONS = [4 / 255, 8 / 255, 16 / 255]
PGD_EPSILON = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 10

DEVICE = torch.device("cpu")

OUTPUT_DIR = "adversarial_robustness_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

print("=" * 65)
print("CIFAR-10 ADVERSARIAL ROBUSTNESS ANALYSIS")
print("=" * 65)
print("Device:", DEVICE)
print("Random seed:", SEED)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# CIFAR-10 normalization
CIFAR_MEAN = torch.tensor(
    [0.4914, 0.4822, 0.4465]
).view(1, 3, 1, 1)

CIFAR_STD = torch.tensor(
    [0.2470, 0.2435, 0.2616]
).view(1, 3, 1, 1)

# ImageNet normalization for ResNet-50
IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)


# ============================================================
# 1. MODEL ARCHITECTURES
# ============================================================

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


def build_resnet18():
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3,
        stride=1, padding=1, bias=False
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


def build_resnet50():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


# ============================================================
# 2. LOAD CHECKPOINTS
# ============================================================

print("\nLoading trained models...")

models_dict = {
    "Custom CNN": CIFAR10CNN(),
    "ResNet-18": build_resnet18(),
    "Fine-tuned ResNet-50": build_resnet50()
}

checkpoint_paths = {
    "Custom CNN": "cifar10_model.pth",
    "ResNet-18": "cifar10_resnet18.pth",
    "Fine-tuned ResNet-50": "cifar10_resnet50_finetuned.pth"
}

for name, model in models_dict.items():

    checkpoint_path = checkpoint_paths[name]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print(f"{name}: checkpoint loaded successfully")


# ============================================================
# 3. MODEL-SPECIFIC PREPROCESSING
# ============================================================

def model_logits(model, images, model_name):
    """
    Input images must be raw RGB tensors in [0, 1].
    Normalization is applied inside this function.
    """

    if model_name == "Fine-tuned ResNet-50":

        mean = IMAGENET_MEAN.to(images.device)
        std = IMAGENET_STD.to(images.device)

        # Differentiable resizing for adversarial gradients.
        images = F.interpolate(
            images,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True
        )

    else:

        mean = CIFAR_MEAN.to(images.device)
        std = CIFAR_STD.to(images.device)

    normalized_images = (images - mean) / std

    return model(normalized_images)


# ============================================================
# 4. LOAD A BALANCED CIFAR-10 TEST SAMPLE
# ============================================================

print("\nLoading CIFAR-10 test dataset...")

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=None
)

selected_indices = []

targets = np.array(test_dataset.targets)

for class_index in range(10):

    class_indices = np.where(targets == class_index)[0]

    selected_indices.extend(
        class_indices[:NUM_SAMPLES_PER_CLASS].tolist()
    )

test_subset = Subset(
    datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=lambda image: torch.from_numpy(
            np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )
    ),
    selected_indices
)

test_loader = DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Selected test images:", len(test_subset))
print("Images per class:", NUM_SAMPLES_PER_CLASS)


# ============================================================
# 5. FGSM ATTACK
# ============================================================

def fgsm_attack(model, images, labels, model_name, epsilon):

    original_images = images.detach()

    attacked_images = original_images.clone()
    attacked_images.requires_grad_(True)

    outputs = model_logits(
        model,
        attacked_images,
        model_name
    )

    loss = F.cross_entropy(outputs, labels)

    model.zero_grad(set_to_none=True)
    loss.backward()

    gradient = attacked_images.grad.data

    perturbed_images = (
        attacked_images + epsilon * gradient.sign()
    )

    perturbed_images = torch.clamp(
        perturbed_images,
        0.0,
        1.0
    )

    return perturbed_images.detach()


# ============================================================
# 6. PGD ATTACK
# ============================================================

def pgd_attack(
    model,
    images,
    labels,
    model_name,
    epsilon,
    alpha,
    steps
):

    original_images = images.detach()

    # Random initialization inside the epsilon ball.
    perturbed_images = original_images + torch.empty_like(
        original_images
    ).uniform_(-epsilon, epsilon)

    perturbed_images = torch.clamp(
        perturbed_images,
        0.0,
        1.0
    )

    for step in range(steps):

        perturbed_images = perturbed_images.detach()
        perturbed_images.requires_grad_(True)

        outputs = model_logits(
            model,
            perturbed_images,
            model_name
        )

        loss = F.cross_entropy(outputs, labels)

        model.zero_grad(set_to_none=True)
        loss.backward()

        gradient = perturbed_images.grad.data

        perturbed_images = (
            perturbed_images.detach()
            + alpha * gradient.sign()
        )

        # Project back into the permitted perturbation range.
        difference = torch.clamp(
            perturbed_images - original_images,
            min=-epsilon,
            max=epsilon
        )

        perturbed_images = torch.clamp(
            original_images + difference,
            0.0,
            1.0
        )

    return perturbed_images.detach()


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_clean(model, model_name):

    correct = 0
    total = 0

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():

            outputs = model_logits(
                model,
                images,
                model_name
            )

            predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


def evaluate_attack(
    model,
    model_name,
    attack_name,
    epsilon,
    alpha=None,
    steps=None
):

    total = 0
    adversarial_correct = 0
    clean_correct = 0
    successful_attacks = 0

    example_images = None
    example_adversarial = None
    example_labels = None
    example_predictions = None

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():

            clean_outputs = model_logits(
                model,
                images,
                model_name
            )

            clean_predictions = clean_outputs.argmax(dim=1)

        clean_mask = clean_predictions.eq(labels)

        if attack_name == "FGSM":

            adversarial_images = fgsm_attack(
                model,
                images,
                labels,
                model_name,
                epsilon
            )

        elif attack_name == "PGD":

            adversarial_images = pgd_attack(
                model,
                images,
                labels,
                model_name,
                epsilon,
                alpha,
                steps
            )

        else:
            raise ValueError("Unknown attack")

        with torch.no_grad():

            adversarial_outputs = model_logits(
                model,
                adversarial_images,
                model_name
            )

            adversarial_predictions = (
                adversarial_outputs.argmax(dim=1)
            )

        adversarial_mask = adversarial_predictions.eq(labels)

        batch_clean_correct = clean_mask.sum().item()

        clean_correct += batch_clean_correct

        adversarial_correct += adversarial_mask.sum().item()

        # Attack success is measured only among initially
        # correctly classified images.
        successful_attacks += (
            clean_mask & (~adversarial_mask)
        ).sum().item()

        total += labels.size(0)

        if example_images is None:
            example_images = images[:4].detach().cpu()
            example_adversarial = (
                adversarial_images[:4].detach().cpu()
            )
            example_labels = labels[:4].detach().cpu()
            example_predictions = (
                adversarial_predictions[:4].detach().cpu()
            )

    robust_accuracy = 100.0 * adversarial_correct / total

    if clean_correct > 0:
        attack_success_rate = (
            100.0 * successful_attacks / clean_correct
        )
    else:
        attack_success_rate = 0.0

    return {
        "Model": model_name,
        "Attack": attack_name,
        "Epsilon": epsilon,
        "Clean Accuracy (%)": None,
        "Adversarial Accuracy (%)": robust_accuracy,
        "Attack Success Rate (%)": attack_success_rate,
        "Clean Correct": clean_correct,
        "Successful Attacks": successful_attacks,
        "Total Images": total,
        "Original Images": example_images,
        "Adversarial Images": example_adversarial,
        "Labels": example_labels,
        "Adversarial Predictions": example_predictions
    }


# ============================================================
# 8. RUN EXPERIMENTS
# ============================================================

results = []
visual_examples = {}

for model_name, model in models_dict.items():

    print("\n" + "=" * 65)
    print("Evaluating:", model_name)
    print("=" * 65)

    clean_accuracy = evaluate_clean(
        model,
        model_name
    )

    print(f"Clean accuracy: {clean_accuracy:.2f}%")

    for epsilon in FGSM_EPSILONS:

        print(
            f"\nRunning FGSM with epsilon={epsilon:.5f}"
        )

        result = evaluate_attack(
            model,
            model_name,
            "FGSM",
            epsilon
        )

        result["Clean Accuracy (%)"] = clean_accuracy

        results.append(result)

        print(
            f"Adversarial accuracy: "
            f"{result['Adversarial Accuracy (%)']:.2f}%"
        )

        print(
            f"Attack success rate: "
            f"{result['Attack Success Rate (%)']:.2f}%"
        )

        if abs(epsilon - PGD_EPSILON) < 1e-8:
            visual_examples[(model_name, "FGSM")] = result

    print("\nRunning PGD attack...")

    pgd_result = evaluate_attack(
        model,
        model_name,
        "PGD",
        PGD_EPSILON,
        PGD_ALPHA,
        PGD_STEPS
    )

    pgd_result["Clean Accuracy (%)"] = clean_accuracy

    results.append(pgd_result)

    visual_examples[(model_name, "PGD")] = pgd_result

    print(
        f"PGD adversarial accuracy: "
        f"{pgd_result['Adversarial Accuracy (%)']:.2f}%"
    )

    print(
        f"PGD attack success rate: "
        f"{pgd_result['Attack Success Rate (%)']:.2f}%"
    )


# ============================================================
# 9. SAVE CSV RESULTS
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_adversarial_results.csv"
)

csv_columns = [
    "Model",
    "Attack",
    "Epsilon",
    "Clean Accuracy (%)",
    "Adversarial Accuracy (%)",
    "Attack Success Rate (%)",
    "Clean Correct",
    "Successful Attacks",
    "Total Images"
]

with open(csv_path, "w", newline="") as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=csv_columns
    )

    writer.writeheader()

    for result in results:

        writer.writerow({
            key: result[key]
            for key in csv_columns
        })

print("\nCSV saved:", csv_path)


# ============================================================
# 10. GENERATE ACCURACY COMPARISON CHART
# ============================================================

models_names = list(models_dict.keys())

attack_labels = [
    "FGSM 4/255",
    "FGSM 8/255",
    "FGSM 16/255",
    "PGD 8/255"
]

fig, axes = plt.subplots(
    1, 3,
    figsize=(17, 5)
)

for axis, model_name in zip(axes, models_names):

    model_results = [
        r for r in results if r["Model"] == model_name
    ]

    values = [
        r["Adversarial Accuracy (%)"]
        for r in model_results
    ]

    axis.bar(attack_labels, values)

    axis.set_title(model_name)
    axis.set_ylabel("Adversarial Accuracy (%)")
    axis.set_ylim(0, 100)
    axis.tick_params(axis="x", rotation=35)

    axis.grid(axis="y", alpha=0.25)

plt.suptitle(
    "CIFAR-10 Adversarial Robustness Comparison"
)

plt.tight_layout()

chart_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_adversarial_accuracy_comparison.png"
)

plt.savefig(chart_path, dpi=200)
plt.close()

print("Accuracy chart saved:", chart_path)


# ============================================================
# 11. VISUALIZE CLEAN AND ADVERSARIAL EXAMPLES
# ============================================================

fig, axes = plt.subplots(
    3, 3,
    figsize=(12, 11)
)

for row, model_name in enumerate(models_names):

    fgsm_data = visual_examples[(model_name, "FGSM")]
    pgd_data = visual_examples[(model_name, "PGD")]

    original = fgsm_data["Original Images"][0].permute(
        1, 2, 0
    ).numpy()

    fgsm_image = fgsm_data["Adversarial Images"][0].permute(
        1, 2, 0
    ).numpy()

    pgd_image = pgd_data["Adversarial Images"][0].permute(
        1, 2, 0
    ).numpy()

    label = int(fgsm_data["Labels"][0])

    fgsm_prediction = int(
        fgsm_data["Adversarial Predictions"][0]
    )

    pgd_prediction = int(
        pgd_data["Adversarial Predictions"][0]
    )

    images_to_show = [
        original,
        fgsm_image,
        pgd_image
    ]

    titles = [
        f"Original: {CLASSES[label]}",
        f"FGSM: {CLASSES[fgsm_prediction]}",
        f"PGD: {CLASSES[pgd_prediction]}"
    ]

    for column in range(3):

        axes[row, column].imshow(
            np.clip(images_to_show[column], 0, 1)
        )

        axes[row, column].set_title(titles[column])
        axes[row, column].axis("off")

    axes[row, 0].set_ylabel(
        model_name,
        fontsize=10
    )

plt.suptitle(
    "Clean and Adversarial Image Examples"
)

plt.tight_layout()

examples_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_adversarial_examples.png"
)

plt.savefig(examples_path, dpi=200)
plt.close()

print("Example visualization saved:", examples_path)


# ============================================================
# 12. GENERATE RESEARCH REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_adversarial_research_report.txt"
)

with open(report_path, "w") as report:

    report.write(
        "CIFAR-10 ADVERSARIAL ROBUSTNESS RESEARCH\n"
    )

    report.write("=" * 65 + "\n\n")

    report.write("Dataset: CIFAR-10 test set\n")
    report.write(f"Images per model: {len(test_subset)}\n")
    report.write(
        f"Images per class: {NUM_SAMPLES_PER_CLASS}\n"
    )

    report.write("Device: " + str(DEVICE) + "\n")
    report.write(f"Random seed: {SEED}\n\n")

    report.write("Attack configurations:\n")
    report.write(f"FGSM epsilons: {FGSM_EPSILONS}\n")
    report.write(f"PGD epsilon: {PGD_EPSILON}\n")
    report.write(f"PGD alpha: {PGD_ALPHA}\n")
    report.write(f"PGD steps: {PGD_STEPS}\n\n")

    report.write("=" * 65 + "\n")
    report.write("EXPERIMENTAL RESULTS\n")
    report.write("=" * 65 + "\n\n")

    for result in results:

        report.write(f"Model: {result['Model']}\n")
        report.write(f"Attack: {result['Attack']}\n")
        report.write(f"Epsilon: {result['Epsilon']:.6f}\n")

        report.write(
            f"Clean accuracy: "
            f"{result['Clean Accuracy (%)']:.2f}%\n"
        )

        report.write(
            f"Adversarial accuracy: "
            f"{result['Adversarial Accuracy (%)']:.2f}%\n"
        )

        report.write(
            f"Attack success rate: "
            f"{result['Attack Success Rate (%)']:.2f}%\n"
        )

        report.write(
            f"Initially correct images: "
            f"{result['Clean Correct']}\n"
        )

        report.write(
            f"Successful attacks: "
            f"{result['Successful Attacks']}\n"
        )

        report.write("-" * 50 + "\n")

    report.write("\nRESEARCH NOTES\n")
    report.write(
        "Attack success rate is calculated among images "
        "correctly classified before the attack.\n"
    )

    report.write(
        "Adversarial accuracy is calculated over the full "
        "selected test sample.\n"
    )

    report.write(
        "Perturbations are applied in raw pixel space "
        "with values constrained to [0, 1].\n"
    )

    report.write(
        "ResNet-50 uses differentiable tensor resizing "
        "to 224x224 during attack evaluation.\n"
    )

    report.write(
        "These results use a balanced 200-image sample "
        "and should not be interpreted as full test-set "
        "robustness measurements.\n"
    )

    report.write(
        "The experiment evaluates empirical robustness "
        "against the specified attacks, not guaranteed "
        "robustness against all adversarial attacks.\n"
    )

print("Research report saved:", report_path)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("ADVERSARIAL ROBUSTNESS EXPERIMENT COMPLETED")
print("=" * 65)

print("\nGenerated files:")

for filename in [
    "cifar10_adversarial_results.csv",
    "cifar10_adversarial_accuracy_comparison.png",
    "cifar10_adversarial_examples.png",
    "cifar10_adversarial_research_report.txt"
]:

    print(os.path.join(OUTPUT_DIR, filename))

print("\nAll experiments completed successfully.")
