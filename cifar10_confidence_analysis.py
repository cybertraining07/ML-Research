
import os
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==================================================
# 1. CONFIGURATION
# ==================================================

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OUTPUT_DIR = "confidence_analysis_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 64
NUM_BINS = 15


# ==================================================
# 2. CUSTOM CNN ARCHITECTURE
# ==================================================

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


# ==================================================
# 3. LOAD TRAINED MODELS
# ==================================================

def load_models():

    cnn = CIFAR10CNN()

    cnn.load_state_dict(
        torch.load(
            "cifar10_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    cnn = cnn.to(device)
    cnn.eval()

    resnet18 = models.resnet18(weights=None)

    resnet18.conv1 = nn.Conv2d(
        3, 64, kernel_size=3,
        stride=1, padding=1, bias=False
    )

    resnet18.maxpool = nn.Identity()

    resnet18.fc = nn.Linear(
        resnet18.fc.in_features, 10
    )

    resnet18.load_state_dict(
        torch.load(
            "cifar10_resnet18.pth",
            map_location=device,
            weights_only=True
        )
    )

    resnet18 = resnet18.to(device)
    resnet18.eval()

    resnet50 = models.resnet50(weights=None)

    resnet50.fc = nn.Linear(
        resnet50.fc.in_features, 10
    )

    resnet50.load_state_dict(
        torch.load(
            "cifar10_resnet50_finetuned.pth",
            map_location=device,
            weights_only=True
        )
    )

    resnet50 = resnet50.to(device)
    resnet50.eval()

    print("All models loaded successfully.")

    return {
        "Custom CNN": cnn,
        "ResNet-18": resnet18,
        "Fine-tuned ResNet-50": resnet50
    }


# ==================================================
# 4. DATASET AND PREPROCESSING
# ==================================================

cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

resnet50_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

test_datasets = {
    "Custom CNN": datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=cifar_transform
    ),

    "ResNet-18": datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=cifar_transform
    ),

    "Fine-tuned ResNet-50": datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=resnet50_transform
    )
}


# ==================================================
# 5. MODEL EVALUATION
# ==================================================

def evaluate_model(model, loader, model_name):

    all_probabilities = []
    all_labels = []

    total = len(loader.dataset)

    print(
        f"\nEvaluating {model_name}: "
        f"{total} test images"
    )

    model.eval()

    with torch.inference_mode():

        for batch_index, (images, labels) in enumerate(loader):

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs, dim=1
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

            if (batch_index + 1) % 25 == 0:

                processed = min(
                    (batch_index + 1) * loader.batch_size,
                    total
                )

                print(
                    f"Processed {processed}/{total}"
                )

    probabilities = np.concatenate(
        all_probabilities
    )

    labels = np.concatenate(all_labels)

    predictions = probabilities.argmax(axis=1)

    confidence = probabilities.max(axis=1)

    correct = predictions == labels

    return {
        "probabilities": probabilities,
        "labels": labels,
        "predictions": predictions,
        "confidence": confidence,
        "correct": correct
    }


# ==================================================
# 6. CALCULATE RESEARCH METRICS
# ==================================================

def calculate_metrics(results):

    probabilities = results["probabilities"]
    labels = results["labels"]
    predictions = results["predictions"]
    confidence = results["confidence"]
    correct = results["correct"]

    accuracy = np.mean(correct)

    avg_confidence = np.mean(confidence)

    correct_confidence = (
        np.mean(confidence[correct])
        if np.any(correct) else float("nan")
    )

    incorrect_confidence = (
        np.mean(confidence[~correct])
        if np.any(~correct) else float("nan")
    )

    # Expected Calibration Error
    bin_edges = np.linspace(
        0.0, 1.0, NUM_BINS + 1
    )

    ece = 0.0

    reliability = []

    for i in range(NUM_BINS):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == NUM_BINS - 1:

            mask = (
                (confidence >= lower) &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower) &
                (confidence < upper)
            )

        if np.any(mask):

            bin_accuracy = np.mean(correct[mask])

            bin_confidence = np.mean(confidence[mask])

            bin_count = np.sum(mask)

            ece += (
                bin_count / len(confidence)
            ) * abs(
                bin_accuracy - bin_confidence
            )

            reliability.append({
                "confidence": bin_confidence,
                "accuracy": bin_accuracy,
                "count": int(bin_count)
            })

    # Predictive entropy in nats
    clipped = np.clip(
        probabilities, 1e-12, 1.0
    )

    entropy = -np.sum(
        clipped * np.log(clipped),
        axis=1
    )

    mean_entropy = np.mean(entropy)

    # Multiclass Brier score
    one_hot = np.eye(10)[labels]

    brier_score = np.mean(
        np.sum(
            (probabilities - one_hot) ** 2,
            axis=1
        )
    )

    high_confidence_errors = (
        (~correct) & (confidence >= 0.90)
    )

    low_confidence_correct = (
        correct & (confidence <= 0.50)
    )

    return {
        "accuracy": accuracy,
        "average_confidence": avg_confidence,
        "correct_confidence": correct_confidence,
        "incorrect_confidence": incorrect_confidence,
        "ece": ece,
        "entropy": mean_entropy,
        "brier": brier_score,
        "high_confidence_errors": int(
            np.sum(high_confidence_errors)
        ),
        "low_confidence_correct": int(
            np.sum(low_confidence_correct)
        ),
        "reliability": reliability
    }


# ==================================================
# 7. CONFIDENCE DISTRIBUTION CHART
# ==================================================

def save_confidence_chart(all_results):

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5)
    )

    for ax, (name, results) in zip(
        axes, all_results.items()
    ):

        confidence = results["confidence"]

        correct = results["correct"]

        ax.hist(
            confidence[correct],
            bins=20,
            alpha=0.65,
            label="Correct"
        )

        ax.hist(
            confidence[~correct],
            bins=20,
            alpha=0.65,
            label="Incorrect"
        )

        ax.set_title(name)

        ax.set_xlabel("Prediction confidence")

        ax.set_ylabel("Number of images")

        ax.legend()

    fig.suptitle(
        "CIFAR-10 Prediction Confidence Distributions"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "cifar10_confidence_distributions.png"
    )

    plt.savefig(path, dpi=200)

    plt.close()

    print("Saved:", path)


# ==================================================
# 8. RELIABILITY DIAGRAM
# ==================================================

def save_reliability_chart(all_metrics):

    plt.figure(figsize=(8, 7))

    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    for name, metrics in all_metrics.items():

        bins = metrics["reliability"]

        if not bins:
            continue

        confidence = [
            item["confidence"] for item in bins
        ]

        accuracy = [
            item["accuracy"] for item in bins
        ]

        plt.plot(
            confidence,
            accuracy,
            marker="o",
            label=name
        )

    plt.xlabel("Mean predicted confidence")

    plt.ylabel("Observed accuracy")

    plt.title(
        "CIFAR-10 Model Calibration / Reliability"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "cifar10_reliability_diagram.png"
    )

    plt.savefig(path, dpi=200)

    plt.close()

    print("Saved:", path)


# ==================================================
# 9. UNCERTAINTY COMPARISON
# ==================================================

def save_uncertainty_chart(all_results):

    names = list(all_results.keys())

    entropy_values = []

    for name in names:

        probabilities = (
            all_results[name]["probabilities"]
        )

        clipped = np.clip(
            probabilities, 1e-12, 1.0
        )

        entropy = -np.sum(
            clipped * np.log(clipped),
            axis=1
        )

        entropy_values.append(entropy)

    plt.figure(figsize=(10, 6))

    plt.boxplot(
        entropy_values,
        tick_labels=names,
        showfliers=False
    )

    plt.ylabel("Predictive entropy (nats)")

    plt.title(
        "Predictive Uncertainty Across CIFAR-10 Models"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "cifar10_uncertainty_comparison.png"
    )

    plt.savefig(path, dpi=200)

    plt.close()

    print("Saved:", path)


# ==================================================
# 10. EXPORT HIGH-CONFIDENCE ERRORS
# ==================================================

def save_error_analysis(all_results):

    path = os.path.join(
        OUTPUT_DIR,
        "cifar10_high_confidence_errors.csv"
    )

    with open(
        path, "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Model",
            "Test_Image_Index",
            "Actual_Class",
            "Predicted_Class",
            "Confidence"
        ])

        for name, results in all_results.items():

            mask = (
                (~results["correct"]) &
                (results["confidence"] >= 0.90)
            )

            indices = np.where(mask)[0]

            indices = sorted(
                indices,
                key=lambda i: results["confidence"][i],
                reverse=True
            )

            for index in indices[:50]:

                writer.writerow([
                    name,
                    int(index),
                    classes[results["labels"][index]],
                    classes[results["predictions"][index]],
                    f"{results['confidence'][index]:.6f}"
                ])

    print("Saved:", path)


# ==================================================
# 11. RESEARCH REPORT
# ==================================================

def save_research_report(all_metrics):

    path = os.path.join(
        OUTPUT_DIR,
        "cifar10_confidence_research_report.txt"
    )

    with open(
        path, "w", encoding="utf-8"
    ) as file:

        file.write(
            "CIFAR-10 CONFIDENCE AND UNCERTAINTY ANALYSIS\n"
        )

        file.write("=" * 65 + "\n\n")

        file.write("Dataset: CIFAR-10 test set\n")
        file.write("Test images per model: 10,000\n")
        file.write("Confidence threshold: 90%\n")
        file.write(f"Calibration bins: {NUM_BINS}\n\n")

        for name, metrics in all_metrics.items():

            file.write("\n" + "=" * 65 + "\n")

            file.write(name + "\n")

            file.write("=" * 65 + "\n")

            file.write(
                f"Accuracy: {metrics['accuracy']:.4%}\n"
            )

            file.write(
                f"Average confidence: "
                f"{metrics['average_confidence']:.4%}\n"
            )

            file.write(
                f"Average confidence on correct predictions: "
                f"{metrics['correct_confidence']:.4%}\n"
            )

            file.write(
                f"Average confidence on incorrect predictions: "
                f"{metrics['incorrect_confidence']:.4%}\n"
            )

            file.write(
                f"Expected Calibration Error (ECE): "
                f"{metrics['ece']:.6f}\n"
            )

            file.write(
                f"Mean predictive entropy: "
                f"{metrics['entropy']:.6f}\n"
            )

            file.write(
                f"Multiclass Brier score: "
                f"{metrics['brier']:.6f}\n"
            )

            file.write(
                f"High-confidence errors (>=90%): "
                f"{metrics['high_confidence_errors']}\n"
            )

            file.write(
                f"Low-confidence correct predictions (<=50%): "
                f"{metrics['low_confidence_correct']}\n"
            )

        file.write("\n\nINTERPRETATION NOTES\n")

        file.write(
            "Lower ECE indicates closer agreement between "
            "confidence and observed accuracy for the chosen bins.\n"
        )

        file.write(
            "Lower multiclass Brier score indicates better "
            "probability predictions.\n"
        )

        file.write(
            "Predictive entropy measures uncertainty in the "
            "model's probability distribution. It is not a "
            "standalone measure of calibration.\n"
        )

        file.write(
            "High-confidence errors identify incorrect predictions "
            "made with at least 90% softmax confidence.\n"
        )

        file.write(
            "These are test-set measurements. They do not guarantee "
            "performance on new or out-of-distribution images.\n"
        )

    print("Saved:", path)


# ==================================================
# 12. MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    models_to_test = load_models()

    all_results = {}

    all_metrics = {}

    for name, model in models_to_test.items():

        loader = DataLoader(
            test_datasets[name],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        results = evaluate_model(
            model, loader, name
        )

        all_results[name] = results

        metrics = calculate_metrics(results)

        all_metrics[name] = metrics

        print(
            f"\n{name} accuracy: "
            f"{metrics['accuracy']:.4%}"
        )

        print(
            f"{name} ECE: "
            f"{metrics['ece']:.6f}"
        )

        print(
            f"{name} high-confidence errors: "
            f"{metrics['high_confidence_errors']}"
        )

    save_confidence_chart(all_results)

    save_reliability_chart(all_metrics)

    save_uncertainty_chart(all_results)

    save_error_analysis(all_results)

    save_research_report(all_metrics)

    print("\n" + "=" * 60)

    print("CONFIDENCE AND UNCERTAINTY ANALYSIS COMPLETED")

    print("=" * 60)

    print("\nGenerated research files:")

    for filename in sorted(os.listdir(OUTPUT_DIR)):

        print("-", filename)
