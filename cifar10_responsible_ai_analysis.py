
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
np.random.seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OUTPUT_DIR = "responsible_ai_results"
BATCH_SIZE = 64
NUM_BINS = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)


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

    print("\nLoading trained models...")

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

    print("All three models loaded successfully.")

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

            all_labels.append(labels.numpy())

            if (batch_index + 1) % 25 == 0:

                processed = min(
                    (batch_index + 1) * loader.batch_size,
                    total
                )

                print(
                    f"Processed {processed}/{total}"
                )

    probabilities = np.concatenate(all_probabilities)
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
# 6. CLASS-WISE CALIBRATION
# ==================================================

def calculate_binary_ece(probabilities, targets):

    """
    One-vs-rest expected calibration error.

    Each class is treated as a binary prediction:
    class probability versus whether the image
    actually belongs to that class.
    """

    bin_edges = np.linspace(0, 1, NUM_BINS + 1)

    ece = 0.0

    for bin_index in range(NUM_BINS):

        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == NUM_BINS - 1:

            mask = (
                (probabilities >= lower) &
                (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower) &
                (probabilities < upper)
            )

        if np.any(mask):

            mean_probability = np.mean(
                probabilities[mask]
            )

            observed_frequency = np.mean(
                targets[mask]
            )

            ece += (
                np.sum(mask) / len(probabilities)
            ) * abs(
                mean_probability - observed_frequency
            )

    return float(ece)


# ==================================================
# 7. CALCULATE CLASS-WISE METRICS
# ==================================================

def calculate_class_metrics(results, model_name):

    probabilities = results["probabilities"]
    labels = results["labels"]
    predictions = results["predictions"]
    confidence = results["confidence"]
    correct = results["correct"]

    class_rows = []

    confusion = np.zeros((10, 10), dtype=np.int64)

    np.add.at(confusion, (labels, predictions), 1)

    for class_index, class_name in enumerate(CLASSES):

        true_positive = confusion[
            class_index, class_index
        ]

        support = np.sum(labels == class_index)

        predicted_count = np.sum(
            predictions == class_index
        )

        false_positive = predicted_count - true_positive

        false_negative = support - true_positive

        class_accuracy = (
            true_positive / support if support else 0.0
        )

        precision = (
            true_positive / predicted_count
            if predicted_count else 0.0
        )

        recall = class_accuracy

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        class_mask = labels == class_index

        class_correct = correct[class_mask]

        class_confidence = confidence[class_mask]

        class_true_probability = probabilities[
            class_mask, class_index
        ]

        correct_confidence = (
            float(np.mean(class_confidence[class_correct]))
            if np.any(class_correct) else 0.0
        )

        incorrect_confidence = (
            float(np.mean(class_confidence[~class_correct]))
            if np.any(~class_correct) else 0.0
        )

        binary_targets = (
            labels == class_index
        ).astype(np.float64)

        class_probabilities = probabilities[:, class_index]

        class_ece = calculate_binary_ece(
            class_probabilities,
            binary_targets
        )

        class_brier = float(
            np.mean(
                (class_probabilities - binary_targets) ** 2
            )
        )

        mean_true_probability = float(
            np.mean(class_true_probability)
        )

        row = {
            "Model": model_name,
            "Class": class_name,
            "Support": int(support),
            "Accuracy_Percent": round(class_accuracy * 100, 4),
            "Precision_Percent": round(precision * 100, 4),
            "Recall_Percent": round(recall * 100, 4),
            "F1_Percent": round(f1 * 100, 4),
            "False_Positives": int(false_positive),
            "False_Negatives": int(false_negative),
            "Correct_Prediction_Confidence_Percent": round(
                correct_confidence * 100, 4
            ),
            "Incorrect_Prediction_Confidence_Percent": round(
                incorrect_confidence * 100, 4
            ),
            "Mean_True_Class_Probability_Percent": round(
                mean_true_probability * 100, 4
            ),
            "One_vs_Rest_ECE": round(class_ece, 6),
            "One_vs_Rest_Brier": round(class_brier, 6)
        }

        class_rows.append(row)

    return class_rows, confusion


# ==================================================
# 8. MODEL-LEVEL SUMMARY
# ==================================================

def calculate_model_summary(results, class_rows):

    correct = results["correct"]

    accuracy = float(np.mean(correct))

    class_f1 = [
        row["F1_Percent"] for row in class_rows
    ]

    class_accuracy = [
        row["Accuracy_Percent"] for row in class_rows
    ]

    class_ece = [
        row["One_vs_Rest_ECE"] for row in class_rows
    ]

    class_brier = [
        row["One_vs_Rest_Brier"] for row in class_rows
    ]

    worst_index = int(np.argmin(class_accuracy))

    return {
        "Model": class_rows[0]["Model"],
        "Overall_Accuracy_Percent": round(
            accuracy * 100, 4
        ),
        "Macro_F1_Percent": round(
            float(np.mean(class_f1)), 4
        ),
        "Mean_Class_Accuracy_Percent": round(
            float(np.mean(class_accuracy)), 4
        ),
        "Mean_One_vs_Rest_ECE": round(
            float(np.mean(class_ece)), 6
        ),
        "Mean_One_vs_Rest_Brier": round(
            float(np.mean(class_brier)), 6
        ),
        "Worst_Performing_Class": CLASSES[worst_index],
        "Worst_Class_Accuracy_Percent": round(
            class_accuracy[worst_index], 4
        )
    }


# ==================================================
# 9. SAVE CSV FILES
# ==================================================

def save_csv(filename, rows):

    if not rows:
        return

    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", filepath)


# ==================================================
# 10. GENERATE RESEARCH CHARTS
# ==================================================

def generate_charts(all_class_rows, summary_rows):

    model_names = [
        "Custom CNN",
        "ResNet-18",
        "Fine-tuned ResNet-50"
    ]

    # ----------------------------------------------
    # Chart 1: Class-wise accuracy
    # ----------------------------------------------

    x = np.arange(len(CLASSES))
    width = 0.25

    plt.figure(figsize=(15, 7))

    for index, model_name in enumerate(model_names):

        rows = [
            row for row in all_class_rows
            if row["Model"] == model_name
        ]

        values = [
            row["Accuracy_Percent"] for row in rows
        ]

        plt.bar(
            x + (index - 1) * width,
            values,
            width,
            label=model_name
        )

    plt.xticks(x, CLASSES, rotation=30)
    plt.ylabel("Class Accuracy (%)")
    plt.xlabel("CIFAR-10 Class")
    plt.title("Class-wise Accuracy Comparison")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "cifar10_classwise_accuracy.png"
        ),
        dpi=200
    )

    plt.close()

    # ----------------------------------------------
    # Chart 2: Class-wise one-vs-rest ECE
    # ----------------------------------------------

    plt.figure(figsize=(15, 7))

    for index, model_name in enumerate(model_names):

        rows = [
            row for row in all_class_rows
            if row["Model"] == model_name
        ]

        values = [
            row["One_vs_Rest_ECE"] for row in rows
        ]

        plt.bar(
            x + (index - 1) * width,
            values,
            width,
            label=model_name
        )

    plt.xticks(x, CLASSES, rotation=30)
    plt.ylabel("One-vs-Rest Expected Calibration Error")
    plt.xlabel("CIFAR-10 Class")
    plt.title("Class-wise Calibration Error")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "cifar10_classwise_calibration.png"
        ),
        dpi=200
    )

    plt.close()

    # ----------------------------------------------
    # Chart 3: Macro F1 comparison
    # ----------------------------------------------

    plt.figure(figsize=(9, 6))

    names = [
        row["Model"] for row in summary_rows
    ]

    values = [
        row["Macro_F1_Percent"] for row in summary_rows
    ]

    plt.bar(names, values)

    plt.ylabel("Macro F1 (%)")
    plt.title("Macro F1 Comparison")
    plt.ylim(0, 100)
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "cifar10_macro_f1_comparison.png"
        ),
        dpi=200
    )

    plt.close()

    print("All research charts generated.")


# ==================================================
# 11. GENERATE RESEARCH REPORT
# ==================================================

def generate_report(summary_rows, all_class_rows):

    filepath = os.path.join(
        OUTPUT_DIR,
        "cifar10_responsible_ai_report.txt"
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "CIFAR-10 RESPONSIBLE AI AND CLASS-WISE "
            "PERFORMANCE ANALYSIS\n"
        )

        file.write("=" * 65 + "\n\n")

        file.write("EXPERIMENT DESIGN\n")
        file.write("-" * 65 + "\n")

        file.write(
            "Dataset: CIFAR-10 test set\n"
            "Test images per model: 10,000\n"
            "Number of classes: 10\n"
            "Models: Custom CNN, ResNet-18, "
            "Fine-tuned ResNet-50\n"
            "Evaluation device: " + str(device) + "\n\n"
        )

        file.write("MODEL-LEVEL RESULTS\n")
        file.write("-" * 65 + "\n")

        for row in summary_rows:

            file.write(
                f"\nModel: {row['Model']}\n"
            )

            for key, value in row.items():

                if key != "Model":
                    file.write(
                        f"{key}: {value}\n"
                    )

        file.write("\n\nCLASS-WISE OBSERVATIONS\n")
        file.write("-" * 65 + "\n")

        for model_name in [
            "Custom CNN",
            "ResNet-18",
            "Fine-tuned ResNet-50"
        ]:

            rows = [
                row for row in all_class_rows
                if row["Model"] == model_name
            ]

            best = max(
                rows,
                key=lambda row: row["Accuracy_Percent"]
            )

            worst = min(
                rows,
                key=lambda row: row["Accuracy_Percent"]
            )

            file.write(
                f"\n{model_name}\n"
            )

            file.write(
                f"Best class: {best['Class']} "
                f"({best['Accuracy_Percent']}%)\n"
            )

            file.write(
                f"Weakest class: {worst['Class']} "
                f"({worst['Accuracy_Percent']}%)\n"
            )

        file.write("\n\nRESPONSIBLE AI INTERPRETATION\n")
        file.write("-" * 65 + "\n")

        file.write(
            "This experiment measures differences in predictive "
            "performance and probability reliability across "
            "CIFAR-10 object categories.\n\n"
        )

        file.write(
            "Class-wise differences may indicate that a model "
            "does not perform equally well across all object "
            "categories. These differences should be investigated "
            "before deploying a model in settings where errors "
            "have meaningful consequences.\n\n"
        )

        file.write(
            "CIFAR-10 does not provide demographic attributes. "
            "Therefore, this experiment does not establish "
            "demographic fairness, protected-group fairness, "
            "or the absence of societal bias.\n\n"
        )

        file.write(
            "One-vs-rest ECE measures calibration separately "
            "for each class using class probabilities and "
            "binary class membership. It depends on binning "
            "choices and should not be interpreted as a "
            "complete measure of calibration.\n\n"
        )

        file.write(
            "The one-vs-rest Brier score is the mean squared "
            "error between a class probability and its binary "
            "target. Lower values indicate better probabilistic "
            "predictions for that class.\n\n"
        )

        file.write("LIMITATIONS\n")
        file.write("-" * 65 + "\n")

        file.write(
            "1. Results apply to the CIFAR-10 test distribution.\n"
            "2. Object categories are not demographic groups.\n"
            "3. Class-wise metrics do not explain the causes "
            "of performance differences.\n"
            "4. Calibration results depend on the chosen "
            "number of bins.\n"
            "5. No out-of-distribution or real-world deployment "
            "evaluation is included.\n"
            "6. No causal fairness conclusions can be drawn.\n"
            "7. Grad-CAM visual explanations are not causal "
            "proof of model reasoning.\n"
        )

        file.write("\nFUTURE RESEARCH\n")
        file.write("-" * 65 + "\n")

        file.write(
            "1. Test robustness under distribution shifts.\n"
            "2. Evaluate calibration after temperature scaling.\n"
            "3. Investigate class-specific data augmentation.\n"
            "4. Perform repeated experiments with multiple seeds.\n"
            "5. Extend fairness analysis to datasets with "
            "appropriate group attributes and valid evaluation "
            "methodology.\n"
        )

    print("Research report saved:", filepath)


# ==================================================
# 12. MAIN EXECUTION
# ==================================================

def main():

    print("\nStarting Project 11...")
    print("Responsible AI and Class-wise Analysis")

    models_to_evaluate = load_models()

    all_class_rows = []
    summary_rows = []

    for model_name, model in models_to_evaluate.items():

        loader = DataLoader(
            test_datasets[model_name],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        results = evaluate_model(
            model,
            loader,
            model_name
        )

        class_rows, confusion = calculate_class_metrics(
            results,
            model_name
        )

        summary = calculate_model_summary(
            results,
            class_rows
        )

        all_class_rows.extend(class_rows)
        summary_rows.append(summary)

        confusion_path = os.path.join(
            OUTPUT_DIR,
            model_name.lower().replace(" ", "_")
            .replace("-", "_") + "_confusion_matrix.csv"
        )

        np.savetxt(
            confusion_path,
            confusion,
            delimiter=",",
            fmt="%d"
        )

        print(
            f"\n{model_name} overall accuracy: "
            f"{summary['Overall_Accuracy_Percent']}%"
        )

        print(
            "Macro F1:",
            summary["Macro_F1_Percent"]
        )

        print(
            "Weakest class:",
            summary["Worst_Performing_Class"]
        )

    save_csv(
        "cifar10_classwise_results.csv",
        all_class_rows
    )

    save_csv(
        "cifar10_responsible_ai_summary.csv",
        summary_rows
    )

    generate_charts(
        all_class_rows,
        summary_rows
    )

    generate_report(
        summary_rows,
        all_class_rows
    )

    print("\n" + "=" * 60)
    print("PROJECT 11 EVALUATION COMPLETED")
    print("=" * 60)

    print("\nGenerated files:")

    for filename in sorted(os.listdir(OUTPUT_DIR)):
        print("-", filename)


if __name__ == "__main__":
    main()
