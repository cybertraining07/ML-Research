
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

os.makedirs("gradcam_results", exist_ok=True)


# --------------------------------------------------
# 2. CUSTOM CIFAR-10 CNN ARCHITECTURE
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
# 3. LOAD THE THREE TRAINED MODELS
# --------------------------------------------------

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
    resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)

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

    print("All three trained models loaded successfully.")

    return cnn, resnet18, resnet50


# --------------------------------------------------
# 4. GRAD-CAM IMPLEMENTATION
# --------------------------------------------------

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.activations = None
        self.gradients = None

        self.hook = target_layer.register_forward_hook(
            self.forward_hook
        )

    def forward_hook(self, module, inputs, output):

        self.activations = output

        output.register_hook(self.save_gradient)

    def save_gradient(self, gradient):

        self.gradients = gradient

    def generate(self, input_tensor):

        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)

        probabilities = torch.softmax(output, dim=1)

        predicted_class = output.argmax(dim=1).item()

        confidence = probabilities[0, predicted_class].item()

        output[0, predicted_class].backward()

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * self.activations
        ).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(32, 32),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        cam_min = cam.min()
        cam_max = cam.max()

        if (cam_max - cam_min).item() > 1e-8:

            cam = (cam - cam_min) / (cam_max - cam_min)

        else:

            cam = torch.zeros_like(cam)

        return (
            cam.detach().cpu().numpy(),
            predicted_class,
            confidence
        )

    def close(self):

        self.hook.remove()


# --------------------------------------------------
# 5. LOAD CIFAR-10 TEST IMAGES
# --------------------------------------------------

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=None
)

# Six fixed examples for reproducible visualization.
sample_indices = [0, 1, 2, 3, 4, 5]

print("CIFAR-10 test images loaded.")


# --------------------------------------------------
# 6. MODEL PREPROCESSING
# --------------------------------------------------

cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

resnet50_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


# --------------------------------------------------
# 7. GENERATE GRAD-CAM VISUALIZATIONS
# --------------------------------------------------

def run_gradcam(model, target_layer, transform, model_name):

    gradcam = GradCAM(model, target_layer)

    results = []

    for index in sample_indices:

        image, actual_label = test_dataset[index]

        raw_image = transforms.ToTensor()(image)

        input_tensor = transform(image).unsqueeze(0)

        input_tensor = input_tensor.to(device)

        cam, prediction, confidence = gradcam.generate(
            input_tensor
        )

        results.append({
            "index": index,
            "image": raw_image.permute(1, 2, 0).numpy(),
            "cam": cam,
            "actual": actual_label,
            "prediction": prediction,
            "confidence": confidence
        })

        print(
            f"{model_name} | Image {index} | "
            f"Actual: {classes[actual_label]} | "
            f"Predicted: {classes[prediction]} | "
            f"Confidence: {confidence:.4f}"
        )

    gradcam.close()

    return results


# --------------------------------------------------
# 8. CREATE COMPARISON VISUALIZATION
# --------------------------------------------------

def save_comparison(all_results):

    model_names = [
        "Custom CNN",
        "ResNet-18",
        "Fine-tuned ResNet-50"
    ]

    fig, axes = plt.subplots(
        3, len(sample_indices),
        figsize=(18, 10)
    )

    for row, model_name in enumerate(model_names):

        for col, result in enumerate(all_results[model_name]):

            ax = axes[row, col]

            ax.imshow(result["image"])

            ax.imshow(
                result["cam"],
                cmap="jet",
                alpha=0.45,
                vmin=0,
                vmax=1
            )

            actual = classes[result["actual"]]
            predicted = classes[result["prediction"]]

            status = (
                "CORRECT"
                if result["actual"] == result["prediction"]
                else "INCORRECT"
            )

            ax.set_title(
                f"True: {actual}\n"
                f"Pred: {predicted}\n"
                f"Confidence: {result['confidence']:.2%}\n"
                f"{status}",
                fontsize=8
            )

            ax.axis("off")

        axes[row, 0].set_ylabel(
            model_name,
            fontsize=12
        )

    plt.suptitle(
        "CIFAR-10 Grad-CAM Model Explainability Comparison",
        fontsize=16
    )

    plt.tight_layout()

    output_path = (
        "gradcam_results/"
        "cifar10_gradcam_three_model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Grad-CAM comparison saved:", output_path)


# --------------------------------------------------
# 9. GENERATE RESEARCH REPORT
# --------------------------------------------------

def save_report(all_results):

    report_path = (
        "gradcam_results/"
        "cifar10_gradcam_research_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:

        file.write(
            "CIFAR-10 MODEL EXPLAINABILITY RESEARCH\n"
        )

        file.write("=" * 55 + "\n\n")

        file.write("Method: Gradient-weighted Class Activation Mapping\n")

        file.write("Dataset: CIFAR-10 test dataset\n")

        file.write("Number of images analyzed per model: 6\n\n")

        for model_name, results in all_results.items():

            file.write("\n" + "=" * 55 + "\n")

            file.write(model_name + "\n")

            file.write("=" * 55 + "\n")

            for result in results:

                actual = classes[result["actual"]]

                predicted = classes[result["prediction"]]

                status = (
                    "CORRECT"
                    if result["actual"] == result["prediction"]
                    else "INCORRECT"
                )

                file.write(
                    f"\nImage index: {result['index']}\n"
                    f"Actual class: {actual}\n"
                    f"Predicted class: {predicted}\n"
                    f"Confidence: {result['confidence']:.4f}\n"
                    f"Prediction status: {status}\n"
                )

        file.write(
            "\n\nResearch note:\n"
            "Grad-CAM highlights image regions that contribute "
            "to a selected model prediction. These visualizations "
            "are qualitative explanations, not proof of causality. "
            "Confidence scores are not necessarily calibrated "
            "probabilities.\n"
        )

    print("Research report saved:", report_path)


# --------------------------------------------------
# 10. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    cnn, resnet18, resnet50 = load_models()

    all_results = {}

    print("\nGenerating CNN Grad-CAM...")

    all_results["Custom CNN"] = run_gradcam(
        cnn,
        cnn.features[8],
        cifar_transform,
        "Custom CNN"
    )

    print("\nGenerating ResNet-18 Grad-CAM...")

    all_results["ResNet-18"] = run_gradcam(
        resnet18,
        resnet18.layer4,
        cifar_transform,
        "ResNet-18"
    )

    print("\nGenerating ResNet-50 Grad-CAM...")

    all_results["Fine-tuned ResNet-50"] = run_gradcam(
        resnet50,
        resnet50.layer4,
        resnet50_transform,
        "Fine-tuned ResNet-50"
    )

    save_comparison(all_results)

    save_report(all_results)

    print("\n" + "=" * 55)

    print("GRAD-CAM ANALYSIS COMPLETED SUCCESSFULLY")

    print("=" * 55)
