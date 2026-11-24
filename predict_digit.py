
import torch
from torchvision import datasets, transforms
import torch.nn as nn

# Load the same model architecture
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# Load saved model
model = DigitClassifier()
model.load_state_dict(
    torch.load("mnist_model.pth", map_location="cpu")
)
model.eval()

# Load a test image
dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

image, actual_label = dataset[0]

# Make prediction
with torch.no_grad():
    output = model(image.unsqueeze(0))
    prediction = output.argmax(dim=1).item()

print("Actual digit:", actual_label)
print("Predicted digit:", prediction)

if prediction == actual_label:
    print("Prediction is correct!")
else:
    print("Prediction is incorrect.")
