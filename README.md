\# Handwritten Digit Recognition Using PyTorch



\*\*Author:\*\* Mukaila Owowole  

\*\*Project Type:\*\* Machine Learning / Deep Learning  

\*\*Status:\*\* Initial implementation completed



\## 1. Project Overview



This project demonstrates the development of a handwritten digit recognition system using Python, PyTorch, and the MNIST dataset.



The objective was to build and train a neural network that can recognize handwritten digits from 0 to 9.



The project covers the machine-learning workflow, including dataset preparation, model development, training, evaluation, model saving, and prediction.



\## 2. Project Objectives



\- Develop a neural network using PyTorch.

\- Load and preprocess the MNIST dataset.

\- Train a model using supervised learning.

\- Evaluate model performance on test data.

\- Save the trained model.

\- Reload the model and perform predictions.



\## 3. Technologies and Tools



| Technology | Purpose |

|---|---|

| Python | Programming language |

| PyTorch | Deep learning framework |

| Torchvision | Dataset loading and image transformations |

| MNIST | Handwritten digit dataset |

| PowerShell | Command-line environment |

| Windows | Operating system |



\## 4. Dataset Description



The project uses the MNIST handwritten digit dataset.



Dataset characteristics:



\- 60,000 training images.

\- 10,000 test images.

\- Image dimensions: 28 × 28 pixels.

\- Classification categories: 0–9.



Images were converted into tensors using `transforms.ToTensor()`.



\## 5. Model Architecture



A feedforward neural network was developed using PyTorch.



The model contains:



1\. Flatten layer: converts each 28 × 28 image into 784 features.

2\. Fully connected layer: 784 input features and 128 neurons.

3\. ReLU activation function.

4\. Output layer: 10 neurons representing digits 0–9.



\## 6. Training Configuration



| Parameter | Value |

|---|---|

| Framework | PyTorch |

| Optimizer | Adam |

| Learning rate | 0.001 |

| Loss function | CrossEntropyLoss |

| Batch size | 64 |

| Training epochs | 3 |

| Computing device | CPU |



\## 7. Training Results



The model was trained for three epochs.



| Epoch | Training Loss |

|---|---:|

| 1 | 0.3433 |

| 2 | 0.1566 |

| 3 | 0.1108 |



The training loss decreased across all three epochs, indicating that the model learned from the training dataset.



\## 8. Model Evaluation



The trained model was evaluated using the MNIST test dataset.



\*\*Final test accuracy: 96.84%\*\*



The result indicates that the model correctly classified approximately 96.84% of the test images.



\## 9. Prediction Experiment



A separate Python script was created to load the saved model and perform inference.



The first test image produced the following result:



| Metric | Result |

|---|---|

| Actual digit | 7 |

| Predicted digit | 7 |

| Prediction status | Correct |



The model successfully loaded its saved parameters and predicted the correct digit.



\## 10. Model Saving



The trained model parameters were saved in:



`mnist\_model.pth`



The prediction script successfully loaded the saved parameters.



\## 11. Project Files



\- `first\_ml\_model.py` — model development and training.

\- `predict\_digit.py` — model loading and prediction.

\- `mnist\_model.pth` — saved model parameters.

\- `data/` — MNIST dataset.

\- `README.md` — project documentation.



\## 12. Skills Practised



\- Python programming.

\- PyTorch fundamentals.

\- Neural network development.

\- Supervised machine learning.

\- Image preprocessing.

\- Model training and evaluation.

\- Loss functions and optimization.

\- Model saving and loading.

\- Machine-learning inference.



\## 13. Future Improvements



Planned improvements include:



\- Building a Convolutional Neural Network (CNN).

\- Improving classification accuracy.

\- Experimenting with different learning rates.

\- Comparing neural network architectures.

\- Testing custom handwritten digit images.

\- Measuring model inference performance.



\## 14. Conclusion



This project provided practical experience with the machine-learning development lifecycle.



The neural network achieved 96.84% accuracy on the MNIST test

## Experiment 2: Convolutional Neural Network (CNN)

A second MNIST classifier was developed using a Convolutional Neural Network (CNN) with PyTorch.

### CNN Architecture
- Convolutional layer: 1 input channel, 32 output channels
- ReLU activation
- Max pooling
- Second convolutional layer: 32 to 64 channels
- Second ReLU activation and max pooling
- Fully connected layer: 128 neurons
- Dropout: 0.25
- Output layer: 10 digit classes

### Training Configuration
- Dataset: MNIST
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss
- Batch size: 64
- Epochs: 5
- Hardware: CPU

### Results

| Metric | Result |
|---|---:|
| Correct predictions | 9,911 |
| Test images | 10,000 |
| Test accuracy | 99.11% |
| Final training loss | 0.0299 |

### Model Comparison

| Model | Test Accuracy |
|---|---:|
| Feedforward Neural Network | 96.84% |
| Convolutional Neural Network | 99.11% |

The CNN achieved a 2.27 percentage-point improvement over the original feedforward model.

Saved model: `cnn_mnist_model.pth`


## Project 3: CIFAR-10 Image Classification

### Overview
Developed a Convolutional Neural Network (CNN) using PyTorch to classify colour images into 10 categories using the CIFAR-10 dataset.

### Dataset
- 50,000 training images
- 10,000 testing images
- Image dimensions: 32 × 32 pixels
- Image channels: RGB
- 10 object categories

### Model Architecture
- Convolutional layers: 32, 64 and 128 filters
- Batch normalization
- ReLU activation
- Max pooling
- Fully connected layer: 256 neurons
- Dropout: 0.4
- Output: 10 classes

### Training Configuration
- Framework: PyTorch
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss
- Batch size: 128
- Epochs: 5
- Device: CPU
- Data augmentation: Random crop and horizontal flip

### Training Results

| Epoch | Training Loss |
|---|---:|
| 1 | 1.5208 |
| 2 | 1.1907 |
| 3 | 1.0648 |
| 4 | 0.9860 |
| 5 | 0.9356 |

### Evaluation Results

- Test images: 10,000
- Correct predictions: 7,240
- Test accuracy: 72.40%

### Per-Class Accuracy

| Class | Accuracy |
|---|---:|
| Airplane | 80.80% |
| Automobile | 86.50% |
| Bird | 43.70% |
| Cat | 58.60% |
| Deer | 70.60% |
| Dog | 57.90% |
| Frog | 78.40% |
| Horse | 88.50% |
| Ship | 81.30% |
| Truck | 77.70% |

### Model File
`cifar10_model.pth`

### Future Improvements
- Train for more epochs.
- Experiment with learning-rate scheduling.
- Improve animal classification performance.
- Evaluate deeper CNN architectures.
- Generate a confusion matrix.
- Compare precision, recall and F1-score.


## Project 4: CIFAR-10 ResNet-18 Image Classification

### Overview
Improved the original CIFAR-10 image classifier by implementing ResNet-18 using PyTorch.

The experiment evaluates a deeper residual neural network against the original CNN.

### Model Architecture
- ResNet-18
- Modified first convolution for 32 × 32 RGB images
- Removed initial max pooling
- Final fully connected layer with 10 output classes
- Residual connections

### Training Configuration
- Framework: PyTorch
- Dataset: CIFAR-10
- Optimizer: SGD
- Learning rate: 0.1
- Momentum: 0.9
- Weight decay: 0.0005
- Scheduler: CosineAnnealingLR
- Batch size: 128
- Epochs: 10
- Device: CPU
- Data augmentation: Random crop and horizontal flip

### Training Loss

| Epoch | Training Loss |
|---|---:|
| 1 | 1.9960 |
| 2 | 1.4222 |
| 3 | 1.1236 |
| 4 | 0.8799 |
| 5 | 0.7033 |
| 6 | 0.5737 |
| 7 | 0.4813 |
| 8 | 0.3978 |
| 9 | 0.3214 |
| 10 | 0.2761 |

### Evaluation Results

- Test images: 10,000
- Correct predictions: 8,839
- Test accuracy: 88.39%
- Macro precision: 88.36%
- Macro recall: 88.39%
- Macro F1-score: 88.36%

### Original CNN vs ResNet-18

| Model | Test Accuracy |
|---|---:|
| Original CNN | 72.40% |
| ResNet-18 | 88.39% |

Improvement: 15.99 percentage points.

### Per-Class F1-Score

| Class | F1-score |
|---|---:|
| Airplane | 89.81% |
| Automobile | 95.57% |
| Bird | 83.63% |
| Cat | 76.50% |
| Deer | 87.31% |
| Dog | 81.89% |
| Frog | 90.04% |
| Horse | 91.34% |
| Ship | 93.59% |
| Truck | 93.96% |

### Model File
`cifar10_resnet18.pth`

### Future Improvements
- Experiment with longer training schedules.
- Investigate cat and dog classification errors.
- Generate visual confusion matrix.
- Compare additional architectures.
- Evaluate model generalization on unseen data.

## Project 5: CIFAR-10 Research Evaluation and Visualization

### Objective
Evaluate the trained CIFAR-10 ResNet-18 model using
classification metrics and visualizations.

### Evaluation Results
- Dataset: CIFAR-10
- Test images: 10,000
- Model: ResNet-18
- Overall accuracy: 88.39%
- Macro F1-score: 88.36%
- Weighted F1-score: 88.36%

### Evaluation Artifacts
The evaluation script generates:

1. classification_report.txt
2. cifar10_confusion_matrix.png
3. cifar10_prediction_examples.png
4. cifar10_model_comparison.png

### Key Findings
- Automobile achieved the highest F1-score (95.57%).
- Cat achieved the lowest F1-score (76.50%).
- The ResNet-18 model improved accuracy by
  15.99 percentage points compared with the original CNN.

### Tools and Technologies
- Python
- PyTorch
- Torchvision
- Scikit-learn
- Matplotlib
- NumPy

### Research Skills Demonstrated
- Deep learning model evaluation
- Image classification
- Performance measurement
- Confusion matrix analysis
- Precision, recall and F1-score interpretation
- Research visualization
