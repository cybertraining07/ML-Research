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
