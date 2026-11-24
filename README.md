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

## Project 6: CIFAR-10 Transfer Learning with ResNet-50

### Research Objective
Investigate transfer learning using pretrained ResNet-50
and compare frozen feature extraction with partial fine-tuning.

### Experiment 1: Frozen Feature Extraction
- Architecture: ResNet-50
- Pretrained weights: ImageNet V2
- Dataset: CIFAR-10
- Training images: 50,000
- Test images: 10,000
- Epochs: 3
- Frozen layers: All pretrained layers
- Trainable layer: Final classification layer
- Test accuracy: 82.12%

### Experiment 2: Partial Fine-Tuning
- Architecture: ResNet-50
- Pretrained weights: Previously trained Experiment 1 checkpoint
- Dataset: CIFAR-10
- Epochs: 2
- Trainable layers: Layer4 and final classification layer
- Test accuracy: 89.96%

### Experimental Comparison

| Model | Test Accuracy |
|---|---:|
| CIFAR-10 CNN | 72.40% |
| ResNet-18 | 88.39% |
| ResNet-50 Frozen | 82.12% |
| ResNet-50 Fine-Tuned | 89.96% |

### Key Findings
- Partial fine-tuning improved accuracy by 7.84 percentage points
  compared with frozen feature extraction.
- Fine-tuned ResNet-50 outperformed ResNet-18 by
  1.57 percentage points.
- Adapting pretrained features to the target dataset
  improved classification performance.

### Technologies
- Python
- PyTorch
- Torchvision
- ResNet-50
- ImageNet pretrained weights
- CIFAR-10
- Transfer Learning
- Deep Learning

### Skills Demonstrated
- Transfer learning
- Feature extraction
- Partial fine-tuning
- Neural network optimization
- Model evaluation
- Experimental comparison
- Research documentation

## Project 7: Advanced CIFAR-10 Model Evaluation and Research Analysis

### Objective
Conduct a comparative evaluation of three deep learning models trained or fine-tuned for CIFAR-10 image classification.

### Models Evaluated
1. **Custom CNN** — Convolutional neural network trained from scratch.
2. **ResNet-18** — Modified ResNet-18 architecture trained for CIFAR-10.
3. **ResNet-50 Fine-tuned** — ImageNet-pretrained ResNet-50 with partial fine-tuning.

### Evaluation Results

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 72.40% |
| ResNet-18 | 88.39% |
| ResNet-50 Fine-tuned | 89.96% |

### Evaluation Methods
- Classification reports and precision, recall, and F1-score analysis.
- Confusion matrix analysis.
- Per-class accuracy comparison.
- Correct and incorrect prediction visualization.
- Overall model accuracy comparison.
- Consolidated research summary.

### Generated Research Artifacts
- `cifar10_advanced_classification_reports.txt`
- `cifar10_three_model_confusion_matrices.png`
- `cifar10_three_model_per_class_accuracy.png`
- `cifar10_three_model_comparison.png`
- `cifar10_three_model_prediction_examples.png`
- `cifar10_advanced_research_summary.txt`

### Main Finding
The fine-tuned ResNet-50 achieved the highest test accuracy of 89.96%, followed by ResNet-18 at 88.39% and the custom CNN at 72.40%.

**Research note:** The models use different training approaches and input resolutions. The results therefore compare complete model pipelines rather than isolating architecture as the only variable.

### Reproducibility
The evaluation script is available as `cifar10_advanced_evaluation.py`. The pretrained and fine-tuned model checkpoints are retained locally and are not included in the Git repository due to their file sizes.

## Project 8: Model Explainability, Confidence and Uncertainty Analysis

### Overview
This project investigates model explainability, prediction confidence, probability calibration, and uncertainty using three CIFAR-10 image classification models.

### Models Evaluated
- Custom CNN
- ResNet-18
- Fine-tuned ResNet-50

### Research Techniques
- Grad-CAM visual explanations
- Confidence distribution analysis
- Reliability diagrams
- Expected Calibration Error (ECE)
- Predictive entropy
- Multiclass Brier score
- High-confidence error detection
- Low-confidence correct prediction analysis

### Experimental Results

| Metric | Custom CNN | ResNet-18 | Fine-tuned ResNet-50 |
|---|---:|---:|---:|
| Accuracy | 72.40% | 88.39% | 89.96% |
| Average Confidence | 68.75% | 90.32% | 92.48% |
| ECE | 0.036544 | 0.021931 | 0.025726 |
| Predictive Entropy | 0.868665 | 0.281236 | 0.212385 |
| Brier Score | 0.379192 | 0.170210 | 0.146634 |

### Key Findings
- Fine-tuned ResNet-50 achieved the highest classification accuracy at 89.96%.
- ResNet-18 achieved the lowest Expected Calibration Error at 0.021931.
- Fine-tuned ResNet-50 achieved the lowest multiclass Brier score at 0.146634.
- ResNet-50 produced the lowest mean predictive entropy.
- High-confidence classification errors demonstrate that strong model confidence does not always guarantee correct predictions.

### Generated Research Outputs

**Grad-CAM**
- `gradcam_results/cifar10_gradcam_three_model_comparison.png`
- `gradcam_results/cifar10_gradcam_research_report.txt`

**Confidence and Uncertainty**
- `confidence_analysis_results/cifar10_confidence_distributions.png`
- `confidence_analysis_results/cifar10_reliability_diagram.png`
- `confidence_analysis_results/cifar10_uncertainty_comparison.png`
- `confidence_analysis_results/cifar10_high_confidence_errors.csv`
- `confidence_analysis_results/cifar10_confidence_research_report.txt`

### Research Conclusion
This experiment demonstrates that classification accuracy alone is insufficient for evaluating machine learning reliability. Model explainability, probability calibration, confidence, and uncertainty provide additional information about prediction behavior.

All measurements were obtained using the CIFAR-10 test set and do not guarantee performance on unseen or out-of-distribution data.

## Project 9: Adversarial Robustness and Model Security Analysis

### Overview
Investigated the vulnerability of CIFAR-10 image classification models to adversarial attacks using Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD).

The objective was to measure how small, deliberately crafted image perturbations affect model predictions and compare the adversarial robustness of three trained models.

### Models Evaluated
- Custom CNN
- ResNet-18
- Fine-tuned ResNet-50

### Attack Methods

**FGSM (Fast Gradient Sign Method)**
- White-box gradient-based attack.
- Perturbation strengths: 4/255, 8/255 and 16/255.

**PGD (Projected Gradient Descent)**
- Iterative gradient-based attack.
- Perturbation strength: 8/255.
- Step size: 2/255.
- Iterations: 10.
- Random initialization.

### Evaluation Configuration
- Dataset: CIFAR-10
- Evaluation sample: 200 balanced images.
- Sampling: First 20 test images from each of the 10 classes.
- Evaluation device: CPU.
- Metrics: Clean accuracy, adversarial accuracy and attack success rate.

### Experimental Findings

The following results were obtained using FGSM at 8/255 and PGD at 8/255.

| Model | Clean Accuracy | FGSM Accuracy | PGD Accuracy |
|---|---:|---:|---:|
| Custom CNN | 75.00% | 3.00% | 0.50% |
| ResNet-18 | 90.00% | 5.50% | 0.00% |
| Fine-tuned ResNet-50 | 91.00% | 37.00% | 0.00% |

### Key Findings
- All three models experienced substantial accuracy reductions under adversarial attacks.
- Fine-tuned ResNet-50 retained the highest FGSM adversarial accuracy in this experiment.
- PGD was particularly effective against the tested models.
- High clean accuracy did not necessarily indicate strong adversarial robustness.

### Generated Research Artifacts
- `adversarial_robustness_results/cifar10_adversarial_results.csv`
- `adversarial_robustness_results/cifar10_adversarial_accuracy_comparison.png`
- `adversarial_robustness_results/cifar10_adversarial_examples.png`
- `adversarial_robustness_results/cifar10_adversarial_research_report.txt`

### Research Skills Demonstrated
- Adversarial machine learning
- FGSM and PGD implementation
- White-box security testing
- Model vulnerability assessment
- Adversarial accuracy measurement
- Attack success rate analysis
- Research documentation and visualization

### Limitations
The evaluation used a balanced 200-image test sample and a limited set of attack configurations. Results demonstrate empirical robustness under the tested conditions, not universal protection against adversarial attacks.

---

## Project 10: Adversarial Defense and Robust Model Training

### Overview
Investigated adversarial defense strategies for CIFAR-10 image classification models using adversarial training.

The project explored whether training with adversarial examples could improve model resistance to FGSM and PGD attacks while preserving clean classification accuracy.

### Research Objectives
- Implement adversarial-only training.
- Investigate mixed clean and adversarial training.
- Compare different defense strategies.
- Measure clean and adversarial classification accuracy.
- Analyze the trade-off between model accuracy and robustness.
- Evaluate defended models using a larger test sample.

### Defense Approaches

**Experiment 1: FGSM Adversarial Training**
- Model: Custom CIFAR-10 CNN.
- Training samples: 10,000.
- Epochs: 2.
- FGSM perturbation: 8/255.
- Training approach: Adversarial examples only.

**Experiment 2: PGD Adversarial Training**
- Model: Custom CIFAR-10 CNN.
- Training samples: 10,000.
- Epochs: 2.
- PGD step size: 2/255.
- Training iterations: 3.
- Training approach: Adversarial examples only.

**Experiment 3: Mixed Clean and FGSM Adversarial Training**
- Model: Custom CIFAR-10 CNN.
- Training samples: 10,000.
- Epochs: 5.
- FGSM perturbation: 8/255.
- Clean/adversarial loss ratio: 50/50.
- Optimizer: Adam.
- Learning rate: 0.0002.

The mixed training objective combined clean and adversarial classification losses with equal weighting.

### Initial Defense Evaluation

The first comparison used a balanced 200-image test sample.

| Model | Clean Accuracy | FGSM Accuracy | PGD Accuracy |
|---|---:|---:|---:|
| Baseline CNN | 75.00% | 3.00% | 0.50% |
| FGSM-only Defense | 49.00% | 20.00% | 11.00% |
| PGD-only Defense | 51.00% | 17.00% | 10.50% |
| Mixed FGSM Defense | 67.50% | 14.50% | 5.50% |

### Full Defense Evaluation

A further evaluation compared all four models using a larger CIFAR-10 test sample.

**Evaluation configuration**
- Clean evaluation: 10,000 test images.
- FGSM evaluation: 10,000 test images.
- PGD evaluation: 2,000 balanced test images.
- PGD sampling: 200 images per class.
- FGSM perturbation: 8/255.
- PGD perturbation: 8/255.
- PGD step size: 2/255.
- PGD iterations: 10.

### Final Evaluation Results

| Model | Clean Accuracy | FGSM Accuracy | FGSM ASR | PGD Accuracy | PGD ASR |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 72.40% | 3.39% | 95.37% | 0.05% | 99.93% |
| FGSM Defense | 46.50% | 20.72% | 62.17% | 12.65% | 78.76% |
| PGD Defense | 48.85% | 19.12% | 63.36% | 12.15% | 77.51% |
| Mixed FGSM Defense | 66.60% | 20.20% | 69.95% | 8.90% | 86.85% |

ASR represents the percentage of initially correctly classified images that became incorrectly classified after an attack.

### Key Research Findings

1. The baseline CNN achieved 72.40% clean accuracy but was highly vulnerable to the tested adversarial attacks.

2. FGSM adversarial training achieved the highest measured adversarial accuracy, reaching 20.72% under FGSM and 12.65% under PGD.

3. PGD adversarial training improved robustness compared with the baseline, although its measured accuracy was slightly below the FGSM defense under these evaluation settings.

4. Mixed adversarial training preserved more clean accuracy, achieving 66.60%, compared with 46.50% for FGSM-only defense and 48.85% for PGD-only defense.

5. The experiments demonstrate a trade-off between clean classification performance and adversarial robustness.

### Generated Research Artifacts

**Adversarial-only training**
- `adversarial_defense_results/cifar10_adversarial_defense_results.csv`
- `adversarial_defense_results/cifar10_adversarial_training_history.txt`
- `adversarial_defense_results/cifar10_adversarial_defense_comparison.png`
- `adversarial_defense_results/cifar10_adversarial_defense_report.txt`

**Mixed adversarial training**
- `mixed_adversarial_results/cifar10_mixed_adversarial_results.csv`
- `mixed_adversarial_results/cifar10_mixed_training_history.txt`
- `mixed_adversarial_results/cifar10_mixed_adversarial_report.txt`
- `mixed_adversarial_results/cifar10_mixed_training_accuracy.png`

**Full defense evaluation**
- `full_defense_evaluation/cifar10_full_defense_results.csv`
- `full_defense_evaluation/cifar10_full_defense_comparison.png`
- `full_defense_evaluation/cifar10_full_defense_report.txt`

### Research Skills Demonstrated
- Adversarial training
- Robust machine learning
- FGSM and PGD attack evaluation
- Model defense implementation
- Clean and adversarial performance analysis
- Experimental comparison
- Attack success rate calculation
- Research reporting and visualization

### Limitations
- Adversarial training used a 10,000-image subset of the training dataset.
- PGD evaluation used a balanced 2,000-image test subset.
- Clean and FGSM evaluations used the full 10,000-image test set.
- Each model was evaluated from one training run.
- Only selected adversarial attacks and configurations were investigated.
- Empirical robustness does not establish universal protection.

### Future Research Improvements
- Evaluate defenses using multiple random seeds.
- Investigate stronger adversarial attacks.
- Experiment with longer training schedules.
- Evaluate additional defense techniques.
- Investigate adaptive attacks against the defended models.
- Perform full-test-set PGD evaluation.
- Compare robustness across additional neural network architectures.

### Overall Conclusion

This project demonstrated the implementation and evaluation of adversarial defense strategies for CIFAR-10 image classification.

The results showed that adversarial training improved resistance to the tested attacks compared with the original baseline model. FGSM-only training achieved the strongest measured adversarial accuracy, while mixed training preserved more clean classification performance.

The experiments highlight the importance of evaluating both ordinary classification accuracy and adversarial robustness when developing reliable machine learning systems.

---

## Project 11 — Responsible AI, Model Interpretability and Class-wise Bias Analysis

### Project Overview

This project evaluates three CIFAR-10 image classification models from a Responsible AI perspective, focusing on class-wise performance differences, prediction reliability, confidence, and model evaluation transparency.

The objective was to investigate whether models perform consistently across different image categories and how reliably their predicted probabilities correspond to observed outcomes.

**Models evaluated:**
- Custom CNN
- ResNet-18
- Fine-tuned ResNet-50

**Dataset:** CIFAR-10  
**Evaluation dataset:** Full 10,000-image CIFAR-10 test set  
**Framework:** PyTorch  
**Evaluation tools:** Scikit-learn, NumPy, Matplotlib

### Research Objectives

- Evaluate class-wise classification accuracy, precision, recall, and F1-score.
- Identify the strongest and weakest-performing image categories.
- Analyze class-wise prediction reliability.
- Calculate one-vs-rest Expected Calibration Error (ECE).
- Calculate one-vs-rest Brier scores.
- Compare confusion matrices across architectures.
- Investigate performance differences between model architectures.
- Document limitations and Responsible AI considerations.

### Key Results

| Metric | Custom CNN | ResNet-18 | Fine-tuned ResNet-50 |
|---|---:|---:|---:|
| Overall accuracy | 72.40% | 88.39% | 89.96% |
| Macro F1-score | 72.24% | 88.36% | 89.92% |
| Mean class accuracy | 72.40% | 88.39% | 89.96% |
| Mean one-vs-rest ECE | 0.017528 | 0.005938 | 0.007806 |
| Mean one-vs-rest Brier | 0.037919 | 0.017021 | 0.014664 |
| Weakest class | Bird | Cat | Cat |
| Weakest class accuracy | 43.70% | 75.20% | 80.60% |

### Major Research Findings

**1. Overall predictive performance**

Fine-tuned ResNet-50 achieved the highest overall accuracy at 89.96%, followed by ResNet-18 at 88.39%. Both significantly outperformed the Custom CNN at 72.40%.

**2. Class-wise performance differences**

Fine-tuned ResNet-50 achieved the highest class-wise accuracy in eight of the ten CIFAR-10 categories. ResNet-18 performed better on airplane and automobile.

The Custom CNN showed a wider performance gap between its strongest and weakest classes, with bird classification achieving only 43.70% accuracy.

**3. Probability calibration**

ResNet-18 achieved the lowest mean one-vs-rest ECE of 0.005938, indicating the smallest average calibration error under the selected 15-bin evaluation method.

**4. Brier score comparison**

Fine-tuned ResNet-50 achieved the lowest mean one-vs-rest Brier score of 0.014664. This indicates the strongest average probability prediction performance under the class-wise Brier evaluation used in this experiment.

**5. Responsible AI interpretation**

The experiment identifies differences in model performance across image categories. These differences are useful for investigating model reliability, but CIFAR-10 image categories are not demographic or protected groups. Therefore, this experiment does not establish demographic fairness or societal bias.

### Generated Research Artifacts

**Evaluation script**
- `cifar10_responsible_ai_analysis.py`

**Visualizations**
- `responsible_ai_results/cifar10_classwise_accuracy.png`
- `responsible_ai_results/cifar10_classwise_calibration.png`
- `responsible_ai_results/cifar10_macro_f1_comparison.png`

**Reports and datasets**
- `responsible_ai_results/cifar10_classwise_results.csv`
- `responsible_ai_results/cifar10_responsible_ai_summary.csv`
- `responsible_ai_results/cifar10_responsible_ai_report.txt`

**Confusion matrices**
- `responsible_ai_results/custom_cnn_confusion_matrix.csv`
- `responsible_ai_results/resnet_18_confusion_matrix.csv`
- `responsible_ai_results/fine_tuned_resnet_50_confusion_matrix.csv`

### Research Skills Demonstrated

- Responsible AI evaluation
- Class-wise model performance analysis
- Model interpretability concepts
- Classification metrics and confusion matrices
- Probability calibration analysis
- Expected Calibration Error (ECE)
- Brier score evaluation
- Model reliability assessment
- Python and PyTorch research implementation
- Data visualization and research reporting
- Experimental methodology and documentation

### Limitations

- Evaluation was conducted using the CIFAR-10 test distribution.
- Only one trained checkpoint per architecture was evaluated.
- Results may vary across independent training runs and random seeds.
- Calibration measurements depend on the selected binning method.
- Class-wise performance differences do not establish their underlying causes.
- No out-of-distribution or real-world deployment evaluation was conducted.
- CIFAR-10 categories do not provide demographic attributes for protected-group fairness analysis.
- No causal fairness conclusions can be drawn from these experiments.

### Future Research Improvements

- Repeat experiments using multiple random seeds.
- Investigate distribution shifts and out-of-distribution performance.
- Apply temperature scaling and compare calibration before and after adjustment.
- Investigate class-specific data augmentation.
- Explore additional interpretability techniques.
- Extend fairness evaluation using datasets with appropriate demographic or group attributes and valid evaluation methods.

### Overall Conclusion

This project demonstrated a practical approach to evaluating machine learning models beyond overall classification accuracy.

The results showed that Fine-tuned ResNet-50 achieved the strongest overall predictive performance and lowest mean one-vs-rest Brier score, while ResNet-18 achieved the lowest mean one-vs-rest ECE.

The class-wise analysis also revealed that model performance varies across image categories, highlighting the importance of evaluating reliability and individual class performance rather than relying exclusively on overall accuracy.

This research contributes to a broader understanding of model evaluation, interpretability, reliability, and responsible machine learning development.

