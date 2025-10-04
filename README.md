#Training Hyperparameters:
Initial Phase:
Learning Rate: 0.001 with exponential decay (decay_rate=0.96, decay_steps=1000)
Justification: Good starting point with adaptive decay. 

Batch Size: 64
Justification: Balanced choice. 

Epochs: 1000 (with EarlyStopping patience=30)

Fine-Tuning Phase:
Learning Rate: 1e-5 for last 20 layers
Justification: Conservative rate prevents overwriting useful features.

Regularization:
Data Augmentation: Horizontal flips, ±10% rotation/zoom/translation, Gaussian noise


Dropout: 0.2 
A moderate rate that:
Reduces overfitting without severely limiting capacity.
Works well empirically for many architectures (studies often use 0.2–0.5).


L2 Weight Regularization: 1e-4
Justification: Standard strength. 

L1 Bias Regularization: 1e-6


Loss & Metrics:
Focal Loss: gamma=2.0, alpha=0.25
Justification: Good for class imbalance. 
Custom Metrics: Balanced Accuracy + F1 Score
Justification: Essential for imbalanced data. 
#  Pneumonia Detection with ResNet-50 & Focal Loss

This project fine-tunes a ResNet-50 model to detect pneumonia using chest X-ray images from the PneumoniaMNIST dataset. It includes data augmentation, custom loss (Focal Loss), and evaluation using custom metrics (Balanced Accuracy, F1 Score, AUC).

---



