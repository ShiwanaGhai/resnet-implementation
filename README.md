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
# 🧠 Pneumonia Detection with ResNet-50 & Focal Loss

This project fine-tunes a ResNet-50 model to detect pneumonia using chest X-ray images from the PneumoniaMNIST dataset. It includes data augmentation, custom loss (Focal Loss), and evaluation using custom metrics (Balanced Accuracy, F1 Score, AUC).

---

## 📦 Setup

Make sure you have Python 3.7+ and install the required libraries:

```bash
pip install -r requirements.txt
pip install tensorflow numpy matplotlib kagglehub
import kagglehub
path = kagglehub.dataset_download("rijulshr/pneumoniamnist")
import numpy as np
dataset = np.load("pneumoniamnist.npz")
xtrain, ytrain = dataset["train_images"], dataset["train_labels"]
xvalid, yvalid = dataset["val_images"], dataset["val_labels"]
xtest, ytest = dataset["test_images"], dataset["test_labels"]
xtrain = xtrain[..., None] / 255.0
xvalid = xvalid[..., None] / 255.0
xtest = xtest[..., None] / 255.0
history = model.fit(
    xtrain, ytrain,
    validation_data=(xvalid, yvalid),
    epochs=1000,
    callbacks=[checkpoint, restore_best_weights],
    verbose=1
)
for layer in base_model.layers[-20:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=binary_focal_loss(gamma=2.0, alpha=0.25),
    metrics=[BalancedAccuracy(), tf.keras.metrics.AUC(), F1Score()]
)

fine_tune_history = model.fit(
    xtrain, ytrain,
    validation_data=(xvalid, yvalid),
    epochs=30,
    callbacks=[checkpoint, restore_best_weights],
    verbose=1
)
model.load_weights('/kaggle/working/best_model.weights.h5')
model.evaluate(xtest, ytest)

