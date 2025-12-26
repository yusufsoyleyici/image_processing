import os
import struct
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

# ==========================================
# 1. HELPER FUNCTIONS TO LOAD DATA (OFFLINE)
# ==========================================
def load_images(path):
    with open(path, 'rb') as f:
        # Read magic number and dimensions
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the image data
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(path):
    with open(path, 'rb') as f:
        # Read magic number and count
        magic, num = struct.unpack(">II", f.read(8))
        # Read the labels
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# ==========================================
# 2. PATH SETUP & DATA LOADING
# ==========================================
# Ensure these paths match your folder structure exactly
train_img_path = os.path.join("MNIST-dataset", "train-images.idx3-ubyte")
train_label_path = os.path.join("MNIST-dataset", "train-labels.idx1-ubyte")
test_img_path = os.path.join("MNIST-dataset", "t10k-images.idx3-ubyte")
test_label_path = os.path.join("MNIST-dataset", "t10k-labels.idx1-ubyte")

print("Loading data...")
try:
    train_images = load_images(train_img_path)
    train_labels = load_labels(train_label_path)
    test_images = load_images(test_img_path)
    test_labels = load_labels(test_label_path)
except FileNotFoundError as e:
    print(f"\nERROR: Could not find the dataset files. \nCheck that the folder 'MNIST-dataset' exists and contains the files.\n{e}")
    exit()

# ==========================================
# 3. FEATURE EXTRACTION (HU MOMENTS)
# ==========================================
print("Extracting features (Hu Moments)... This may take a minute.")

train_huMoments = np.empty((len(train_images), 7))
test_huMoments = np.empty((len(test_images), 7))

# Calculate Hu Moments for Training Data
for train_idx, train_img in enumerate(train_images):
    train_moments = cv2.moments(train_img, True)
    # flattened to 1D array of 7 elements
    train_huMoments[train_idx] = cv2.HuMoments(train_moments).reshape(7)

# Calculate Hu Moments for Test Data
for test_idx, test_img in enumerate(test_images):
    test_moments = cv2.moments(test_img, True)
    test_huMoments[test_idx] = cv2.HuMoments(test_moments).reshape(7)

# Normalization (Standardization)
# We save mean and std to use them on the STM32 later
features_mean = np.mean(train_huMoments, axis=0)
features_std = np.std(train_huMoments, axis=0)

train_huMoments = (train_huMoments - features_mean) / features_std
test_huMoments = (test_huMoments - features_mean) / features_std

# ==========================================
# 4. PREPARE LABELS ("0" vs "Not 0")
# ==========================================
# The assignment asks for "Zero" vs "Not Zero" classification
# Create copies to avoid overwriting original data if needed
train_labels_binary = train_labels.copy()
test_labels_binary = test_labels.copy()

# Set all non-zero labels to 1
train_labels_binary[train_labels_binary != 0] = 1
test_labels_binary[test_labels_binary != 0] = 1

# ==========================================
# 5. MODEL DEFINITION & TRAINING
# ==========================================
print("Training Single Neuron Model...")

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[7], activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

history = model.fit(
    train_huMoments,
    train_labels_binary,
    batch_size=128,
    epochs=50,
    class_weight={0: 8, 1: 1}, # Give more weight to '0' since it's rare (1 vs 9)
    verbose=1
)

# ==========================================
# 6. EVALUATION
# ==========================================
print("Evaluating...")
perceptron_preds = model.predict(test_huMoments)
conf_matrix = confusion_matrix(test_labels_binary, perceptron_preds > 0.5)

# Plot Confusion Matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
cm_display.plot()
cm_display.ax_.set_title("Single Neuron Classifier Confusion Matrix")
plt.show()

# Save the Keras model file
model.save("mnist_single_neuron.h5")
print("Model saved to 'mnist_single_neuron.h5'")

# ==========================================
# 7. EXPORT WEIGHTS FOR STM32 (C CODE)
# ==========================================
print("\n" + "#"*50)
print("     COPY PASTE THE BELOW INTO YOUR C CODE     ")
print("#"*50 + "\n")

# A. Export Normalization Parameters
print("// 1. Normalization Parameters (Mean & Std)")
print(f"float features_mean[7] = {{ {', '.join([f'{x:.6f}f' for x in features_mean])} }};")
print(f"float features_std[7]  = {{ {', '.join([f'{x:.6f}f' for x in features_std])} }};")
print()

# B. Export Model Weights and Bias
weights, bias = model.layers[0].get_weights()
weights_flat = weights.flatten()

print("// 2. Neuron Weights & Bias")
print(f"float neuron_weights[7] = {{ {', '.join([f'{w:.6f}f' for w in weights_flat])} }};")
print(f"float neuron_bias       = {bias[0]:.6f}f;")

print("\n" + "#"*50)