import os
import struct
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

# ==========================================
# 1. HELPER FUNCTIONS & SETUP
# ==========================================
def load_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# Ensure these match your folder structure
train_img_path = os.path.join("MNIST-dataset", "train-images.idx3-ubyte")
train_label_path = os.path.join("MNIST-dataset", "train-labels.idx1-ubyte")
test_img_path = os.path.join("MNIST-dataset", "t10k-images.idx3-ubyte")
test_label_path = os.path.join("MNIST-dataset", "t10k-labels.idx1-ubyte")

# ==========================================
# 2. FEATURE EXTRACTION (HU MOMENTS)
# ==========================================
print("Loading Data and Extracting Features...")

# Load Data
try:
    train_images = load_images(train_img_path)
    train_labels = load_labels(train_label_path)
    test_images = load_images(test_img_path)
    test_labels = load_labels(test_label_path)
except FileNotFoundError:
    print("Error: MNIST files not found. Check your folder path.")
    exit()

# Extract Hu Moments
train_huMoments = np.empty((len(train_images), 7))
test_huMoments = np.empty((len(test_images), 7))

for i, img in enumerate(train_images):
    moments = cv2.moments(img, True)
    train_huMoments[i] = cv2.HuMoments(moments).reshape(7)

for i, img in enumerate(test_images):
    moments = cv2.moments(img, True)
    test_huMoments[i] = cv2.HuMoments(moments).reshape(7)

# Normalization (Important for Neural Networks!)
# We reuse the logic from Q1 as it ensures the network trains correctly
features_mean = np.mean(train_huMoments, axis=0)
features_std = np.std(train_huMoments, axis=0)

train_huMoments = (train_huMoments - features_mean) / features_std
test_huMoments = (test_huMoments - features_mean) / features_std

# ==========================================
# 3. DEFINE MULTILAYER PERCEPTRON (MLP)
# ==========================================
print("Building and Training Model (Section 11.8)...")

model = keras.models.Sequential([
    # Layer 1: 100 Neurons, ReLU
    keras.layers.Dense(100, input_shape=[7], activation="relu", name="dense_1"),
    # Layer 2: 100 Neurons, ReLU
    keras.layers.Dense(100, activation="relu", name="dense_2"),
    # Layer 3: 10 Output Neurons (0-9), Softmax
    keras.layers.Dense(10, activation="softmax", name="output")
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# Callbacks
mc_callback = keras.callbacks.ModelCheckpoint("mlp_mnist_model.h5", save_best_only=True)
es_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=5)

# Train
model.fit(
    train_huMoments, train_labels,
    epochs=100, # 100 is usually enough for convergence
    verbose=1,
    callbacks=[mc_callback, es_callback]
)

# ==========================================
# 4. EXPORT WEIGHTS TO C HEADER FILE
# ==========================================
print("\nExporting weights to 'weights_data.h'...")

def write_layer(f, layer, layer_num):
    weights, bias = layer.get_weights()
    # Weights shape: (Inputs, Neurons) -> Flatten for C (Row-major)
    # Note: Keras stores weights as [Input][Neuron]. We usually want [Neuron][Input] for dot product?
    # Actually, for standard loops, we can keep it flat.
    w_flat = weights.flatten()
    
    f.write(f"// Layer {layer_num} Weights ({weights.shape[0]}x{weights.shape[1]})\n")
    f.write(f"const float w{layer_num}[{len(w_flat)}] = {{\n")
    for i, val in enumerate(w_flat):
        f.write(f"{val:.6f}f, ")
        if (i+1) % 10 == 0: f.write("\n")
    f.write("};\n\n")

    f.write(f"// Layer {layer_num} Bias ({len(bias)})\n")
    f.write(f"const float b{layer_num}[{len(bias)}] = {{\n")
    for i, val in enumerate(bias):
        f.write(f"{val:.6f}f, ")
    f.write("};\n\n")

with open("weights_data.h", "w") as f:
    f.write("#ifndef WEIGHTS_DATA_H\n#define WEIGHTS_DATA_H\n\n")
    
    # Write Normalization params
    f.write("// Normalization Parameters\n")
    f.write(f"const float features_mean[7] = {{ {', '.join([f'{x:.6f}f' for x in features_mean])} }};\n")
    f.write(f"const float features_std[7]  = {{ {', '.join([f'{x:.6f}f' for x in features_std])} }};\n\n")
    
    # Write Layers
    write_layer(f, model.layers[0], 1)
    write_layer(f, model.layers[1], 2)
    write_layer(f, model.layers[2], 3)
    
    f.write("#endif\n")

print("Done! Copy 'weights_data.h' to your STM32 project folder.")

# Evaluate
nn_preds = model.predict(test_huMoments)
predicted_classes = np.argmax(nn_preds, axis=1)
conf_matrix = confusion_matrix(test_labels, predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
cm_display.plot()
cm_display.ax_.set_title("Neural Network Confusion Matrix")
plt.show()