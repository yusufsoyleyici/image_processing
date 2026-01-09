import tensorflow as tf
import numpy as np
import os
import struct

# --- CONFIGURATION ---
# PASTE YOUR DATASET PATH HERE (Same as train_digits.py)
DATASET_FOLDER = r"C:\Users\yyigi\Desktop\yusuf\ders\Introduction to Embedded Image Processing\EE4065_STM32_nucleo_F446RE-Image-Transfer-STM-to-PC-and-PC-to-STM\image_transfer_nucleof446_python_PC\MNIST-dataset"
INPUT_SHAPE = (32, 32, 3)

# --- 1. DATA LOADER (Needed for calibration) ---
def load_mnist_subset(path):
    images_path = os.path.join(path, 'train-images.idx3-ubyte')
    with open(images_path, 'rb') as imgpath:
        struct.unpack('>IIII', imgpath.read(16))
        # Read just 500 images for calibration (enough to learn ranges)
        images = np.frombuffer(imgpath.read(28*28*500), dtype=np.uint8)
        images = images.reshape(500, 28, 28)
    return images

def preprocess_subset(images):
    images = tf.expand_dims(images, axis=-1)
    images = tf.image.grayscale_to_rgb(images)
    images = tf.image.resize(images, [INPUT_SHAPE[0], INPUT_SHAPE[1]])
    images = images / 255.0
    return images

# --- 2. QUANTIZATION CONVERTER ---
print("Loading model for quantization...")
model = tf.keras.models.load_model("squeezenet_mnist.h5")

# Prepare calibration data generator
print("Loading calibration data...")
raw_images = load_mnist_subset(DATASET_FOLDER)
calib_images = preprocess_subset(raw_images)

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(calib_images).batch(1).take(100):
        yield [input_value]

print("Converting to INT8 TFLite (this shrinks size by 4x)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure operations are compatible with TFLite Micro
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set inputs/outputs to float32 (easier for STM32 C-code integration)
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model_quant = converter.convert()

# --- 3. SAVE ---
output_path = "squeezenet_quantized.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model_quant)

size_kb = os.path.getsize(output_path) / 1024
print(f"\nSUCCESS! Generated: {output_path}")
print(f"New Model Size: {size_kb:.2f} KB (Should be around 130 KB)")