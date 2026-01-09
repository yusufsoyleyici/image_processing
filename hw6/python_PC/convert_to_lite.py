import tensorflow as tf
import numpy as np

# 1. Load your existing model
print("Loading model...")
model = tf.keras.models.load_model("squeezenet_mnist.h5")

# 2. Setup the Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- KEY STEP: OPTIMIZATION (Quantization) ---
# This tells TensorFlow to shrink weights to be 4x smaller
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 3. Convert
print("Converting to TFLite...")
tflite_model = converter.convert()

# 4. Save the file
output_path = "squeezenet_mnist.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"SUCCESS: Generated {output_path}")
print("Please load THIS file into STM32CubeMX instead of the .h5 file.")