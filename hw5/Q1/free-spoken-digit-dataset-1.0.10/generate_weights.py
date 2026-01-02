import tensorflow as tf
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_PATH = 'mlp_fsdd_model.h5'
OUTPUT_FILE = 'weights_data.h'
# ---------------------

def write_c_array(f, name, data):
    f.write(f"const float {name}[] = {{\n")
    # Flatten array
    flat_data = data.flatten()
    for i, val in enumerate(flat_data):
        f.write(f"    {val:.8f}f,")
        if (i + 1) % 10 == 0:
            f.write("\n")
    f.write("\n};\n\n")

# Load model
if not os.path.exists(MODEL_PATH):
    print(f"Error: Could not find {MODEL_PATH}")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

with open(OUTPUT_FILE, "w") as f:
    f.write("#ifndef WEIGHTS_DATA_H\n")
    f.write("#define WEIGHTS_DATA_H\n\n")

    # Layer 1
    weights1, bias1 = model.layers[0].get_weights()
    # Transpose weights (Input x Output -> Output x Input) for standard C loops
    write_c_array(f, "w1", weights1.T) 
    write_c_array(f, "b1", bias1)

    # Layer 2
    weights2, bias2 = model.layers[1].get_weights()
    write_c_array(f, "w2", weights2.T)
    write_c_array(f, "b2", bias2)

    # Layer 3 (Output)
    weights3, bias3 = model.layers[2].get_weights()
    write_c_array(f, "w3", weights3.T)
    write_c_array(f, "b3", bias3)

    f.write("#endif\n")

print(f"Successfully generated {OUTPUT_FILE}")