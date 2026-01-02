import numpy as np
import tensorflow as tf
import os

def save_header(filename, model):
    with open(filename, 'w') as f:
        f.write("/* Generated weights and biases for Keyword Spotting */\n")
        f.write("#ifndef WEIGHTS_DATA_H_\n")
        f.write("#define WEIGHTS_DATA_H_\n\n")
        f.write("#include <stdint.h>\n\n")

        # Layer 1: Dense (100 neurons)
        # Weights shape: (26, 100) -> Flattened for C
        w1, b1 = model.layers[0].get_weights()
        f.write(f"// Layer 1: Input (26) -> Dense (100)\n")
        f.write(f"const float layer1_weights[{w1.size}] = {{\n")
        # Transpose might be needed depending on C matrix mul implementation, 
        # but standard dense implementation usually expects Row-Major or Col-Major.
        # We will write it flat.
        w1_flat = w1.flatten() 
        for i, val in enumerate(w1_flat):
            f.write(f"{val:.8f}f, ")
            if (i+1) % 10 == 0: f.write("\n")
        f.write("};\n\n")

        f.write(f"const float layer1_biases[{b1.size}] = {{\n")
        for i, val in enumerate(b1):
            f.write(f"{val:.8f}f, ")
            if (i+1) % 10 == 0: f.write("\n")
        f.write("};\n\n")

        # Layer 2: Dense (100 neurons)
        w2, b2 = model.layers[1].get_weights()
        f.write(f"// Layer 2: Dense (100) -> Dense (100)\n")
        f.write(f"const float layer2_weights[{w2.size}] = {{\n")
        w2_flat = w2.flatten()
        for i, val in enumerate(w2_flat):
            f.write(f"{val:.8f}f, ")
            if (i+1) % 10 == 0: f.write("\n")
        f.write("};\n\n")

        f.write(f"const float layer2_biases[{b2.size}] = {{\n")
        for i, val in enumerate(b2):
            f.write(f"{val:.8f}f, ")
            if (i+1) % 10 == 0: f.write("\n")
        f.write("};\n\n")

        # Layer 3: Output (10 neurons)
        w3, b3 = model.layers[2].get_weights()
        f.write(f"// Layer 3: Dense (100) -> Output (10)\n")
        f.write(f"const float output_weights[{w3.size}] = {{\n")
        w3_flat = w3.flatten()
        for i, val in enumerate(w3_flat):
            f.write(f"{val:.8f}f, ")
            if (i+1) % 10 == 0: f.write("\n")
        f.write("};\n\n")

        f.write(f"const float output_biases[{b3.size}] = {{\n")
        for i, val in enumerate(b3):
            f.write(f"{val:.8f}f, ")
            if (i+1) % 10 == 0: f.write("\n")
        f.write("};\n\n")

        f.write("#endif\n")
        print(f"Successfully saved weights to {filename}")

# Load the saved model
try:
    model = tf.keras.models.load_model("mlp_fsdd_model.h5")
    model.summary()
    save_header("weights_data.h", model)
except Exception as e:
    print(f"Error loading model: {e}")