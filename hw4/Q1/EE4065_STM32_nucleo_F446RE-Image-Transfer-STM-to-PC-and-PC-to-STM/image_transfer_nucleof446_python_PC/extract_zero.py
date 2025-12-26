import os
import struct
import numpy as np
import cv2

# Paths to the offline files
img_path = os.path.join("MNIST-dataset", "t10k-images.idx3-ubyte")
lbl_path = os.path.join("MNIST-dataset", "t10k-labels.idx1-ubyte")

print("Searching for a Zero...")

with open(lbl_path, 'rb') as fl, open(img_path, 'rb') as fi:
    # 1. Skip Headers
    fl.read(8)   # Label file header
    fi.read(16)  # Image file header

    # 2. Loop through the dataset until we find a '0'
    for i in range(1000): # Check first 1000 images
        # Read one label
        label_byte = fl.read(1)
        if not label_byte: break # End of file
        
        label = int.from_bytes(label_byte, 'big')
        
        # Read the corresponding image (28x28 = 784 bytes)
        image_data = fi.read(784)
        
        if label == 0:
            print(f"Found a Zero at index {i}!")
            
            # Convert bytes to numpy array
            img = np.frombuffer(image_data, dtype=np.uint8).reshape(28, 28)
            
            # Save it
            filename = "test_digit_0.png"
            cv2.imwrite(filename, img)
            print(f"Saved '{filename}'.")
            break