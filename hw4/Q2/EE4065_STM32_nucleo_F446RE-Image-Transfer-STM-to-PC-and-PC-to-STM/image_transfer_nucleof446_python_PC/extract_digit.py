import os
import struct
import numpy as np
import cv2

# Change this to whatever number you want to test!
TARGET_DIGIT = 0 

img_path = os.path.join("MNIST-dataset", "t10k-images.idx3-ubyte")
lbl_path = os.path.join("MNIST-dataset", "t10k-labels.idx1-ubyte")

print(f"Searching for a number {TARGET_DIGIT}...")

with open(lbl_path, 'rb') as fl, open(img_path, 'rb') as fi:
    fl.read(8)   # Skip header
    fi.read(16)  # Skip header

    for i in range(1000):
        label = int.from_bytes(fl.read(1), 'big')
        image_data = fi.read(784)
        
        if label == TARGET_DIGIT:
            img = np.frombuffer(image_data, dtype=np.uint8).reshape(28, 28)
            filename = f"test_digit_{TARGET_DIGIT}.png"
            cv2.imwrite(filename, img)
            print(f"Found it! Saved '{filename}'.")
            break