import os
import struct
import numpy as np
import cv2

# Define path to the downloaded offline file
filename = os.path.join("MNIST-dataset", "t10k-images.idx3-ubyte")

with open(filename, 'rb') as f:
    # Skip the header (magic number, count, rows, cols)
    f.read(16)
    # Read the first 28x28 pixels (first image is a '7')
    buffer = f.read(28 * 28)
    img = np.frombuffer(buffer, dtype=np.uint8).reshape(28, 28)

# Save it as a standard picture file
cv2.imwrite("test_digit_7.png", img)
print("Saved 'test_digit_7.png'. Use this file for testing!")