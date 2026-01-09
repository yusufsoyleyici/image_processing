import serial
import time
import numpy as np
import tensorflow as tf
import struct
import os

# --- CONFIG ---
# CHECK YOUR COM PORT! (e.g., COM3 on Windows, /dev/ttyACM0 on Linux)
SERIAL_PORT = 'COM6' 
BAUD_RATE = 115200

# Path to your data
DATASET_FOLDER = r"C:\Users\yyigi\Desktop\yusuf\ders\Introduction to Embedded Image Processing\EE4065_STM32_nucleo_F446RE-Image-Transfer-STM-to-PC-and-PC-to-STM\image_transfer_nucleof446_python_PC\MNIST-dataset"

def load_mnist_test(path):
    images_path = os.path.join(path, 't10k-images.idx3-ubyte')
    labels_path = os.path.join(path, 't10k-labels.idx1-ubyte')
    with open(labels_path, 'rb') as lbpath:
        struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 28, 28)
    return images, labels

def preprocess_single_image(image):
    # Resize and Colorize to match the 32x32x3 input
    img = tf.expand_dims(image, axis=-1)
    img = tf.image.grayscale_to_rgb(img)
    img = tf.image.resize(img, [32, 32])
    # Cast to uint8 (0-255) for UART transmission
    img = tf.cast(img, tf.uint8)
    return img.numpy()

# 1. Setup Serial
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
print(f"Connected to {SERIAL_PORT}")

# 2. Load Data
images, labels = load_mnist_test(DATASET_FOLDER)

# 3. Loop through random images
while True:
    idx = np.random.randint(0, len(images))
    target_img = images[idx]
    actual_label = labels[idx]
    
    # Process
    processed_img = preprocess_single_image(target_img) # Shape (32, 32, 3)
    flat_data = processed_img.flatten().tobytes()
    
    print(f"\nSending Digit: {actual_label} ...")
    ser.write(flat_data)
    
    # Wait for response
    time.sleep(0.5)
    while ser.in_waiting:
        print(f"STM32: {ser.readline().decode().strip()}")
        
    input("Press Enter to send next image...")