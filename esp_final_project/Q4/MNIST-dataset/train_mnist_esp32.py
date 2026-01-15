import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import struct
import os

print("TensorFlow version:", tf.__version__)

# ========================================
# 1. LOAD MNIST DATASET FROM IDX FILES
# ========================================

def read_idx_images(filename):
    """Read images from IDX3-UBYTE format"""
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        
        print(f"Loading {num_images} images of {rows}x{cols}")
        
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        
    return images

def read_idx_labels(filename):
    """Read labels from IDX1-UBYTE format"""
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        
        print(f"Loading {num_labels} labels")
        
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels

# Load your dataset
print("\n=== LOADING MNIST DATASET ===")
train_images = read_idx_images('train-images.idx3-ubyte')
train_labels = read_idx_labels('train-labels.idx1-ubyte')
test_images = read_idx_images('t10k-images.idx3-ubyte')
test_labels = read_idx_labels('t10k-labels.idx1-ubyte')

# Preprocess: resize to 32x32 and convert to RGB (3 channels) to match your ESP32 code
print("\n=== PREPROCESSING ===")
print("Resizing from 28x28 to 32x32 and converting to RGB format...")

def preprocess_images(images):
    # Resize to 32x32
    resized = tf.image.resize(images[..., np.newaxis], [32, 32])
    # Convert to RGB (repeat grayscale across 3 channels)
    rgb = tf.repeat(resized, 3, axis=-1)
    # INVERT: Convert from white-on-black to black-on-white to match your camera
    inverted = 255.0 - rgb
    # Normalize to [0, 1]
    normalized = inverted / 255.0
    return normalized.numpy()

X_train = preprocess_images(train_images)
X_test = preprocess_images(test_images)
y_train = train_labels
y_test = test_labels

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Label distribution: {np.bincount(y_train)}")

# ========================================
# 2. BUILD SQUEEZENET MODEL
# ========================================

def fire_module(x, squeeze_filters, expand_filters):
    """Fire module for SqueezeNet"""
    squeeze = layers.Conv2D(squeeze_filters, (1, 1), activation='relu', padding='same')(x)
    
    expand_1x1 = layers.Conv2D(expand_filters, (1, 1), activation='relu', padding='same')(squeeze)
    expand_3x3 = layers.Conv2D(expand_filters, (3, 3), activation='relu', padding='same')(squeeze)
    
    return layers.Concatenate()([expand_1x1, expand_3x3])

def build_squeezenet():
    """Lightweight SqueezeNet for ESP32"""
    inputs = keras.Input(shape=(32, 32, 3))
    
    x = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = fire_module(x, 8, 16)
    x = fire_module(x, 8, 16)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = fire_module(x, 16, 32)
    x = fire_module(x, 16, 32)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='squeezenet')
    return model

# ========================================
# 3. BUILD EFFICIENTNET MODEL
# ========================================

def build_efficientnet():
    """Lightweight EfficientNet-style model for ESP32"""
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Initial conv
    x = layers.Conv2D(16, (3, 3), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # MBConv blocks (simplified)
    x = layers.DepthwiseConv2D((3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(24, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.DepthwiseConv2D((3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='efficientnet')
    return model

# ========================================
# 4. TRAIN MODELS
# ========================================

def train_model(model, name):
    print(f"\n=== TRAINING {name} ===")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{name} Test Accuracy: {test_acc*100:.2f}%")
    
    return model

print("\n" + "="*60)
print("TRAINING SQUEEZENET")
print("="*60)
squeezenet_model = build_squeezenet()
squeezenet_model = train_model(squeezenet_model, "SqueezeNet")

print("\n" + "="*60)
print("TRAINING EFFICIENTNET")
print("="*60)
efficientnet_model = build_efficientnet()
efficientnet_model = train_model(efficientnet_model, "EfficientNet")

# ========================================
# 5. CONVERT TO TFLITE
# ========================================

def convert_to_tflite(model, name):
    """Convert Keras model to TFLite format optimized for ESP32"""
    print(f"\n=== CONVERTING {name} TO TFLITE ===")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimizations for microcontrollers
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    
    # Representative dataset for quantization
    def representative_dataset():
        for i in range(100):
            yield [X_train[i:i+1].astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_filename = f"{name.lower()}_model.tflite"
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved {tflite_filename} ({len(tflite_model)} bytes)")
    
    return tflite_model, tflite_filename

squeezenet_tflite, squeezenet_file = convert_to_tflite(squeezenet_model, "SqueezeNet")
efficientnet_tflite, efficientnet_file = convert_to_tflite(efficientnet_model, "EfficientNet")

# ========================================
# 6. GENERATE C HEADER FILES
# ========================================

def generate_header_file(tflite_model, model_name):
    """Generate C header file for ESP32"""
    header_filename = f"{model_name.lower()}_model.h"
    
    with open(header_filename, 'w') as f:
        f.write(f"// Auto-generated model file for {model_name}\n")
        f.write(f"// Model size: {len(tflite_model)} bytes\n\n")
        f.write("#ifndef " + model_name.upper() + "_MODEL_H\n")
        f.write("#define " + model_name.upper() + "_MODEL_H\n\n")
        
        f.write(f"const unsigned int {model_name.lower()}_model_len = {len(tflite_model)};\n")
        f.write(f"const unsigned char {model_name.lower()}_model[] = {{\n")
        
        # Write bytes in rows of 12
        for i in range(0, len(tflite_model), 12):
            chunk = tflite_model[i:i+12]
            hex_values = ', '.join([f'0x{b:02x}' for b in chunk])
            f.write(f"  {hex_values}")
            if i + 12 < len(tflite_model):
                f.write(",")
            f.write("\n")
        
        f.write("};\n\n")
        f.write("#endif\n")
    
    print(f"Generated {header_filename}")

print("\n=== GENERATING HEADER FILES ===")
generate_header_file(squeezenet_tflite, "SqueezeNet")
generate_header_file(efficientnet_tflite, "EfficientNet")

# ========================================
# 7. TEST TFLITE MODELS
# ========================================

def test_tflite_model(tflite_model, name):
    """Test TFLite model accuracy"""
    print(f"\n=== TESTING {name} TFLITE MODEL ===")
    
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    correct = 0
    total = 0
    
    for i in range(min(1000, len(X_test))):
        interpreter.set_tensor(input_details[0]['index'], X_test[i:i+1].astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        predicted = np.argmax(output[0])
        if predicted == y_test[i]:
            correct += 1
        total += 1
    
    accuracy = (correct / total) * 100
    print(f"{name} TFLite Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Test with a few examples
    print(f"\nSample predictions from {name}:")
    for i in range(5):
        interpreter.set_tensor(input_details[0]['index'], X_test[i:i+1].astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted = np.argmax(output)
        confidence = output[predicted] * 100
        print(f"  True: {y_test[i]}, Predicted: {predicted}, Confidence: {confidence:.1f}%")

test_tflite_model(squeezenet_tflite, "SqueezeNet")
test_tflite_model(efficientnet_tflite, "EfficientNet")

print("\n" + "="*60)
print("DONE! Generated files:")
print("  - squeezenet_model.h")
print("  - efficientnet_model.h")
print("  - squeezenet_model.tflite")
print("  - efficientnet_model.tflite")
print("\nCopy the .h files to your ESP32 project folder.")
print("="*60)