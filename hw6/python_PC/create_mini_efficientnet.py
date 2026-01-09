import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

# --- CONFIGURATION ---
INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 32
EPOCHS = 5  # Needs a few more epochs since we are building from scratch

# --- 1. DATA PREPARATION ---
print("Loading MNIST Data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def preprocess(images, labels):
    images = tf.expand_dims(images, axis=-1)
    images = tf.image.grayscale_to_rgb(images)
    images = tf.image.resize(images, [32, 32])
    images = images / 255.0
    return images, labels

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess).shuffle(1000).batch(BATCH_SIZE)

# --- 2. DEFINE MINI-EFFICIENTNET BLOCKS ---
# EfficientNet uses "MBConv" (Mobile Inverted Residual Bottleneck)
def mb_conv_block(inputs, filters, expansion_factor, stride):
    in_channels = inputs.shape[-1]
    expanded_channels = in_channels * expansion_factor

    x = inputs
    
    # 1. Expansion Phase (1x1 Conv)
    if expansion_factor != 1:
        x = layers.Conv2D(expanded_channels, kernel_size=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x) # EfficientNet uses Swish activation

    # 2. Depthwise Convolution (3x3)
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', strides=stride, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    # 3. Projection Phase (1x1 Conv, Linear activation)
    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # 4. Skip Connection (Residual)
    if stride == 1 and in_channels == filters:
        x = layers.Add()([inputs, x])
    
    return x

# --- 3. BUILD THE MODEL ---
print("Building Custom Mini-EfficientNet...")
inputs = layers.Input(shape=INPUT_SHAPE)

# Initial Stem
x = layers.Conv2D(16, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('swish')(x)

# MBConv Blocks (Scaled down version of EfficientNet)
# We use fewer blocks and filters to keep size < 200KB
x = mb_conv_block(x, filters=16, expansion_factor=1, stride=1)
x = mb_conv_block(x, filters=24, expansion_factor=4, stride=2)
x = mb_conv_block(x, filters=24, expansion_factor=4, stride=1)
x = mb_conv_block(x, filters=32, expansion_factor=4, stride=2)

# Head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.summary() # Check parameter count!

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 4. TRAIN ---
print("Training...")
model.fit(train_ds, epochs=EPOCHS)

# --- 5. QUANTIZE ---
print("Quantizing to Int8...")
def representative_data_gen():
    for images, _ in train_ds.take(20):
        for i in range(BATCH_SIZE):
            yield [tf.expand_dims(images[i], axis=0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

output_path = "mini_efficientnet.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"\nSUCCESS! Created: {output_path}")
print(f"Size: {os.path.getsize(output_path)/1024:.2f} KB")