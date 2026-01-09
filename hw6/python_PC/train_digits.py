import tensorflow as tf
import numpy as np
import os
import struct
from tensorflow import keras
from keras import layers
from st_efficientnet_lc_v1 import get_st_efficientnet_lc_v1

# --- CONFIGURATION ---
# Check your path one last time!
DATASET_FOLDER = r"C:\Users\yyigi\Desktop\yusuf\ders\Introduction to Embedded Image Processing\EE4065_STM32_nucleo_F446RE-Image-Transfer-STM-to-PC-and-PC-to-STM\image_transfer_nucleof446_python_PC\MNIST-dataset"

BATCH_SIZE = 32
EPOCHS = 5 
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3) 

# --- 1. DEFINING SQUEEZENET (FIXED: No '/' in names) ---
def fire_module(x, fire_id, squeeze=16, expand=64):
    # FIXED: Changed '/' to '_' to prevent ValueError
    s_id = 'fire' + str(fire_id) + '_' 
    
    # Squeeze Layer (1x1 Conv)
    sq = layers.Conv2D(squeeze, (1, 1), padding='valid', name=s_id + 'squeeze1x1')(x)
    sq = layers.BatchNormalization()(sq) 
    sq = layers.Activation('relu')(sq)

    # Expand Layer (1x1 Conv)
    ex1 = layers.Conv2D(expand, (1, 1), padding='valid', name=s_id + 'expand1x1')(sq)
    ex1 = layers.BatchNormalization()(ex1)
    ex1 = layers.Activation('relu')(ex1)

    # Expand Layer (3x3 Conv)
    ex3 = layers.Conv2D(expand, (3, 3), padding='same', name=s_id + 'expand3x3')(sq)
    ex3 = layers.BatchNormalization()(ex3)
    ex3 = layers.Activation('relu')(ex3)

    # Concatenate
    x = layers.concatenate([ex1, ex3], axis=3)
    return x

def create_squeezenet(num_classes=10, input_shape=(32,32,3)):
    inputs = layers.Input(shape=input_shape)
    
    # Initial Conv
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    # Fire Modules 2 & 3
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    # Fire Modules 4 & 5
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    
    # End Classifier
    x = layers.Dropout(0.5, name='dropout')(x)
    
    x = layers.Conv2D(num_classes, (1, 1), padding='valid', name='conv10')(x)
    x = layers.Activation('relu')(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation('softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name='SqueezeNet_Fixed')
    return model

# --- 2. DATA LOADING ---
def load_mnist_from_binary(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images.idx3-ubyte')
    with open(labels_path, 'rb') as lbpath:
        struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 28, 28)
    return images, labels

print("Loading Data...")
train_images, train_labels = load_mnist_from_binary(DATASET_FOLDER, kind='train')
val_images, val_labels = load_mnist_from_binary(DATASET_FOLDER, kind='t10k')

# --- 3. PREPROCESSING ---
def preprocess_data(images, labels):
    images = tf.expand_dims(images, axis=-1)
    images = tf.image.grayscale_to_rgb(images)
    images = tf.image.resize(images, [INPUT_SHAPE[0], INPUT_SHAPE[1]])
    images = images / 255.0
    labels = tf.one_hot(labels, NUM_CLASSES)
    return images, labels

print("Preprocessing...")
train_X, train_y = preprocess_data(train_images, train_labels)
val_X, val_y = preprocess_data(val_images, val_labels)

# --- 4. TRAIN SQUEEZENET ---
print("\n=== TRAINING FIXED SQUEEZENET ===")
squeeze_model = create_squeezenet(NUM_CLASSES, INPUT_SHAPE)
squeeze_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
squeeze_model.fit(train_X, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_X, val_y))

# Save
squeeze_model.save("squeezenet_mnist.h5")
print("SUCCESS: Saved squeezenet_mnist.h5")