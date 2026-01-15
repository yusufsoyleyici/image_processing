import cv2
import numpy as np
import tensorflow as tf
import os

# --- CONFIGURATION ---
MODEL_PATH = "recovered_model.tflite"
IMAGE_FILENAME = "resim.png"  # <--- REPLACE THIS with your image name
INPUT_SIZE = 96  # From your Netron screenshot

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    # 1. CHECK FILE EXISTENCE
    if not os.path.exists(IMAGE_FILENAME):
        print(f"Error: Image '{IMAGE_FILENAME}' not found in this folder.")
        return

    # 2. LOAD MODEL
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 3. LOAD AND PREPROCESS IMAGE
    original_img = cv2.imread(IMAGE_FILENAME)
    if original_img is None:
        print("Error: Could not read image. Check format.")
        return

    print(f"Loaded image: {original_img.shape}")

    # Resize to 96x96 (Model Requirement)
    # Note: If your image is not square, this might squish the digit.
    # ideally, crop it to a square first manually if results are bad.
    input_img = cv2.resize(original_img, (INPUT_SIZE, INPUT_SIZE))
    
    # Convert BGR (OpenCV) to RGB (AI)
    input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    
    # Add Batch Dimension: [1, 96, 96, 3]
    input_tensor = np.expand_dims(input_rgb, axis=0)

    # Normalize based on model type
    if input_details[0]['dtype'] == np.uint8:
        print("Model Input: Quantized (uint8)")
        input_data = input_tensor.astype(np.uint8)
    else:
        print("Model Input: Float32")
        input_data = (input_tensor.astype(np.float32) / 255.0)

    # 4. RUN INFERENCE
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # 5. DECODE OUTPUT (YOLO Style)
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Handle Output Quantization
    scale = output_details[0]['quantization'][0]
    zero_point = output_details[0]['quantization'][1]
    if scale > 0: 
        output_data = (output_data.astype(np.float32) - zero_point) * scale

    # Find Best Prediction
    max_conf = 0
    best_class = -1
    
    # Loop through all 567 potential boxes
    for i in range(output_data.shape[0]):
        # Confidence is usually at index 4
        raw_conf = output_data[i][4] 
        confidence = sigmoid(raw_conf)

        if confidence > max_conf:
            max_conf = confidence
            # Classes are indices 5 to 14 (for 10 digits)
            class_scores = output_data[i][5:]
            best_class = np.argmax(class_scores)

    # 6. REPORT RESULTS
    print("\n" + "="*30)
    print(f"   PREDICTION RESULT")
    print("="*30)
    
    if max_conf > 0.5:
        print(f"DETECTED DIGIT: {best_class}")
        print(f"Confidence:     {max_conf:.2f} ({(max_conf*100):.1f}%)")
    else:
        print("No digit detected (Confidence too low)")
        print(f"Best Guess: {best_class} ({max_conf:.2f})")
    
    print("="*30 + "\n")

    # 7. SHOW WHAT THE MODEL SAW
    # It helps to see if the resize squished the digit
    cv2.imshow("Input to AI (96x96)", input_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()