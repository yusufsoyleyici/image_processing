import cv2
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
MODEL_PATH = "recovered_model.tflite"
INPUT_SIZE = 96  # From your Netron screenshot
THRESHOLD = 0.6  # Confidence threshold

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    # 1. LOAD MODEL
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error: {e}")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check data types
    is_quantized_input = (input_details[0]['dtype'] == np.uint8)
    
    # 2. START WEBCAM
    cap = cv2.VideoCapture(0)
    
    print("---------------------------------------")
    print("Running Part B: Custom Handwriting Model")
    print("Model trained on: REAL PHOTOS (Dark pen, White paper)")
    print("Press 'q' to quit")
    print("---------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 3. PRE-PROCESS (CROP & RESIZE)
        # We crop the center to make it square
        h, w, _ = frame.shape
        min_dim = min(h, w)
        crop_img = frame[(h-min_dim)//2:(h+min_dim)//2, (w-min_dim)//2:(w+min_dim)//2]
        
        # Resize to 96x96
        input_img = cv2.resize(crop_img, (INPUT_SIZE, INPUT_SIZE))
        
        # --- CRITICAL CHANGE: NO INVERSION ---
        # We send the image AS IS because your model knows what "paper" looks like.
        
        # Convert BGR to RGB (AI standard)
        input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension [1, 96, 96, 3]
        input_tensor = np.expand_dims(input_rgb, axis=0)

        # Normalize Input
        if is_quantized_input:
            input_data = input_tensor.astype(np.uint8)
        else:
            # Standard normalization 0.0 - 1.0
            input_data = (input_tensor.astype(np.float32) / 255.0)

        # 4. INFERENCE
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # 5. DECODE OUTPUT
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Handle Output Quantization
        scale = output_details[0]['quantization'][0]
        zero_point = output_details[0]['quantization'][1]
        
        if scale > 0: 
            output_data = (output_data.astype(np.float32) - zero_point) * scale

        # Find best prediction
        max_conf = 0
        best_class = -1

        for i in range(output_data.shape[0]):
            # Apply Sigmoid to get probability (0.0 - 1.0)
            raw_conf = output_data[i][4] # Objectness
            confidence = sigmoid(raw_conf)

            if confidence > max_conf:
                max_conf = confidence
                class_scores = output_data[i][5:]
                best_class = np.argmax(class_scores)

        # 6. VISUALIZATION
        display_frame = crop_img.copy()
        
        # Show what the AI sees (Small box top-left)
        preview_size = 150
        ai_view = cv2.resize(input_rgb, (preview_size, preview_size))
        # Convert back to BGR for OpenCV display
        ai_view_bgr = cv2.cvtColor(ai_view, cv2.COLOR_RGB2BGR)
        display_frame[0:preview_size, 0:preview_size] = ai_view_bgr

        # Display Text
        status_color = (0, 0, 255) # Red
        status_text = f"Searching..."
        
        if max_conf > THRESHOLD:
            status_color = (0, 255, 0) # Green
            status_text = f"DIGIT: {best_class} ({max_conf:.2f})"
        
        cv2.putText(display_frame, status_text, (10, preview_size + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # Draw a rectangle around the "AI View" to show where to look
        cv2.rectangle(display_frame, (0,0), (preview_size, preview_size), (255, 255, 0), 2)

        cv2.imshow('Part B - Final', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()