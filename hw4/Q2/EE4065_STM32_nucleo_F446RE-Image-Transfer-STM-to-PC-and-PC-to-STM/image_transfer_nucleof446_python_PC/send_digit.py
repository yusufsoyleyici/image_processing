import py_serialimg
import numpy as np
import cv2 
import time

# --- SETUP ---
COM_PORT = "COM6" 
TEST_IMAGE_FILENAME = "test_digit_0.png" # Provide an image of a digit here
# -------------

print(f"Initializing Serial on {COM_PORT}...")
py_serialimg.SERIAL_Init(COM_PORT)

print("\n--- WAITING FOR REQUEST ---")
print("Reset your STM32 board now if needed.")
print("The STM32 should request an image shortly...")

try:
    while True:
        # Check if STM32 is asking for data
        rqType, height, width, format = py_serialimg.SERIAL_IMG_PollForRequest()

        # 1. STM32 Wants to Read (PC -> STM32)
        if rqType == py_serialimg.MCU_READS:
            print(f"\n[STM32 Request] Send Image sized {width}x{height}")
            
            # Load and Resize Image
            try:
                img = cv2.imread(TEST_IMAGE_FILENAME)
                if img is None:
                    raise Exception(f"Could not load {TEST_IMAGE_FILENAME}")
                
                # Resize to what STM32 asked for (should be 28x28)
                img_resized = cv2.resize(img, (width, height))
                
                # Send it
                py_serialimg.SERIAL_IMG_Write(TEST_IMAGE_FILENAME) # This func handles internal write
                print(f"Sent {TEST_IMAGE_FILENAME} (Resized to {width}x{height})")
                print(">>> CHECK YOUR STM32 TERMINAL/DEBUGGER FOR PREDICTION RESULT! <<<")
                
            except Exception as e:
                print(f"Error preparing image: {e}")

        # 2. STM32 Wants to Write (STM32 -> PC)
        # Our code sends the image back as an 'Echo' to confirm reception
        elif rqType == py_serialimg.MCU_WRITES:
            print(f"\n[STM32 Sending] Receiving echo back...")
            img = py_serialimg.SERIAL_IMG_Read()
            print("Echo received. Cycle complete.")

except KeyboardInterrupt:
    print("\nStopped.")
except Exception as e:
    print(f"Error: {e}")