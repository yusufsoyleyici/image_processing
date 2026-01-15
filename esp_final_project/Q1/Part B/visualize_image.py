import serial
import numpy as np
import cv2
import time
import sys

class ESP32CameraSystem:
    def __init__(self, port='COM8', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.connected = False
        
    def connect(self):
        """Connect to ESP32-CAM"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=5.0,
                write_timeout=5.0
            )
            time.sleep(2)  # Wait for connection
            
            # Clear buffer
            self.ser.reset_input_buffer()
            
            print(f"Connected to {self.port}")
            
            # Wait for READY message
            start_time = time.time()
            while time.time() - start_time < 10:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    print(f"ESP32: {line}")
                    if "READY" in line:
                        self.connected = True
                        return True
                time.sleep(0.1)
            
            # Try sending TEST command
            self.ser.write(b"TEST\n")
            time.sleep(0.1)
            response = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if response == "TEST_OK":
                self.connected = True
                return True
            
            return False
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def read_line_with_timeout(self, timeout=2.0):
        """Read a line with a specific timeout"""
        start = time.time()
        while time.time() - start < timeout:
            if self.ser.in_waiting:
                return self.ser.readline().decode('utf-8', errors='ignore').strip()
            time.sleep(0.01)
        return None
    
    def capture_image(self):
        """Capture and process image from ESP32"""
        if not self.connected:
            print("Not connected to ESP32")
            return None, None, None
        
        print("\n[DEBUG] Sending CAPTURE command to ESP32...")
        self.ser.reset_input_buffer()  # Clear any old data
        self.ser.write(b"CAPTURE\n")
        self.ser.flush()
        print("[DEBUG] Command sent, waiting for response...")
        time.sleep(0.2)
        
        original_image = None
        thresholded_image = None
        threshold_info = {}
        width = 320
        height = 240
        
        start_time = time.time()
        
        while time.time() - start_time < 15:  # Increased timeout
            if self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                
                print(f"ESP32: {line}")
                
                if line == "GET_READY":
                    print("⚠️  GET READY! Point camera at target NOW! Photo in 2 seconds...")
                
                elif line == "CAPTURING_NOW":
                    print("📸 FLASH & CAPTURE HAPPENING NOW!")
                
                elif line == "CAPTURE_DONE":
                    print("✓ Photo captured! Processing...")
                
                elif line.startswith("PROCESSED:"):
                    threshold_info['processed'] = int(line.split(':')[1])
                
                elif line.startswith("THRESHOLD:"):
                    threshold_info['threshold'] = int(line.split(":")[1])
                
                elif line.startswith("ABOVE:"):
                    threshold_info['above'] = int(line.split(":")[1])
                
                elif line.startswith("NEEDED:"):
                    threshold_info['needed'] = int(line.split(":")[1])
                
                elif line.startswith("WHITE_PIXELS:"):
                    threshold_info['white_pixels'] = int(line.split(":")[1])
                
                elif line == "ORIGINAL_START":
                    # Read dimensions
                    width_line = self.read_line_with_timeout()
                    height_line = self.read_line_with_timeout()
                    
                    if width_line and width_line.startswith("WIDTH:"):
                        width = int(width_line.split(":")[1])
                    if height_line and height_line.startswith("HEIGHT:"):
                        height = int(height_line.split(":")[1])
                    
                    print(f"Receiving original image ({width}x{height})...")
                    
                    # Read image data
                    total_pixels = width * height
                    data = bytearray()
                    
                    bytes_read = 0
                    last_progress = 0
                    while len(data) < total_pixels:
                        remaining = total_pixels - len(data)
                        chunk = self.ser.read(min(4096, remaining))
                        if not chunk:
                            time.sleep(0.01)
                            continue
                        data.extend(chunk)
                        bytes_read += len(chunk)
                        
                        # Show progress every 10%
                        progress = int((bytes_read / total_pixels) * 100)
                        if progress >= last_progress + 10:
                            print(f"  Progress: {progress}%")
                            last_progress = progress
                    
                    if len(data) == total_pixels:
                        original_image = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
                        print("✓ Original image received")
                    else:
                        print(f"Error: Expected {total_pixels} bytes, got {len(data)}")
                    
                    # Read end marker
                    end_marker = self.read_line_with_timeout()
                    if end_marker:
                        print(f"ESP32: {end_marker}")
                
                elif line == "THRESHOLDED_START":
                    # Read dimensions
                    width_line = self.read_line_with_timeout()
                    height_line = self.read_line_with_timeout()
                    
                    if width_line and width_line.startswith("WIDTH:"):
                        width = int(width_line.split(":")[1])
                    if height_line and height_line.startswith("HEIGHT:"):
                        height = int(height_line.split(":")[1])
                    
                    print(f"Receiving thresholded image ({width}x{height})...")
                    
                    # Read image data
                    total_pixels = width * height
                    data = bytearray()
                    
                    bytes_read = 0
                    last_progress = 0
                    while len(data) < total_pixels:
                        remaining = total_pixels - len(data)
                        chunk = self.ser.read(min(4096, remaining))
                        if not chunk:
                            time.sleep(0.01)
                            continue
                        data.extend(chunk)
                        bytes_read += len(chunk)
                        
                        # Show progress every 10%
                        progress = int((bytes_read / total_pixels) * 100)
                        if progress >= last_progress + 10:
                            print(f"  Progress: {progress}%")
                            last_progress = progress
                    
                    if len(data) == total_pixels:
                        thresholded_image = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
                        print("✓ Thresholded image received")
                    else:
                        print(f"Error: Expected {total_pixels} bytes, got {len(data)}")
                    
                    # Read end marker
                    end_marker = self.read_line_with_timeout()
                    if end_marker:
                        print(f"ESP32: {end_marker}")
                    
                    # Continue reading to get WHITE_PIXELS and DONE messages
                    start_time = time.time()  # Reset timeout for final messages
                
                elif line == "DONE":
                    print("✓ Processing complete")
                    return original_image, thresholded_image, threshold_info
            else:
                time.sleep(0.01)
        
        print("Timeout - but may have received images")
        # Return what we have even if we didn't get DONE
        if original_image is not None or thresholded_image is not None:
            return original_image, thresholded_image, threshold_info
        return None, None, None
    
    def display_images(self, original, thresholded, info):
        """Display the captured images"""
        if original is not None:
            # Apply colormap to grayscale for better visibility
            original_colored = cv2.applyColorMap(original, cv2.COLORMAP_JET)
            cv2.imshow("Original Image (Heatmap)", original_colored)
            
            # Also show actual grayscale
            cv2.imshow("Original Image (Grayscale)", original)
        
        if thresholded is not None:
            # Display clean thresholded image without text
            cv2.imshow("Thresholded Image (Binary)", thresholded)
        
        print("\nImages displayed. Press any key in image windows to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save_images(self, original, thresholded, prefix=""):
        """Save images to files"""
        if original is not None:
            filename = f"{prefix}original_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, original)
            print(f"Saved {filename}")
        
        if thresholded is not None:
            filename = f"{prefix}thresholded_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, thresholded)
            print(f"Saved {filename}")
    
    def close(self):
        """Close serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed")

def main():
    print("=" * 60)
    print("ESP32-CAM GRAYSCALE THRESHOLDING SYSTEM")
    print("=" * 60)
    print("\nThis system:")
    print("1. Captures grayscale image from ESP32-CAM")
    print("2. Calculates threshold for exactly 1000 white pixels")
    print("3. Applies threshold with tie-breaking")
    print("4. Displays both original and thresholded images")
    print("=" * 60)
    
    # Create camera system
    camera = ESP32CameraSystem(port='COM8')
    
    # Connect to ESP32
    if not camera.connect():
        print("\nFailed to connect to ESP32-CAM")
        print("Please check:")
        print("1. COM port is correct (currently using COM8)")
        print("2. ESP32-CAM is properly connected")
        print("3. No other program is using the serial port")
        print("4. ESP32 has the correct firmware uploaded")
        return
    
    print("\nConnection successful!")
    print("\nCommands:")
    print("  c - Capture and process image")
    print("  s - Save current images")
    print("  q - Quit")
    print("-" * 30)
    
    current_original = None
    current_thresholded = None
    current_info = None
    
    try:
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'q':
                print("Quitting...")
                break
                
            elif command == 'c':
                print("\n" + "=" * 40)
                print("CAPTURING IMAGE...")
                print("=" * 40)
                
                original, thresholded, info = camera.capture_image()
                
                if original is not None or thresholded is not None:
                    current_original = original
                    current_thresholded = thresholded
                    current_info = info
                    
                    print("\n" + "=" * 40)
                    print("CAPTURE COMPLETE")
                    print("=" * 40)
                    
                    # Display info
                    if info:
                        print(f"\nThreshold Information:")
                        print(f"  Threshold Value: {info.get('threshold', 'N/A')}")
                        print(f"  Pixels above threshold: {info.get('above', 'N/A')}")
                        print(f"  Additional needed: {info.get('needed', 'N/A')}")
                        print(f"  Total white pixels: {info.get('white_pixels', 'N/A')}")
                    
                    camera.display_images(original, thresholded, info)
                else:
                    print("Failed to capture images")
                
            elif command == 's':
                if current_original is not None or current_thresholded is not None:
                    camera.save_images(current_original, current_thresholded, "capture_")
                    print("Images saved")
                else:
                    print("No images to save. Capture an image first.")
            
            else:
                print("Unknown command. Use 'c', 's', or 'q'")
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("\nProgram ended")

if __name__ == "__main__":
    main()