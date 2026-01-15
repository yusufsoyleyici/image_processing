import serial
import numpy as np
import cv2
import time

SERIAL_PORT = 'COM8'
BAUD_RATE = 115200

def main():
    print("Connecting to ESP32...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
    time.sleep(2)
    
    # Clear buffer
    ser.reset_input_buffer()
    
    print("Waiting for ESP32...")
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(f"ESP32: {line}")
            if "READY" in line:
                break
    
    print("\n=== ESP32 Commands ===")
    print("Available commands:")
    print("  a          - Default (1.5x up, 0.666x down)")
    print("  u<value>   - Upsample only (e.g., u2.0, u1.5)")
    print("  d<value>   - Downsample only (e.g., d0.5, d0.75)")
    print("  b<up>,<down> - Both custom (e.g., b2.0,0.5)")
    print("  q          - Quit")
    
    while True:
        cmd = input("\nEnter command: ").strip()
        
        if cmd == 'q':
            break
        
        if not cmd:
            continue
        
        # Validate command format
        if cmd[0] not in ['a', 'u', 'd', 'b']:
            print("Invalid command! Use a, u, d, or b")
            continue
        
        # Send command
        print(f"\nSending '{cmd}' command...")
        ser.write(cmd.encode() + b'\n')
        ser.flush()
        
        # Process images
        images = process_images(ser)
        
        # Display and save
        if images:
            display_and_save_all(images)
    
    ser.close()

def process_images(ser):
    """Process and return all images"""
    images = {}
    
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        
        print(f"> {line}")
        
        if line == "START":
            print("Processing started...")
        
        elif line.startswith("ORIGINAL:"):
            parts = line.split(':')
            w = int(parts[1])
            h = int(parts[2])
            expected = w * h
            print(f"Receiving original: {w}x{h} ({expected} bytes)")
            
            data = read_image_data(ser, expected, "original")
            if data is not None:
                try:
                    img = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
                    images['original'] = img
                    print(f"✓ Original received: {w}x{h}")
                except Exception as e:
                    print(f"✗ Failed to reshape original: {e}")
        
        elif line.startswith("UPSAMPLED:"):
            parts = line.split(':')
            w = int(parts[1])
            h = int(parts[2])
            scale = parts[3]
            expected = w * h
            print(f"Receiving upsampled ({scale}x): {w}x{h} ({expected} bytes)")
            
            data = read_image_data(ser, expected, "upsampled")
            if data is not None:
                try:
                    img = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
                    images[f'upsampled_{scale}'] = {'img': img, 'scale': scale}
                    print(f"✓ Upsampled received: {w}x{h}")
                except Exception as e:
                    print(f"✗ Failed to reshape upsampled: {e}")
        
        elif line.startswith("DOWNSAMPLED:"):
            parts = line.split(':')
            w = int(parts[1])
            h = int(parts[2])
            scale = parts[3]
            expected = w * h
            print(f"Receiving downsampled ({scale}x): {w}x{h} ({expected} bytes)")
            
            data = read_image_data(ser, expected, "downsampled")
            if data is not None:
                try:
                    img = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
                    images[f'downsampled_{scale}'] = {'img': img, 'scale': scale}
                    print(f"✓ Downsampled received: {w}x{h}")
                except Exception as e:
                    print(f"✗ Failed to reshape downsampled: {e}")
        
        elif line == "FINISHED":
            print("\n✓ All processing finished!")
            break
        
        elif line.startswith("ERROR"):
            print(f"\n✗ ESP32 Error: {line}")
            break
    
    return images

def read_image_data(ser, expected_bytes, name, timeout=60):
    """Read image data with better timeout handling"""
    data = bytearray()
    start_time = time.time()
    last_update = start_time
    last_data_time = start_time
    
    print(f"  Reading {name}...")
    
    while len(data) < expected_bytes:
        current_time = time.time()
        
        # Check timeout (no data received for 10 seconds)
        if current_time - last_data_time > 10:
            print(f"  ✗ Timeout on {name}! No data for 10s. Got {len(data)}/{expected_bytes} bytes")
            print(f"    Missing {expected_bytes - len(data)} bytes")
            break
        
        # Check total timeout
        if current_time - start_time > timeout:
            print(f"  ✗ Total timeout on {name}! Got {len(data)}/{expected_bytes} bytes")
            break
        
        # Read available data
        if ser.in_waiting:
            to_read = min(ser.in_waiting, expected_bytes - len(data))
            chunk = ser.read(to_read)
            data.extend(chunk)
            last_data_time = current_time
            
            # Progress update every 0.5 seconds
            if current_time - last_update > 0.5:
                percent = (len(data) / expected_bytes) * 100
                print(f"    {len(data)}/{expected_bytes} bytes ({percent:.1f}%)")
                last_update = current_time
        else:
            time.sleep(0.001)
    
    print(f"  {name}: Read {len(data)} bytes total")
    
    if len(data) == expected_bytes:
        print(f"  ✓ {name} complete")
        return bytes(data)
    elif len(data) > 0:
        print(f"  ⚠ {name} incomplete: {len(data)}/{expected_bytes}")
        if len(data) < expected_bytes:
            data.extend([0] * (expected_bytes - len(data)))
        return bytes(data)
    else:
        print(f"  ✗ {name}: No data received")
        return None

def display_and_save_all(images):
    """Display and save all images"""
    print(f"\n=== Received {len(images)} images ===")
    
    # Save all images
    for name, content in images.items():
        if name == 'original':
            img = content
            h, w = img.shape
            filename = f"original_{w}x{h}.png"
            cv2.imwrite(filename, img)
            print(f"✓ Saved: {filename} ({w}x{h})")
        else:
            img = content['img']
            scale = content['scale']
            h, w = img.shape
            
            if 'upsampled' in name:
                filename = f"upsampled_{scale}x_{w}x{h}.png"
            else:
                filename = f"downsampled_{scale}x_{w}x{h}.png"
            
            cv2.imwrite(filename, img)
            print(f"✓ Saved: {filename} ({w}x{h})")
    
    # Display comparison if we have images
    if len(images) >= 2:
        display_comparison(images)

def display_comparison(images):
    """Display comparison of available images"""
    display_h = 250
    display_imgs = []
    labels = []
    
    # Sort images: downsampled, original, upsampled
    img_list = []
    
    # Get original
    if 'original' in images:
        img = images['original']
        h, w = img.shape
        img_list.append(('Original (1.0x)', img, 1.0, w, h))
    
    # Get downsampled
    for key, content in images.items():
        if 'downsampled' in key:
            img = content['img']
            scale = content['scale']
            h, w = img.shape
            img_list.insert(0, (f'Downsampled ({scale}x)', img, float(scale), w, h))
    
    # Get upsampled
    for key, content in images.items():
        if 'upsampled' in key:
            img = content['img']
            scale = content['scale']
            h, w = img.shape
            img_list.append((f'Upsampled ({scale}x)', img, float(scale), w, h))
    
    # Create display images
    for label, img, scale, w, h in img_list:
        # Resize for display
        display_w = int(w * (display_h / h))
        display_img = cv2.resize(img, (display_w, display_h), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Add border and label
        bordered = cv2.copyMakeBorder(display_img, 40, 10, 10, 10, 
                                     cv2.BORDER_CONSTANT, value=200)
        full_label = f"{label}: {w}x{h}"
        cv2.putText(bordered, full_label, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
        
        display_imgs.append(bordered)
        labels.append(label)
    
    if display_imgs:
        # Combine images
        comparison = np.hstack(display_imgs)
        
        # Add title
        title = f"ESP32-CAM Image Resizing ({len(display_imgs)} images)"
        cv2.putText(comparison, title, (20, display_h + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
        
        # Display
        cv2.imshow("Image Comparison", comparison)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"\n✓ Displayed {len(display_imgs)} images")
    else:
        print("\n✗ No images to display")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()