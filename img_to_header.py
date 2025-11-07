from PIL import Image
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_IMAGE_FILE = 'my_grayscale_image.png'  
OUTPUT_HEADER_FILE = 'image_data.h'
ARRAY_NAME = 'car_image_data'
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 96
# ---------------------

def generate_c_header(image_path, header_path, array_name, width, height):
    """Reads a grayscale image and generates a C header file."""
    try:
        # 1. Load the image and ensure it is 8-bit grayscale ('L' mode)
        img = Image.open(image_path).convert('L')

        if img.size != (width, height):
            print(f"Error: Image size is {img.size}. Expected {width}x{height}.")
            return

        # 2. Convert pixel data to a flat list (array) of integers
        pixel_array = np.array(img).flatten()
        
        # 3. Start writing the header file
        with open(header_path, 'w') as f:
            f.write(f"// Generated from {os.path.basename(image_path)} by Python script\n")
            f.write(f"#ifndef {array_name.upper()}_H\n")
            f.write(f"#define {array_name.upper()}_H\n\n")
            
            f.write(f"#define IMAGE_WIDTH {width}\n")
            f.write(f"#define IMAGE_HEIGHT {height}\n\n")
            
            # Use 'const unsigned char' as pixel values are 0-255 (8-bit)
            f.write(f"const unsigned char {array_name}[IMAGE_HEIGHT * IMAGE_WIDTH] = {{\n")
            
            # 4. Write the pixel values, formatting them for readability (e.g., 20 values per line)
            for i in range(0, len(pixel_array), 20):
                line_data = pixel_array[i:i+20]
                # Join the numbers with commas, and add a tab for indentation
                f.write("    " + ", ".join(map(str, line_data)) + ",\n")
            
            # The last line written has an extra comma, which C compilers usually ignore, 
            # but we remove it for clean code by overwriting the end of the file/string
            f.seek(f.tell() - 2, 0) 
            
            f.write("\n};\n\n")
            f.write("#endif // " + f"{array_name.upper()}_H\n")

        print(f"SUCCESS: C header file '{header_path}' created with {len(pixel_array)} pixel values.")

    except FileNotFoundError:
        print(f"Error: The input file '{image_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the function

generate_c_header(INPUT_IMAGE_FILE, OUTPUT_HEADER_FILE, ARRAY_NAME, IMAGE_WIDTH, IMAGE_HEIGHT)
