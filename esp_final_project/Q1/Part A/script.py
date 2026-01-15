import cv2
import numpy as np

def extract_exact_pixels(image_path, target_count=1000):
    # 1. Load Image in COLOR (to show the true original)
    original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if original_img is None:
        print("Error: Image not found.")
        return

    # 2. Convert to Grayscale for processing
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # 3. Histogram Analysis on the Grayscale image
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    current_count = 0
    threshold_val = 0
    
    # Find the threshold where we cross the target
    for i in range(255, -1, -1):
        count_at_level = int(hist[i][0])
        if current_count + count_at_level >= target_count:
            threshold_val = i
            break
        current_count += count_at_level
        
    # Calculate how many pixels we need from the "borderline" intensity level
    needed_from_threshold_level = target_count - current_count
    
    print(f"Threshold Level: {threshold_val}")
    print(f"Pixels brighter than {threshold_val}: {current_count}")
    print(f"Pixels needed from level {threshold_val}: {needed_from_threshold_level}")

    # 4. Create the Binary Image (The Tie-Breaker)
    binary_img = np.zeros_like(gray_img)
    
    # A. Set all pixels strictly BRIGHTER than threshold to White (255)
    binary_img[gray_img > threshold_val] = 255
    
    # B. Set exactly 'N' pixels AT the threshold to White
    # Get coordinates of all pixels that equal the threshold value
    rows, cols = np.where(gray_img == threshold_val)
    
    # Select the exact number needed to reach 1000
    coords = list(zip(rows, cols))[:needed_from_threshold_level]
    
    for r, c in coords:
        binary_img[r, c] = 255

    # Verify Count
    print(f"Final Total White Pixels: {np.count_nonzero(binary_img)}")

    # 5. Display All Three Stages
    cv2.imshow("1. Original (Color)", original_img)
    cv2.imshow("2. Grayscale Conversion", gray_img)
    cv2.imshow("3. Final Result (Exact 1000)", binary_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage
extract_exact_pixels('Gemini_Generated_Image_9jooi29jooi29joo.png', target_count=1000)