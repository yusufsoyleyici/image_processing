# image_processing
Introduction to Embedded Image Processing course homework repository.
We have our 128x96 my_grayscale_image.png file and we have used img_to_header.py file in order to convert that image into a header file which is image_data.h as in the repository.
In our main.c file, we perform all the image processing operations. All the calculations are finished before the code enters the main while(1) loop.
We store the original image in the car_image_data[] array, which is loaded from image_data.h.
For Q1-a, to show the image is in memory, we read the first pixel car_image_data[0] and store it in a variable called test_pixel_value.
For Q2, we created new global arrays to store the results of each transformation.
Q2-a: Negative Image
We created the negative_image[] array. We looped through the original image and calculated each new pixel as 255 - car_image_data[i].
Q2-b: Thresholding
We created the threshold_image[] array. We used a threshold of 128. If the original pixel was > 128, we set the new pixel to 255. Otherwise, we set it to 0.
Q2-c: Gamma Correction
We created two arrays: gamma_image_3[] (for gamma = 3.0) and gamma_image_1_3[] (for gamma = 1/3). To do this, we first normalized the pixel value to a float (dividing by 255.0). Then we used the powf() function from math.h with the gamma value. Finally, we multiplied by 255.0 to convert it back to an unsigned char.
Q2-d: Piecewise Linear Transformation
We created the piecewise_image[] array. This code performs contrast stretching based on the thresholding in part b.
    If the pixel is < 50, it is set to 0.
    If the pixel is > 200, it is set to 255.
    If the pixel is between 50 and 200, we stretch its value to fit the full 0-255 range.
We become sure that our program works correctly by starting debugging and stopping the program inside while loop, then looking for the necessary locations inside the memory browser by comparing the hexadecimal numbers with our calculated numbers.
