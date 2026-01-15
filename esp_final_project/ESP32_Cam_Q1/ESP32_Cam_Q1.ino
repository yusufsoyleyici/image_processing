#include "esp_camera.h"
#include <Arduino.h>

// Pin definitions for AI-Thinker ESP32-CAM
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Flash LED pin for AI-Thinker ESP32-CAM
#define FLASH_GPIO_NUM     4

#define TARGET_PIXELS 1000
#define IMAGE_WIDTH 320
#define IMAGE_HEIGHT 240
#define TOTAL_PIXELS (IMAGE_WIDTH * IMAGE_HEIGHT)

// Global buffer for image processing
uint8_t image_buffer[TOTAL_PIXELS];

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("ESP32-CAM Grayscale Thresholding System");
  
  // Initialize flash LED pin
  pinMode(FLASH_GPIO_NUM, OUTPUT);
  digitalWrite(FLASH_GPIO_NUM, LOW);  // Make sure it starts OFF
  
  // Test flash on startup
  Serial.println("Testing flash LED...");
  digitalWrite(FLASH_GPIO_NUM, HIGH);
  delay(500);
  digitalWrite(FLASH_GPIO_NUM, LOW);
  Serial.println("Flash test complete");
  
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  
  // Use GRAYSCALE format directly - most reliable
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_LATEST;
  
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    while(1);
  }
  
  // Set special controls for better grayscale
  sensor_t *s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_QVGA);
  s->set_contrast(s, 2);      // Increase contrast
  s->set_brightness(s, 0);
  s->set_saturation(s, -2);   // Reduce saturation
  
  // Discard first few frames to let camera stabilize
  for (int i = 0; i < 3; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) {
      esp_camera_fb_return(fb);
      delay(100);
    }
  }
  
  Serial.println("READY");
}

void processImage() {
  Serial.println("GET_READY");
  Serial.flush();
  delay(2000);  // Give user 2 seconds warning to position camera
  
  Serial.println("CAPTURING_NOW");
  Serial.flush();
  
  // Turn ON flash and capture IMMEDIATELY
  digitalWrite(FLASH_GPIO_NUM, HIGH);
  Serial.println("FLASH_LED_ON");
  delay(100);  // Small delay to let flash stabilize
  
  camera_fb_t *fb = esp_camera_fb_get();
  
  // Turn OFF flash immediately after capture
  digitalWrite(FLASH_GPIO_NUM, LOW);
  Serial.println("FLASH_LED_OFF");
  Serial.println("CAPTURE_DONE");
  
  if (!fb) {
    Serial.println("CAPTURE_FAILED");
    return;
  }
  
  // Copy grayscale data to buffer
  int pixel_count = min((int)fb->len, TOTAL_PIXELS);
  memcpy(image_buffer, fb->buf, pixel_count);
  
  // If we didn't get enough pixels, fill the rest
  if (pixel_count < TOTAL_PIXELS) {
    memset(image_buffer + pixel_count, 128, TOTAL_PIXELS - pixel_count);
  }
  
  Serial.print("PROCESSED:");
  Serial.println(pixel_count);
  
  // Calculate histogram
  int histogram[256] = {0};
  for (int i = 0; i < TOTAL_PIXELS; i++) {
    histogram[image_buffer[i]]++;
  }
  
  // Find threshold for exactly 1000 white pixels
  int count = 0;
  int threshold = 0;
  for (int i = 255; i >= 0; i--) {
    if (count + histogram[i] >= TARGET_PIXELS) {
      threshold = i;
      break;
    }
    count += histogram[i];
  }
  
  int needed = TARGET_PIXELS - count;
  
  Serial.print("THRESHOLD:");
  Serial.println(threshold);
  Serial.print("ABOVE:");
  Serial.println(count);
  Serial.print("NEEDED:");
  Serial.println(needed);
  
  // Send original grayscale image
  Serial.println("ORIGINAL_START");
  Serial.print("WIDTH:");
  Serial.println(IMAGE_WIDTH);
  Serial.print("HEIGHT:");
  Serial.println(IMAGE_HEIGHT);
  Serial.flush();
  
  // Send image data in chunks
  int chunk_size = 1024;
  for (int i = 0; i < TOTAL_PIXELS; i += chunk_size) {
    int bytes_to_send = min(chunk_size, TOTAL_PIXELS - i);
    Serial.write(image_buffer + i, bytes_to_send);
    Serial.flush();
    delay(10);  // Small delay to prevent buffer overflow
  }
  
  Serial.println();
  Serial.println("ORIGINAL_END");
  Serial.flush();
  delay(100);
  
  // Create thresholded image
  Serial.println("THRESHOLDED_START");
  Serial.print("WIDTH:");
  Serial.println(IMAGE_WIDTH);
  Serial.print("HEIGHT:");
  Serial.println(IMAGE_HEIGHT);
  Serial.flush();
  
  int taken = 0;
  int white_count = 0;
  
  // Send thresholded image in chunks
  for (int i = 0; i < TOTAL_PIXELS; i += chunk_size) {
    uint8_t output_buffer[chunk_size];
    int bytes_to_send = min(chunk_size, TOTAL_PIXELS - i);
    
    for (int j = 0; j < bytes_to_send; j++) {
      int idx = i + j;
      uint8_t pixel = image_buffer[idx];
      uint8_t output;
      
      if (pixel > threshold) {
        output = 255;
        white_count++;
      } else if (pixel < threshold) {
        output = 0;
      } else {
        if (taken < needed) {
          output = 255;
          white_count++;
          taken++;
        } else {
          output = 0;
        }
      }
      
      output_buffer[j] = output;
    }
    
    Serial.write(output_buffer, bytes_to_send);
    Serial.flush();
    delay(10);  // Small delay to prevent buffer overflow
  }
  
  Serial.println();
  Serial.println("THRESHOLDED_END");
  Serial.flush();
  
  Serial.print("WHITE_PIXELS:");
  Serial.println(white_count);
  
  esp_camera_fb_return(fb);
  
  Serial.println("DONE");
  Serial.flush();
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    Serial.print("RECEIVED_COMMAND: [");
    Serial.print(cmd);
    Serial.println("]");
    
    if (cmd == "CAPTURE") {
      processImage();
    }
    else if (cmd == "TEST") {
      Serial.println("TEST_OK");
    }
    else if (cmd == "FLASH_TEST") {
      Serial.println("Flash ON for 2 seconds...");
      digitalWrite(FLASH_GPIO_NUM, HIGH);
      delay(2000);
      digitalWrite(FLASH_GPIO_NUM, LOW);
      Serial.println("Flash OFF");
    }
    else {
      Serial.print("UNKNOWN_COMMAND: ");
      Serial.println(cmd);
    }
  }
  
  delay(10);
}