#include "esp_camera.h"

// --- CAMERA PIN DEFINITIONS (AI THINKER) ---
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
#define FRAME_SIZE FRAMESIZE_QVGA

// Increased chunk size for faster transmission
#define CHUNK_SIZE 4096

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("READY");
  Serial.println("Commands:");
  Serial.println("  a - Run with default values (1.5x up, 0.666x down)");
  Serial.println("  u<value> - Upsample only (e.g., u2.0, u1.5)");
  Serial.println("  d<value> - Downsample only (e.g., d0.5, d0.666)");
  Serial.println("  b<up>,<down> - Both (e.g., b1.5,0.666)");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd.length() == 0) return;
    
    char mode = cmd.charAt(0);
    
    if (mode == 'a') {
      sendAllResizings(1.5, 0.6666);
    }
    else if (mode == 'u') {
      float scale = cmd.substring(1).toFloat();
      if (scale > 0 && scale <= 5.0) {
        sendUpsampleOnly(scale);
      } else {
        Serial.println("Invalid upsample value (use 0-5.0)");
      }
    }
    else if (mode == 'd') {
      float scale = cmd.substring(1).toFloat();
      if (scale > 0 && scale < 1.0) {
        sendDownsampleOnly(scale);
      } else {
        Serial.println("Invalid downsample value (use 0-1.0)");
      }
    }
    else if (mode == 'b') {
      int commaPos = cmd.indexOf(',');
      if (commaPos > 0) {
        float upscale = cmd.substring(1, commaPos).toFloat();
        float downscale = cmd.substring(commaPos + 1).toFloat();
        
        if (upscale > 0 && upscale <= 5.0 && downscale > 0 && downscale < 1.0) {
          sendAllResizings(upscale, downscale);
        } else {
          Serial.println("Invalid values (up: 0-5.0, down: 0-1.0)");
        }
      }
    }
  }
}

camera_fb_t* captureImage() {
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
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAME_SIZE;
  config.jpeg_quality = 0;
  config.fb_count = 2;
  
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.println("ERROR: Camera init failed");
    return nullptr;
  }
  
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("ERROR: Frame capture failed");
    esp_camera_deinit();
    return nullptr;
  }
  
  return fb;
}

void sendOriginal(camera_fb_t *fb) {
  Serial.printf("ORIGINAL:%d:%d\n", fb->width, fb->height);
  Serial.flush();
  delay(50);
  
  // Send in larger chunks
  size_t sent = 0;
  while (sent < fb->len) {
    size_t to_send = min((size_t)CHUNK_SIZE, fb->len - sent);
    Serial.write(&fb->buf[sent], to_send);
    sent += to_send;
    
    if (sent % 8192 == 0) {
      Serial.flush();
      delayMicroseconds(500);
    }
  }
  
  Serial.flush();
  delay(100);
  Serial.println("ORIGINAL_DONE");
  delay(50);
}

void sendUpsampled(camera_fb_t *fb, float scale) {
  int w = fb->width;
  int h = fb->height;
  int new_w = (int)(w * scale);
  int new_h = (int)(h * scale);
  
  Serial.printf("UPSAMPLED:%d:%d:%.3f\n", new_w, new_h, scale);
  Serial.flush();
  delay(50);
  
  // Allocate buffer for one row
  uint8_t* row_buffer = (uint8_t*)malloc(new_w);
  if (!row_buffer) {
    Serial.println("ERROR: Memory allocation failed");
    return;
  }
  
  // Process and send row by row
  for (int y = 0; y < new_h; y++) {
    int src_y = (int)(y / scale);
    if (src_y >= h) src_y = h - 1;
    
    // Fill row buffer
    for (int x = 0; x < new_w; x++) {
      int src_x = (int)(x / scale);
      if (src_x >= w) src_x = w - 1;
      row_buffer[x] = fb->buf[src_y * w + src_x];
    }
    
    // Send row
    Serial.write(row_buffer, new_w);
    
    // Periodic flush to prevent buffer overflow
    if (y % 10 == 0) {
      Serial.flush();
      delayMicroseconds(100);
    }
  }
  
  free(row_buffer);
  Serial.flush();
  delay(100);
  Serial.println("UPSAMPLED_DONE");
  delay(50);
}

void sendDownsampled(camera_fb_t *fb, float scale) {
  int w = fb->width;
  int h = fb->height;
  int new_w = (int)(w * scale);
  int new_h = (int)(h * scale);
  
  Serial.printf("DOWNSAMPLED:%d:%d:%.3f\n", new_w, new_h, scale);
  Serial.flush();
  delay(50);
  
  // Allocate buffer for one row
  uint8_t* row_buffer = (uint8_t*)malloc(new_w);
  if (!row_buffer) {
    Serial.println("ERROR: Memory allocation failed");
    return;
  }
  
  // Process and send row by row
  for (int y = 0; y < new_h; y++) {
    int src_y = (int)(y / scale);
    if (src_y >= h) src_y = h - 1;
    
    // Fill row buffer
    for (int x = 0; x < new_w; x++) {
      int src_x = (int)(x / scale);
      if (src_x >= w) src_x = w - 1;
      row_buffer[x] = fb->buf[src_y * w + src_x];
    }
    
    // Send row
    Serial.write(row_buffer, new_w);
    
    // Periodic flush
    if (y % 10 == 0) {
      Serial.flush();
      delayMicroseconds(100);
    }
  }
  
  free(row_buffer);
  Serial.flush();
  delay(100);
  Serial.println("DOWNSAMPLED_DONE");
  delay(50);
}

void sendAllResizings(float upscale, float downscale) {
  camera_fb_t *fb = captureImage();
  if (!fb) return;
  
  Serial.println("START");
  delay(10);
  
  sendOriginal(fb);
  sendUpsampled(fb, upscale);
  sendDownsampled(fb, downscale);
  
  esp_camera_fb_return(fb);
  esp_camera_deinit();
  
  Serial.println("FINISHED");
}

void sendUpsampleOnly(float scale) {
  camera_fb_t *fb = captureImage();
  if (!fb) return;
  
  Serial.println("START");
  delay(10);
  
  sendOriginal(fb);
  sendUpsampled(fb, scale);
  
  esp_camera_fb_return(fb);
  esp_camera_deinit();
  
  Serial.println("FINISHED");
}

void sendDownsampleOnly(float scale) {
  camera_fb_t *fb = captureImage();
  if (!fb) return;
  
  Serial.println("START");
  delay(10);
  
  sendOriginal(fb);
  sendDownsampled(fb, scale);
  
  esp_camera_fb_return(fb);
  esp_camera_deinit();
  
  Serial.println("FINISHED");
}