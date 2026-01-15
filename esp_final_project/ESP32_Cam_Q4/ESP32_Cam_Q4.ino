#include "esp_camera.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model files
#include "squeezenet_model.h"
#include "efficientnet_model.h"

// Camera configuration
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// Image settings
#define INPUT_W 32
#define INPUT_H 32
#define INPUT_CH 3

// Tensor arena
#ifdef BOARD_HAS_PSRAM
constexpr int kTensorArenaSize = 81920;
uint8_t* tensor_arena = nullptr;
#else
constexpr int kTensorArenaSize = 40000;
uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
#endif

// Globals
tflite::ErrorReporter* error_reporter = nullptr;
float input_buffer[INPUT_W * INPUT_H * INPUT_CH];

void setup() {
  Serial.begin(115200);
  delay(3000);
  Serial.println("\n=== ESP32-CAM Digit Recognition ===");
  
  // Check PSRAM
  #ifdef BOARD_HAS_PSRAM
  if (psramInit()) {
    Serial.println("PSRAM detected!");
    tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
    if (tensor_arena) {
      Serial.printf("Allocated %d bytes in PSRAM\n", kTensorArenaSize);
    }
  }
  if (!tensor_arena) {
    Serial.println("Falling back to internal RAM");
    tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
  }
  #endif
  
  // Error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  // Camera config
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
  config.frame_size = FRAMESIZE_QQVGA;
  config.jpeg_quality = 10;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
  } else {
    Serial.println("Camera OK");
    
    // MANUAL CAMERA SETTINGS - FIXED, NO AUTO!
    sensor_t * s = esp_camera_sensor_get();
    if (s) {
      // TURN OFF ALL AUTO CONTROLS
      s->set_exposure_ctrl(s, 0);   // AUTO EXPOSURE OFF
      s->set_gain_ctrl(s, 0);       // Auto gain OFF
      s->set_awb_gain(s, 0);        // Auto white balance OFF
      s->set_brightness(s, 0);      // Neutral brightness
      s->set_contrast(s, 2);        // Max contrast
      s->set_saturation(s, 0);      // Neutral saturation
      s->set_sharpness(s, 0);       // Neutral sharpness
      s->set_wb_mode(s, 0);         // White balance OFF
      s->set_dcw(s, 1);             // DCW ON
      
      // MANUAL EXPOSURE SETTING - MINIMUM EXPOSURE (200)
      s->set_aec_value(s, 200);     // Minimum exposure from calibration
      s->set_agc_gain(s, 0);        // Manual gain = 0
      
      Serial.println("Camera: Manual exposure 200 (minimum)");
      Serial.println("Brightness adjusted in software");
    }
  }
  
  esp_log_level_set("*", ESP_LOG_NONE);
  
  Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());
  #ifdef BOARD_HAS_PSRAM
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
  #endif
  
  Serial.println("\n=== READY ===");
  Serial.println("Commands:");
  Serial.println("  'x' - Test camera (white vs black)");
  Serial.println("  'd' - Debug model input (IMPORTANT!)");
  Serial.println("  '1' - Run SqueezeNet");
  Serial.println("  '2' - Run EfficientNet");
  Serial.println("  '3' - Run both models");
  Serial.println("\nFIRST: Run 'd' to see what model actually receives!");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    
    if (cmd == '1') {
      runSingleModel(squeezenet_model, squeezenet_model_len, "SqueezeNet");
    }
    else if (cmd == '2') {
      runSingleModel(efficientnet_model, efficientnet_model_len, "EfficientNet");
    }
    else if (cmd == '3') {
      runBothModels();
    }
    else if (cmd == 'x') {
      testCameraValues();
    }
    else if (cmd == 'd') {
      debugModelInput();
    }
  }
  delay(10);
}

// Debug what the model actually receives
// Debug what the model actually receives - shows ALL 32x32 values
void debugModelInput() {
  Serial.println("\n=== DEBUG MODEL INPUT ===");
  Serial.println("Show a BLACK DIGIT on WHITE paper");
  Serial.println("Press any key when ready...");
  while (!Serial.available()) delay(100);
  Serial.read();
  delay(500);
  
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Capture failed!");
    return;
  }
  
  Serial.printf("Captured: %dx%d\n", fb->width, fb->height);
  
  // First show what camera sees
  int center_x = fb->width / 2;
  int center_y = fb->height / 2;
  uint8_t center_val = fb->buf[center_y * fb->width + center_x];
  Serial.printf("Camera center pixel: %d/255\n", center_val);
  
  // Process image
  processImageDebug(fb);
  esp_camera_fb_return(fb);
  
  // Show FULL 32x32 grid - ALL pixel values 0.0 to 1.0
  Serial.println("\n=== FULL 32x32 PIXEL VALUES ===");
  Serial.println("Format: Row: [32 values 0.0 to 1.0]");
  Serial.println("0.0 = black, 1.0 = white");
  Serial.println("----------------------------------------");
  
  // Print all 32 rows
  for (int y = 0; y < INPUT_H; y++) {
    Serial.printf("Row %2d: ", y);
    for (int x = 0; x < INPUT_W; x++) {
      int idx = (y * INPUT_W + x) * 3;
      float val = input_buffer[idx];
      
      // Print value with 2 decimal places
      if (val == 0.0f) {
        Serial.print("0.00 ");
      } else if (val < 0.1f) {
        Serial.printf("%.2f ", val);
      } else if (val < 1.0f) {
        Serial.printf("%.2f ", val);
      } else {
        Serial.print("1.00 ");
      }
    }
    Serial.println();
    
    // Add separator every 8 rows for readability
    if (y == 7 || y == 15 || y == 23) {
      Serial.println("----------------------------------------");
    }
  }
  
  // Also show ASCII visualization for quick overview
  Serial.println("\n=== ASCII VISUALIZATION ===");
  Serial.println("'#'=white(0.8-1.0), '='=light(0.6-0.8), '-'=medium(0.4-0.6)");
  Serial.println("'.'=dark(0.2-0.4), ' '=black(0.0-0.2)");
  Serial.println("----------------------------------------");
  
  for (int y = 0; y < INPUT_H; y++) {
    Serial.print("  ");
    for (int x = 0; x < INPUT_W; x++) {
      int idx = (y * INPUT_W + x) * 3;
      float val = input_buffer[idx];
      
      char c = ' ';
      if (val > 0.8f) c = '#';
      else if (val > 0.6f) c = '=';
      else if (val > 0.4f) c = '-';
      else if (val > 0.2f) c = '.';
      else c = ' ';
      
      Serial.print(c);
    }
    // Show min/max/avg for this row
    float row_min = 1.0f, row_max = 0.0f, row_sum = 0.0f;
    for (int x = 0; x < INPUT_W; x++) {
      int idx = (y * INPUT_W + x) * 3;
      float val = input_buffer[idx];
      row_sum += val;
      if (val < row_min) row_min = val;
      if (val > row_max) row_max = val;
    }
    Serial.printf(" | R%02d: min=%.2f max=%.2f avg=%.2f", 
                  y, row_min, row_max, row_sum/INPUT_W);
    Serial.println();
  }
  
  // Calculate and show overall statistics
  Serial.println("\n=== OVERALL STATISTICS ===");
  float min_val = 1.0f, max_val = 0.0f, sum = 0.0f;
  int total_pixels = INPUT_W * INPUT_H;
  
  for (int i = 0; i < total_pixels * 3; i += 3) {
    float val = input_buffer[i];
    sum += val;
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
  }
  
  float avg = sum / total_pixels;
  
  Serial.printf("Minimum value: %.3f\n", min_val);
  Serial.printf("Maximum value: %.3f\n", max_val);
  Serial.printf("Average value: %.3f\n", avg);
  Serial.printf("Value range:   %.3f to %.3f (span: %.3f)\n", 
                min_val, max_val, max_val - min_val);
  
  // Count how many pixels in each range
  int count_black = 0, count_dark = 0, count_medium = 0, count_light = 0, count_white = 0;
  for (int i = 0; i < total_pixels * 3; i += 3) {
    float val = input_buffer[i];
    if (val < 0.2f) count_black++;
    else if (val < 0.4f) count_dark++;
    else if (val < 0.6f) count_medium++;
    else if (val < 0.8f) count_light++;
    else count_white++;
  }
  
  Serial.println("\n=== PIXEL DISTRIBUTION ===");
  Serial.printf("Black (0.0-0.2):  %4d pixels (%3d%%)\n", 
                count_black, (count_black * 100) / total_pixels);
  Serial.printf("Dark  (0.2-0.4):  %4d pixels (%3d%%)\n", 
                count_dark, (count_dark * 100) / total_pixels);
  Serial.printf("Medium(0.4-0.6):  %4d pixels (%3d%%)\n", 
                count_medium, (count_medium * 100) / total_pixels);
  Serial.printf("Light (0.6-0.8):  %4d pixels (%3d%%)\n", 
                count_light, (count_light * 100) / total_pixels);
  Serial.printf("White (0.8-1.0):  %4d pixels (%3d%%)\n", 
                count_white, (count_white * 100) / total_pixels);
  
  // Analysis and recommendations
  Serial.println("\n=== ANALYSIS ===");
  
  if (avg < 0.3f && max_val < 0.5f) {
    Serial.println("❌ PROBLEM: Image too DARK overall");
    Serial.println("   Most pixels are black/dark (0.0-0.4)");
    Serial.println("   Need to INCREASE brightness adjustment");
  } 
  else if (avg > 0.7f && min_val > 0.5f) {
    Serial.println("❌ PROBLEM: Image too BRIGHT overall");
    Serial.println("   Most pixels are light/white (0.6-1.0)");
    Serial.println("   Need to DECREASE brightness adjustment");
  }
  else if (max_val - min_val < 0.3f) {
    Serial.println("❌ PROBLEM: Low contrast");
    Serial.println("   All pixels have similar values");
    Serial.println("   Need more contrast between digit and background");
  }
  else if (count_black > total_pixels * 0.7 || count_white > total_pixels * 0.7) {
    Serial.println("⚠ WARNING: Image mostly one color");
    Serial.println("   Too many pixels in same range");
  }
  else {
    Serial.println("✓ GOOD: Image has reasonable brightness and contrast");
    Serial.println("   Should be suitable for digit recognition");
  }
  
  Serial.println("\nExpected ideal: White paper ~0.8-0.9, Black digit ~0.1-0.3");
  Serial.println("Your image: See distribution above");
}

// Modified processImage for debugging
void processImageDebug(camera_fb_t * fb) {
  uint8_t * image_data = fb->buf;
  int raw_width = fb->width;
  int raw_height = fb->height;
  
  int x_block = raw_width / INPUT_W;
  int y_block = raw_height / INPUT_H;
  int index = 0;

  // SIMPLER BRIGHTNESS ADJUSTMENT - LESS AGGRESSIVE!
  for (int y = 0; y < INPUT_H; y++) {
    for (int x = 0; x < INPUT_W; x++) {
      long sum = 0;
      int pixel_count = 0;

      for (int dy = 0; dy < y_block; dy++) {
        for (int dx = 0; dx < x_block; dx++) {
          int p_x = x * x_block + dx;
          int p_y = y * y_block + dy;
          if (p_x >= raw_width || p_y >= raw_height) continue;
          
          sum += image_data[p_y * raw_width + p_x];
          pixel_count++;
        }
      }

      float gray_value = pixel_count > 0 ? (float)(sum / pixel_count) / 255.0f : 0.0f;
      
      // SIMPLE ADJUSTMENT: Your camera white=1.0, black=0.66
      // We want: white=0.9, black=0.2
      // Simple linear mapping
      gray_value = (gray_value - 0.66f) * (0.9f / (1.0f - 0.66f));
      
      // Add some base so black isn't 0
      gray_value = gray_value + 0.2f;
      
      // Clamp
      if (gray_value < 0.0f) gray_value = 0.0f;
      if (gray_value > 1.0f) gray_value = 1.0f;

      input_buffer[index++] = gray_value;
      input_buffer[index++] = gray_value;  
      input_buffer[index++] = gray_value;
    }
  }
  
  // Show ASCII visualization of what model sees
  Serial.println("\nWhat model sees (center 8x8 area):");
  Serial.println("0.0=black, 1.0=white");
  Serial.println("ASCII: ' '=0.0-0.2, '.'=0.2-0.4, '-'=0.4-0.6, '='=0.6-0.8, '#'=0.8-1.0");
  
  for (int y = 12; y <= 19; y++) {
    Serial.print("  ");
    for (int x = 12; x <= 19; x++) {
      int idx = (y * INPUT_W + x) * 3;
      float val = input_buffer[idx];
      
      // ASCII representation
      char c = ' ';
      if (val > 0.8f) c = '#';
      else if (val > 0.6f) c = '=';
      else if (val > 0.4f) c = '-';
      else if (val > 0.2f) c = '.';
      else c = ' ';
      
      Serial.print(c);
    }
    Serial.print("  ");
    for (int x = 12; x <= 19; x++) {
      int idx = (y * INPUT_W + x) * 3;
      float val = input_buffer[idx];
      Serial.printf("%.1f ", val);
    }
    Serial.println();
  }
  
  // Statistics
  float min_val = 1.0f, max_val = 0.0f, sum = 0.0f;
  int total_pixels = INPUT_W * INPUT_H;
  
  for (int i = 0; i < total_pixels * 3; i += 3) {
    float val = input_buffer[i];
    sum += val;
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
  }
  
  float avg = sum / total_pixels;
  
  Serial.printf("\nMODEL INPUT STATISTICS:\n");
  Serial.printf("  Min value: %.3f\n", min_val);
  Serial.printf("  Max value: %.3f\n", max_val);
  Serial.printf("  Average:   %.3f\n", avg);
  Serial.printf("  Range:     %.3f to %.3f\n", min_val, max_val);
  
  if (avg < 0.3f) {
    Serial.println("❌ PROBLEM: Image too DARK overall");
    Serial.println("   Model sees mostly black (0.0-0.3)");
    Serial.println("   Need to INCREASE brightness adjustment");
  } else if (avg > 0.7f) {
    Serial.println("❌ PROBLEM: Image too BRIGHT overall");
    Serial.println("   Model sees mostly white (0.7-1.0)");
    Serial.println("   Need to DECREASE brightness adjustment");
  } else if (max_val - min_val < 0.3f) {
    Serial.println("❌ PROBLEM: Low contrast");
    Serial.println("   All pixels similar - no digit visible");
  } else {
    Serial.println("✓ GOOD: Image has reasonable contrast");
    Serial.println("   Should work for digit recognition");
  }
  
  Serial.println("\nExpected: White paper ~0.8, Black digit ~0.2");
}

// Test camera
void testCameraValues() {
  Serial.println("\n=== CAMERA TEST ===");
  
  Serial.println("1. Point at WHITE paper");
  Serial.println("2. Press any key...");
  while (!Serial.available()) delay(100);
  Serial.read();
  delay(500);
  
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Capture failed!");
    return;
  }
  
  int center_x = fb->width / 2;
  int center_y = fb->height / 2;
  uint8_t white_val = fb->buf[center_y * fb->width + center_x];
  
  Serial.printf("WHITE paper: %d/255\n", white_val);
  esp_camera_fb_return(fb);
  
  Serial.println("\n3. Point at BLACK digit");
  Serial.println("4. Press any key...");
  while (!Serial.available()) delay(100);
  Serial.read();
  delay(500);
  
  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Capture failed!");
    return;
  }
  
  uint8_t black_val = fb->buf[center_y * fb->width + center_x];
  Serial.printf("BLACK digit: %d/255\n", black_val);
  
  Serial.printf("\nWhite=%d, Black=%d, Difference=%d\n", 
                white_val, black_val, white_val - black_val);
  
  esp_camera_fb_return(fb);
}

// Process image for actual inference (same as debug but without ASCII)
void processImage(camera_fb_t * fb) {
  uint8_t * image_data = fb->buf;
  int raw_width = fb->width;
  int raw_height = fb->height;
  
  int x_block = raw_width / INPUT_W;
  int y_block = raw_height / INPUT_H;
  int index = 0;

  for (int y = 0; y < INPUT_H; y++) {
    for (int x = 0; x < INPUT_W; x++) {
      long sum = 0;
      int pixel_count = 0;

      for (int dy = 0; dy < y_block; dy++) {
        for (int dx = 0; dx < x_block; dx++) {
          int p_x = x * x_block + dx;
          int p_y = y * y_block + dy;
          if (p_x >= raw_width || p_y >= raw_height) continue;
          
          sum += image_data[p_y * raw_width + p_x];
          pixel_count++;
        }
      }

      float gray_value = pixel_count > 0 ? (float)(sum / pixel_count) / 255.0f : 0.0f;
      
      // SAME ADJUSTMENT AS IN processImageDebug
      gray_value = (gray_value - 0.66f) * (0.9f / (1.0f - 0.66f));
      gray_value = gray_value + 0.2f;
      
      if (gray_value < 0.0f) gray_value = 0.0f;
      if (gray_value > 1.0f) gray_value = 1.0f;

      input_buffer[index++] = gray_value;
      input_buffer[index++] = gray_value;  
      input_buffer[index++] = gray_value;
    }
  }
}

bool loadAndRunModel(const unsigned char* model_data, unsigned int model_len, 
                     const char* model_name, float* predictions) {
  Serial.printf("\nLoading %s... ", model_name);
  
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema mismatch!\n");
    return false;
  }
  
  static tflite::AllOpsResolver resolver;
  
  if (tensor_arena) {
    memset(tensor_arena, 0, kTensorArenaSize);
  }
  
  tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  
  TfLiteStatus alloc_status = interpreter.AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.printf("FAILED - AllocateTensors error\n");
    return false;
  }
  
  TfLiteTensor* input = interpreter.input(0);
  
  Serial.printf("OK\nRunning inference...");
  
  for (int i = 0; i < INPUT_W * INPUT_H * INPUT_CH; i++) {
    input->data.f[i] = input_buffer[i];
  }
  
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println(" FAILED");
    return false;
  }
  
  TfLiteTensor* output = interpreter.output(0);
  for (int i = 0; i < 10; i++) {
    predictions[i] = output->data.f[i];
  }
  
  Serial.println(" Done");
  return true;
}

void runSingleModel(const unsigned char* model_data, unsigned int model_len, 
                    const char* model_name) {
  Serial.println("\n--- CAPTURING IMAGE ---");
  
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Capture failed!");
    return;
  }
  
  Serial.printf("Captured: %dx%d\n", fb->width, fb->height);
  processImage(fb);
  esp_camera_fb_return(fb);
  
  float predictions[10];
  if (loadAndRunModel(model_data, model_len, model_name, predictions)) {
    int best_class = 0;
    float best_conf = predictions[0];
    
    Serial.println("\nPredictions:");
    for (int i = 0; i < 10; i++) {
      Serial.printf("  Digit %d: %.1f%%\n", i, predictions[i] * 100);
      if (predictions[i] > best_conf) {
        best_conf = predictions[i];
        best_class = i;
      }
    }
    
    Serial.println("\n=== RESULT ===");
    Serial.printf("%s: Digit %d (%.1f%% confidence)\n", model_name, best_class, best_conf * 100);
    Serial.println("===============\n");
  } else {
    Serial.println("\nModel execution failed!");
  }
}

void runBothModels() {
  Serial.println("\n--- CAPTURING IMAGE ---");
  
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Capture failed!");
    return;
  }
  
  Serial.printf("Captured: %dx%d\n", fb->width, fb->height);
  processImage(fb);
  esp_camera_fb_return(fb);
  
  float sq_pred[10];
  Serial.println("\n[1/2] Running SqueezeNet...");
  bool sq_ok = loadAndRunModel(squeezenet_model, squeezenet_model_len, "SqueezeNet", sq_pred);
  
  delay(100);
  
  float eff_pred[10];
  Serial.println("\n[2/2] Running EfficientNet...");
  bool eff_ok = loadAndRunModel(efficientnet_model, efficientnet_model_len, "EfficientNet", eff_pred);
  
  if (!sq_ok || !eff_ok) {
    Serial.println("\nERROR: One or both models failed");
    return;
  }
  
  int sq_class = 0, eff_class = 0;
  float sq_max = sq_pred[0], eff_max = eff_pred[0];
  
  for (int i = 0; i < 10; i++) {
    if (sq_pred[i] > sq_max) {
      sq_max = sq_pred[i];
      sq_class = i;
    }
    if (eff_pred[i] > eff_max) {
      eff_max = eff_pred[i];
      eff_class = i;
    }
  }
  
  Serial.println("\n=== INDIVIDUAL MODELS ===");
  Serial.printf("SqueezeNet:   Digit %d (%.1f%%)\n", sq_class, sq_max * 100);
  Serial.printf("EfficientNet: Digit %d (%.1f%%)\n", eff_class, eff_max * 100);
  
  float best_conf = 0;
  int best_class = -1;
  
  Serial.println("\nFusion predictions:");
  for (int i = 0; i < 10; i++) {
    float merged = (sq_pred[i] + eff_pred[i]) / 2.0f;
    Serial.printf("  Digit %d: %.1f%%\n", i, merged * 100);
    if (merged > best_conf) {
      best_conf = merged;
      best_class = i;
    }
  }
  
  Serial.println("\n=== FUSED RESULT ===");
  Serial.printf("Digit: %d (%.1f%% confidence)\n", best_class, best_conf * 100);
  
  if (sq_class == eff_class && eff_class == best_class) {
    Serial.println("✓ All methods agree");
  } else {
    Serial.println("Note: Models gave different results");
  }
  
  Serial.println("===========================\n");
}