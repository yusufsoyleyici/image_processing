#include "esp_camera.h"
#include <TensorFlowLite_ESP32.h> 
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h" 

// --- PIN DEFINITIONS (AI THINKER) ---
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

// --- CONFIGURATION ---
#define INPUT_W 96
#define INPUT_H 96
#define INPUT_C 3

#define FLASH_GPIO_NUM 4
#define FLASH_DELAY_MS 100

// MEMORY
#define K_ARENA_SIZE 1500000 
uint8_t* tensor_arena = nullptr;

// Global Objects
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

const char* class_names[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP32-CAM Digit Detection ===\n");
  
  pinMode(FLASH_GPIO_NUM, OUTPUT);
  digitalWrite(FLASH_GPIO_NUM, LOW);

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
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 2;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("❌ Camera Init Failed");
    while(1) delay(1000);
  }
  Serial.println("✓ Camera initialized");

  sensor_t * s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 0);
    s->set_contrast(s, 2);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 1);
    s->set_gain_ctrl(s, 1);
    Serial.println("✓ Camera settings optimized");
  }

  tensor_arena = (uint8_t*)ps_malloc(K_ARENA_SIZE);
  if (!tensor_arena) {
    Serial.println("❌ PSRAM allocation failed");
    while(1) delay(1000);
  }
  Serial.println("✓ PSRAM allocated");

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(model_data); 
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("❌ Model version mismatch");
    while(1) delay(1000);
  }
  Serial.println("✓ Model loaded");
  
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, K_ARENA_SIZE, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ Tensor allocation failed");
    while(1) delay(1000);
  }
  Serial.println("✓ Tensors allocated");

  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("\n--- Model Info ---");
  Serial.printf("Input: [%d×%d×%d] %s\n", 
                input->dims->data[1], input->dims->data[2], input->dims->data[3],
                input->type == kTfLiteUInt8 ? "UInt8" : "Other");
  Serial.printf("Output: [%d] %s\n", 
                output->dims->data[1],
                output->type == kTfLiteUInt8 ? "UInt8" : "Other");
  Serial.printf("Quantization: scale=%.6f, zero=%d\n",
                output->params.scale, output->params.zero_point);
  Serial.println("------------------\n");
  
  Serial.println("✓ System Ready!\n");
  Serial.println("Press 'c' to capture and detect digit\n");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'c') {
      runInference();
    } else if (c == 'f') {
      digitalWrite(FLASH_GPIO_NUM, HIGH);
      Serial.println("Flash ON");
    } else if (c == 'o') {
      digitalWrite(FLASH_GPIO_NUM, LOW);
      Serial.println("Flash OFF");
    }
  }
}

void runInference() {
  Serial.println("========================================");
  
  // Flash ON
  digitalWrite(FLASH_GPIO_NUM, HIGH);
  delay(FLASH_DELAY_MS);
  
  Serial.println("📸 Capturing...");
  camera_fb_t * fb = esp_camera_fb_get();
  digitalWrite(FLASH_GPIO_NUM, LOW);
  
  if (!fb) {
    Serial.println("❌ Capture failed");
    return;
  }

  // --- PREPROCESSING ---
  uint8_t * raw = fb->buf;
  int raw_w = fb->width;
  int raw_h = fb->height;
  int x_scale = raw_w / INPUT_W;
  int y_scale = raw_h / INPUT_H;
  
  long sum_total = 0;
  int count_total = 0;
  uint8_t min_v = 255, max_v = 0;

  for (int y = 0; y < INPUT_H; y++) {
    for (int x = 0; x < INPUT_W; x++) {
      
      long sum = 0;
      int count = 0;

      for (int dy = 0; dy < y_scale; dy++) {
        for (int dx = 0; dx < x_scale; dx++) {
          int py = y * y_scale + dy;
          int px = x * x_scale + dx;
          
          if (py < raw_h && px < raw_w) {
            int idx = py * raw_w + px;
            if (idx < fb->len) {
              uint8_t pix = raw[idx];
              sum += pix;
              count++;
              if (pix < min_v) min_v = pix;
              if (pix > max_v) max_v = pix;
            }
          }
        }
      }
      
      uint8_t gray = (count > 0) ? (uint8_t)(sum / count) : 0;
      sum_total += gray;
      count_total++;
      
      // UInt8: No normalization, replicate to RGB
      int idx = (y * INPUT_W + x) * INPUT_C;
      input->data.uint8[idx + 0] = gray;
      input->data.uint8[idx + 1] = gray;
      input->data.uint8[idx + 2] = gray;
    }
  }
  
  esp_camera_fb_return(fb);

  float avg = sum_total / (float)count_total;
  Serial.printf("✓ Image: Min=%d Max=%d Avg=%.1f\n", min_v, max_v, avg);

  // --- INFERENCE ---
  Serial.println("🤖 Running inference...");
  unsigned long start = millis();
  
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("❌ Inference failed");
    return;
  }
  
  unsigned long duration = millis() - start;
  Serial.printf("✓ Inference: %lu ms\n\n", duration);

  // --- CLASSIFICATION (Last 10 values) ---
  int out_size = output->dims->data[1];
  float scale = output->params.scale;
  int zp = output->params.zero_point;
  
  // DIAGNOSTIC: Print raw values of last 10 outputs
  Serial.println("Raw Last 10 UInt8 Values:");
  Serial.print("  ");
  for (int i = 0; i < 10; i++) {
    int idx = out_size - 10 + i;
    Serial.printf("[%d]=%d ", i, output->data.uint8[idx]);
  }
  Serial.println("\n");
  
  // Extract last 10 values as class probabilities
  float class_probs[10];
  float max_prob = 0;
  int predicted_digit = -1;
  
  Serial.println("Class Probabilities (Last 10):");
  Serial.println("--------------------------------");
  
  for (int i = 0; i < 10; i++) {
    int idx = out_size - 10 + i;
    uint8_t raw_val = output->data.uint8[idx];
    class_probs[i] = (raw_val - zp) * scale;
    
    Serial.printf("  [%d] %s: raw=%3d → %.4f", i, class_names[i], raw_val, class_probs[i]);
    
    if (class_probs[i] > max_prob) {
      max_prob = class_probs[i];
      predicted_digit = i;
    }
    
    if (i == predicted_digit) {
      Serial.print(" ← WINNER");
    }
    Serial.println();
  }
  
  // ALSO try first 10 values
  Serial.println("\nClass Probabilities (First 10):");
  Serial.println("--------------------------------");
  
  float first_probs[10];
  float first_max = 0;
  int first_pred = -1;
  
  for (int i = 0; i < 10; i++) {
    uint8_t raw_val = output->data.uint8[i];
    first_probs[i] = (raw_val - zp) * scale;
    
    Serial.printf("  [%d] %s: raw=%3d → %.4f", i, class_names[i], raw_val, first_probs[i]);
    
    if (first_probs[i] > first_max) {
      first_max = first_probs[i];
      first_pred = i;
    }
    
    if (i == first_pred) {
      Serial.print(" ← WINNER");
    }
    Serial.println();
  }
  
  Serial.println("\n⚠️  COMPARISON:");
  Serial.printf("  Last 10: Predicts %s (%.2f%%)\n", class_names[predicted_digit], max_prob * 100);
  Serial.printf("  First 10: Predicts %s (%.2f%%)\n", class_names[first_pred], first_max * 100);
  
  Serial.println("\n========================================");
  
  // Decide which prediction to use
  // Use FIRST 10 by default since they seem to work better for digits 0,1,2
  Serial.println("✓ Using FIRST 10 values (better for all digits)");
  predicted_digit = first_pred;
  max_prob = first_max;
  
  for (int i = 0; i < 10; i++) {
    class_probs[i] = first_probs[i];
  }
  
  Serial.printf("\n🎯 PREDICTION: %s\n", class_names[predicted_digit]);
  Serial.printf("📊 CONFIDENCE: %.2f%%\n", max_prob * 100);
  Serial.println();
  
  // Quality indicators
  if (max_prob > 0.7) {
    Serial.println("✅ High confidence!");
  } else if (max_prob > 0.4) {
    Serial.println("⚠️  Moderate confidence");
  } else {
    Serial.println("❌ Low confidence - try:");
    Serial.println("   • Better lighting/contrast");
    Serial.println("   • Larger, clearer digit");
    Serial.println("   • Center digit in frame");
    Serial.println("   • Verify training data quality");
  }
  
  // Show runner-up
  float second_max = 0;
  int second_class = -1;
  for (int i = 0; i < 10; i++) {
    if (i != predicted_digit && class_probs[i] > second_max) {
      second_max = class_probs[i];
      second_class = i;
    }
  }
  
  if (second_max > max_prob * 0.5) {
    Serial.printf("\n⚠️  Runner-up: %s (%.2f%%) - model is uncertain\n", 
                  class_names[second_class], second_max * 100);
  }
  
  Serial.println("========================================\n");
}