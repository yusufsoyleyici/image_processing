/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body - HW6 (SqueezeNet & EfficientNet)
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "my_network.h"       // Auto-generated AI Model
#include "my_network_data.h"  // Auto-generated AI Weights
#include "ai_platform.h"      // AI Runtime
#include <stdio.h>
#include <string.h>

/* --- HOMEWORK CONFIGURATION --- */
// Uncomment ONE of the lines below to match the model you flashed in CubeMX
#define MODEL_TYPE_EFFICIENTNET
// #define MODEL_TYPE_SQUEEZENET

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;
CRC_HandleTypeDef hcrc;  // <--- Added CRC Handle

// --- AI HANDLES & BUFFERS ---
ai_handle network = AI_HANDLE_NULL;

AI_ALIGNED(4) ai_u8 activations[AI_MY_NETWORK_DATA_ACTIVATIONS_SIZE];
AI_ALIGNED(4) ai_float in_data[AI_MY_NETWORK_IN_1_SIZE];
AI_ALIGNED(4) ai_float out_data[AI_MY_NETWORK_OUT_1_SIZE];

// UART Buffer
uint8_t uart_rx_buffer[3072];

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_CRC_Init(void); // <--- Added CRC Prototype

/* --- AI INITIALIZATION --- */
int ai_init(void) {
    ai_error err;

    // Create the network
    err = ai_my_network_create(&network, AI_MY_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE) {
        printf("E: Create network failed (Code: %d)\r\n", err.type);
        return -1;
    }

    // Initialize parameters
    const ai_network_params params = {
        .params = AI_MY_NETWORK_DATA_WEIGHTS(ai_my_network_data_weights_get()),
        .activations = AI_MY_NETWORK_DATA_ACTIVATIONS(activations)
    };

    if (!ai_my_network_init(network, &params)) {
        printf("E: Init network failed\r\n");
        return -1;
    }

    return 0;
}

/* --- AI INFERENCE --- */
int ai_run(void) {
    ai_i32 batch;

    ai_buffer* ai_input = ai_my_network_inputs_get(network, NULL);
    ai_buffer* ai_output = ai_my_network_outputs_get(network, NULL);

    ai_input[0].data = AI_HANDLE_PTR(in_data);
    ai_output[0].data = AI_HANDLE_PTR(out_data);

    batch = ai_my_network_run(network, ai_input, ai_output);

    if (batch != 1) {
        printf("E: Run failed (Code: %ld)\r\n", (long)batch);
        return -1;
    }
    return 0;
}

/**
  * @brief  The application entry point.
  */
int main(void)
{
  // MCU Config
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_CRC_Init(); // <--- Initialize CRC hardware

  // --- STARTUP LOGIC ---
  printf("\r\n==========================================\r\n");
#ifdef MODEL_TYPE_EFFICIENTNET
  printf("   STM32 HW6: MINI-EFFICIENTNET MODEL\r\n");
  printf("   Architecture: MBConv + Swish\r\n");
#elif defined(MODEL_TYPE_SQUEEZENET)
  printf("   STM32 HW6: SQUEEZENET MODEL\r\n");
  printf("   Architecture: Fire Modules\r\n");
#endif
  printf("==========================================\r\n");

  if(ai_init() == 0) {
      printf("AI Init: SUCCESS!\r\n");
      printf("Waiting for image data (3072 bytes)...\r\n");
  } else {
      printf("AI Init: FAILED. System Halted.\r\n");
      while(1);
  }

  // --- MAIN LOOP ---
  while (1)
  {
      // HEARTBEAT: Blink Green LED (PA5) to show board is alive
      HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);

      // 1. Receive Image (Blocking)
      if (HAL_UART_Receive(&huart2, uart_rx_buffer, 3072, HAL_MAX_DELAY) == HAL_OK)
      {
          printf("Image Received. Preprocessing...\r\n");

          // 2. Preprocess
          for (int i = 0; i < 3072; i++) {
              in_data[i] = (float)uart_rx_buffer[i] / 255.0f;
          }

          // 3. Run Inference
          if (ai_run() == 0) {
              float max_conf = 0.0f;
              int digit = -1;

              // 4. Post-process
              for (int i = 0; i < 10; i++) {
                  if (out_data[i] > max_conf) {
                      max_conf = out_data[i];
                      digit = i;
                  }
              }

              // 5. Report Result
              #ifdef MODEL_TYPE_EFFICIENTNET
              printf("[EfficientNet] PREDICTION: %d (Conf: %.2f)\r\n", digit, max_conf);
              #else
              printf("[SqueezeNet] PREDICTION: %d (Conf: %.2f)\r\n", digit, max_conf);
              #endif
          }
      }
  }
}

/* --- SYSTEM CONFIGURATION --- */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 180;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  HAL_RCC_OscConfig(&RCC_OscInitStruct);
  HAL_PWREx_EnableOverDrive();
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK|RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;
  HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5);
}

static void MX_CRC_Init(void)
{
  hcrc.Instance = CRC;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_USART2_UART_Init(void)
{
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  HAL_UART_Init(&huart2);
}

static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  __HAL_RCC_GPIOA_CLK_ENABLE();

  // Configure Green LED (PA5)
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}

void Error_Handler(void) { __disable_irq(); while (1); }

/* --- SYSCALL STUBS --- */
int _write(int file, char *ptr, int len) {
    HAL_UART_Transmit(&huart2, (uint8_t*)ptr, len, 100);
    return len;
}
int _read(int file, char *ptr, int len) { return 0; }
int _close(int file) { return -1; }
int _lseek(int file, int ptr, int dir) { return 0; }
int _fstat(int file, void *st) { return 0; }
int _isatty(int file) { return 1; }
