/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : HW4 Q1 - Handwritten Digit Recognition (Single Neuron)
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "lib_image.h"
#include "lib_serialimage.h"
#include <stdio.h>
#include <math.h>

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
// Q1: Weights and Bias from your Python Training
float features_mean[7] = { 0.334226f, 0.044768f, 0.008187f, 0.001415f, 0.000007f, 0.000156f, -0.000006f };
float features_std[7]  = { 0.084046f, 0.063766f, 0.014061f, 0.002583f, 0.000082f, 0.000721f, 0.000063f };
float neuron_weights[7] = { -1.716061f, 4.036231f, 8.247725f, 4.716061f, -1.068714f, -1.818773f, 1.581982f };
float neuron_bias       = 5.926459f;

// Buffers
// We use 28x28 for MNIST. The buffer is large enough for safety.
volatile uint8_t pImage[128 * 128 * 3];
uint8_t pImageOutput[128 * 128]; // Not strictly used for prediction, but needed for struct

IMAGE_HandleTypeDef img_input;
IMAGE_HandleTypeDef img_output;

// MNIST Dimensions
#define IMG_W 28
#define IMG_H 28
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);

/* USER CODE BEGIN 0 */

// --- MATH HELPER FUNCTIONS ---
double power(double base, int exp) {
    double res = 1.0;
    for (int i = 0; i < exp; i++) res *= base;
    return res;
}

// --- CORE LOGIC: HU MOMENTS & PREDICTION ---
void Run_Digit_Recognition(uint8_t *img_data) {

    printf("\n--- STARTING RECOGNITION ---\r\n");

    // 1. Calculate Raw Moments (m_pq)
    // Treating pixels > 0 as 1 (Binary)
    double m00=0, m10=0, m01=0;

    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            // Arrays are 1D in memory: index = y * width + x
            uint8_t pixel = img_data[y * IMG_W + x];
            if (pixel > 0) {
                double val = 1.0;
                m00 += val;
                m10 += x * val;
                m01 += y * val;
            }
        }
    }

    if (m00 == 0) {
        printf("Error: Image is completely black.\r\n");
        return;
    }

    // Centroid
    double cx = m10 / m00;
    double cy = m01 / m00;

    // 2. Central Moments (mu_pq)
    double mu20=0, mu02=0, mu11=0, mu30=0, mu12=0, mu21=0, mu03=0;

    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            uint8_t pixel = img_data[y * IMG_W + x];
            if (pixel > 0) {
                double val = 1.0;
                double dx = x - cx;
                double dy = y - cy;

                mu20 += power(dx, 2) * power(dy, 0) * val;
                mu02 += power(dx, 0) * power(dy, 2) * val;
                mu11 += power(dx, 1) * power(dy, 1) * val;
                mu30 += power(dx, 3) * power(dy, 0) * val;
                mu12 += power(dx, 1) * power(dy, 2) * val;
                mu21 += power(dx, 2) * power(dy, 1) * val;
                mu03 += power(dx, 0) * power(dy, 3) * val;
            }
        }
    }

    // 3. Normalized Central Moments (nu_pq)
    // nu_pq = mu_pq / m00^((p+q)/2 + 1)
    double inv_m00 = 1.0 / m00;
    double nu20 = mu20 * pow(inv_m00, 2);
    double nu02 = mu02 * pow(inv_m00, 2);
    double nu11 = mu11 * pow(inv_m00, 2);
    double nu30 = mu30 * pow(inv_m00, 2.5);
    double nu12 = mu12 * pow(inv_m00, 2.5);
    double nu21 = mu21 * pow(inv_m00, 2.5);
    double nu03 = mu03 * pow(inv_m00, 2.5);

    // 4. Calculate 7 Hu Moments
    double hu[7];
    double t1 = nu20 + nu02;
    double t2 = nu20 - nu02;
    double t3 = nu30 - 3 * nu12;
    double t4 = 3 * nu21 - nu03;
    double t5 = nu30 + nu12;
    double t6 = nu21 + nu03;

    hu[0] = t1;
    hu[1] = t2 * t2 + 4 * nu11 * nu11;
    hu[2] = t3 * t3 + t4 * t4;
    hu[3] = t5 * t5 + t6 * t6;
    hu[4] = t3 * t5 * (t5 * t5 - 3 * t6 * t6) + t4 * t6 * (3 * t5 * t5 - t6 * t6);
    hu[5] = t2 * (t5 * t5 - t6 * t6) + 4 * nu11 * t5 * t6;
    hu[6] = t4 * t5 * (t5 * t5 - 3 * t6 * t6) - t3 * t6 * (3 * t5 * t5 - t6 * t6);

    // 5. Run the Neuron (Inference)
    float z = neuron_bias;

    for (int i = 0; i < 7; i++) {
        // Normalize using Python parameters
        float norm_val = ((float)hu[i] - features_mean[i]) / features_std[i];
        // Dot Product
        z += norm_val * neuron_weights[i];
    }

    // Sigmoid Activation
    float prediction = 1.0f / (1.0f + expf(-z));

    // 6. Output Result
    printf("Raw Probability: %.4f\r\n", prediction);

    if (prediction > 0.5f) {
        printf(">>> PREDICTION: [ NOT A ZERO ] (Digit 1-9)\r\n");
    } else {
        printf(">>> PREDICTION: [ ZERO ] (Digit 0)\r\n");
    }
    printf("------------------------------\r\n");
}
/* USER CODE END 0 */

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_USART2_UART_Init();

  /* USER CODE BEGIN 2 */
  // Initialize for 28x28 Image (Standard MNIST size)
  LIB_IMAGE_InitStruct(&img_input, (uint8_t*)pImage, IMG_W, IMG_H, IMAGE_FORMAT_GRAYSCALE);
  LIB_IMAGE_InitStruct(&img_output, pImageOutput, IMG_W, IMG_H, IMAGE_FORMAT_GRAYSCALE);

  printf("STM32 Ready for Digit Recognition (28x28)...\r\n");
  /* USER CODE END 2 */

  while (1)
  {
    /* USER CODE BEGIN 3 */
    // Wait for image from PC
    if (LIB_SERIAL_IMG_Receive(&img_input) == SERIAL_OK)
    {
        // 1. Run the AI on the received image
        Run_Digit_Recognition(img_input.pData);

        // 2. Send the image back (Echo) so the Python script finishes its handshake
        // We just send back what we received.
        LIB_SERIAL_IMG_Transmit(&img_input);
    }
    /* USER CODE END 3 */
  }
}

// ... (Keep your SystemClock_Config, MX_USART2, MX_GPIO, Error_Handler as they were)
// ... Copy them from your original code if needed, or I can paste them if you lost them.
void SystemClock_Config(void) {
  // Use the same SystemClock_Config from your previous working code
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) Error_Handler();
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK|RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK) Error_Handler();
}
static void MX_USART2_UART_Init(void) {
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 2000000; // Keep high baudrate for images
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK) Error_Handler();
}
static void MX_GPIO_Init(void) {
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
}
void Error_Handler(void) { __disable_irq(); while (1); }
