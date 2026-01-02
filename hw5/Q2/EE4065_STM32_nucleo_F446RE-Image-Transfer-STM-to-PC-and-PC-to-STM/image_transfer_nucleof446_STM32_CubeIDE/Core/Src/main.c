/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : HW4 Q2 - Section 11.8 Multi-Layer Neural Network (Complete)
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "lib_image.h"
#include "lib_serialimage.h"
#include <stdio.h>
#include <math.h>

// 1. INCLUDE THE WEIGHTS FILE
#include "weights_data.h"

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2; // This is the handle the linker was looking for

// Buffers
volatile uint8_t pImage[128 * 128 * 3];
uint8_t pImageOutput[128 * 128];

IMAGE_HandleTypeDef img_input;
IMAGE_HandleTypeDef img_output;

// MNIST Dimensions
#define IMG_W 28
#define IMG_H 28

// NETWORK DEFINITIONS
#define L1_IN 7
#define L1_OUT 100
#define L2_OUT 100
#define L3_OUT 10

// Buffers for hidden layers
float layer1_out[L1_OUT];
float layer2_out[L2_OUT];
float final_out[L3_OUT];

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);

// --- MATH HELPER FUNCTIONS ---
double power(double base, int exp) {
    double res = 1.0;
    for (int i = 0; i < exp; i++) res *= base;
    return res;
}

// ReLU Activation: max(0, x)
float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

// Dense Layer Logic (Inputs -> Outputs)
// Weights are stored flat. We assume row-major mapping based on how we exported them.
void dense_layer(const float* input, int input_size,
                 float* output, int output_size,
                 const float* weights, const float* bias,
                 int use_relu)
{
    for (int n = 0; n < output_size; n++) {
        float sum = bias[n];

        // Dot product for Neuron 'n'
        for (int i = 0; i < input_size; i++) {
            // Mapping: weights[i * output_size + n]
            // This assumes the Python script flattened them correctly.
            sum += input[i] * weights[i * output_size + n];
        }

        if (use_relu) {
            output[n] = relu(sum);
        } else {
            output[n] = sum; // Raw output (logits) for final layer
        }
    }
}

// Softmax for final layer
void softmax(float* input, int size) {
    float sum = 0.0f;
    float max_val = -99999.0f;

    // Numerical Stability: Find max
    for(int i=0; i<size; i++) if(input[i] > max_val) max_val = input[i];

    // Exponentials
    for(int i=0; i<size; i++) {
        input[i] = expf(input[i] - max_val);
        sum += input[i];
    }

    // Normalize
    for(int i=0; i<size; i++) {
        input[i] /= sum;
    }
}

// --- CORE LOGIC ---
void Run_Neural_Network(uint8_t *img_data) {
    printf("\n--- Q2 MLP INFERENCE ---\r\n");

    // 1. Calculate Features (Hu Moments)
    double m00=0, m10=0, m01=0;
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            // Treat pixel > 0 as binary 1
            if (img_data[y * IMG_W + x] > 0) {
                m00 += 1.0; m10 += x; m01 += y;
            }
        }
    }

    if(m00 == 0) {
        printf("Error: Image is blank.\r\n");
        return;
    }

    double cx = m10/m00;
    double cy = m01/m00;

    // Central Moments
    double mu[7] = {0}; // mu20, mu02, mu11, mu30, mu12, mu21, mu03

    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            if (img_data[y * IMG_W + x] > 0) {
                double dx = x - cx; double dy = y - cy;
                mu[0] += dx*dx;     // mu20
                mu[1] += dy*dy;     // mu02
                mu[2] += dx*dy;     // mu11
                mu[3] += dx*dx*dx;  // mu30
                mu[4] += dx*dy*dy;  // mu12
                mu[5] += dx*dx*dy;  // mu21
                mu[6] += dy*dy*dy;  // mu03
            }
        }
    }

    // Normalized Central Moments
    double inv_m00 = 1.0 / m00;
    double nu[7];
    nu[0] = mu[0] * pow(inv_m00, 2);
    nu[1] = mu[1] * pow(inv_m00, 2);
    nu[2] = mu[2] * pow(inv_m00, 2);
    nu[3] = mu[3] * pow(inv_m00, 2.5);
    nu[4] = mu[4] * pow(inv_m00, 2.5);
    nu[5] = mu[5] * pow(inv_m00, 2.5);
    nu[6] = mu[6] * pow(inv_m00, 2.5);

    // Hu Moments
    float features[7];
    double t1 = nu[0] + nu[1];
    double t2 = nu[0] - nu[1];
    double t3 = nu[3] - 3 * nu[4];
    double t4 = 3 * nu[5] - nu[6];
    double t5 = nu[3] + nu[4];
    double t6 = nu[5] + nu[6];

    features[0] = (float)t1;
    features[1] = (float)(t2 * t2 + 4 * nu[2] * nu[2]);
    features[2] = (float)(t3 * t3 + t4 * t4);
    features[3] = (float)(t5 * t5 + t6 * t6);
    features[4] = (float)(t3 * t5 * (t5 * t5 - 3 * t6 * t6) + t4 * t6 * (3 * t5 * t5 - t6 * t6));
    features[5] = (float)(t2 * (t5 * t5 - t6 * t6) + 4 * nu[2] * t5 * t6);
    features[6] = (float)(t4 * t5 * (t5 * t5 - 3 * t6 * t6) - t3 * t6 * (3 * t5 * t5 - t6 * t6));

    // 2. Normalize Features (Using params from weights_data.h)
    for(int i=0; i<7; i++) {
        features[i] = (features[i] - features_mean[i]) / features_std[i];
    }

    // 3. RUN NEURAL NETWORK
    // Layer 1: Input(7) -> Hidden1(100)
    dense_layer(features, 7, layer1_out, 100, w1, b1, 1); // 1 = use ReLU

    // Layer 2: Hidden1(100) -> Hidden2(100)
    dense_layer(layer1_out, 100, layer2_out, 100, w2, b2, 1); // 1 = use ReLU

    // Layer 3: Hidden2(100) -> Output(10)
    dense_layer(layer2_out, 100, final_out, 10, w3, b3, 0); // 0 = No ReLU

    // Softmax
    softmax(final_out, 10);

    // 4. Find Best Prediction
    int best_digit = 0;
    float max_prob = 0.0f;

    printf("Probabilities: ");
    for(int i=0; i<10; i++) {
        if(final_out[i] > max_prob) {
            max_prob = final_out[i];
            best_digit = i;
        }
    }
    printf("\r\n");

    printf(">>> PREDICTION: [ %d ] (Confidence: %.2f)\r\n", best_digit, max_prob);
    printf("------------------------------\r\n");
}

/**
  * @brief  The application entry point.
  */
int main(void)
{
  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();

  /* Initialize Image Structure */
  LIB_IMAGE_InitStruct(&img_input, (uint8_t*)pImage, IMG_W, IMG_H, IMAGE_FORMAT_GRAYSCALE);

  printf("STM32 Neural Network Ready (0-9)...\r\n");

  /* Infinite loop */
  while (1)
  {
    if (LIB_SERIAL_IMG_Receive(&img_input) == SERIAL_OK)
    {
        // 1. Run Logic
        Run_Neural_Network(img_input.pData);

        // 2. Echo back
        LIB_SERIAL_IMG_Transmit(&img_input);
    }
  }
}

/**
  * @brief System Clock Configuration
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators */
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
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  */
static void MX_USART2_UART_Init(void)
{
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 2000000;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief GPIO Initialization Function
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);
}

/**
  * @brief  This function is executed in case of error occurrence.
  */
void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}
