/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body (Updated for HW3 Q1 & Q2)
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "lib_image.h"
#include "lib_serialimage.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
// ==========================================
//  SELECT YOUR HOMEWORK QUESTION HERE
// ==========================================
// Uncomment ONLY ONE of the following lines:

//#define SOLUTION_Q1     // <-- ACTIVE: Grayscale Otsu (128x128)
#define SOLUTION_Q2   // <-- COMMENTED: Color Otsu (128x128 RGB)

// ==========================================
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
uint32_t histogram[256];

// Input Buffer: Allocated for max size (Color = 128*128*3 = 49152 bytes)
// This works for Grayscale too (it just uses the first 16384 bytes)
volatile uint8_t pImage[128 * 128 * 3];

// Output Buffer: Result is always Binary/Grayscale (128*128*1 = 16384 bytes)
uint8_t pImageOutput[128 * 128];

IMAGE_HandleTypeDef img_input;
IMAGE_HandleTypeDef img_output;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

// --- SHARED HELPER FUNCTIONS ---

// Otsu Logic: Calculates "Between Class Variance" to find optimal threshold
uint8_t CalculateOtsuThreshold(uint32_t *pHist, int totalPixels)
{
    float sum = 0;
    for (int i = 0; i < 256; ++i) sum += i * pHist[i];

    float sumB = 0;
    int wB = 0;
    int wF = 0;
    float varMax = 0;
    uint8_t threshold = 0;

    for (int t = 0; t < 256; t++)
    {
        wB += pHist[t];
        if (wB == 0) continue;
        wF = totalPixels - wB;
        if (wF == 0) break;

        sumB += (float)(t * pHist[t]);
        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;
        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);

        if (varBetween > varMax)
        {
            varMax = varBetween;
            threshold = t;
        }
    }
    return threshold;
}

// --- Q1: GRAYSCALE FUNCTIONS ---
#ifdef SOLUTION_Q1
void CalculateHistogram_Gray(uint8_t *pData, uint32_t size, uint32_t *pHist)
{
    for(int i = 0; i < 256; i++) pHist[i] = 0;
    for(uint32_t i = 0; i < size; i++)
    {
        pHist[pData[i]]++;
    }
}

void ApplyThreshold_Gray(uint8_t *pIn, uint8_t *pOut, uint32_t size, uint8_t threshold)
{
    for(uint32_t i = 0; i < size; i++)
    {
        pOut[i] = (pIn[i] > threshold) ? 255 : 0;
    }
}
#endif

// --- Q2: COLOR FUNCTIONS ---
#ifdef SOLUTION_Q2
void CalculateHistogram_RGB(uint8_t *pRGBData, uint32_t width, uint32_t height, uint32_t *pHist)
{
    for(int i = 0; i < 256; i++) pHist[i] = 0;
    uint32_t totalPixels = width * height;

    for(uint32_t i = 0; i < totalPixels; i++)
    {
        uint32_t idx = i * 3;
        // Average Intensity: (R+G+B)/3
        uint8_t intensity = (pRGBData[idx] + pRGBData[idx+1] + pRGBData[idx+2]) / 3;
        pHist[intensity]++;
    }
}

void ApplyThreshold_RGB(uint8_t *pInRGB, uint8_t *pOutMono, uint32_t width, uint32_t height, uint8_t threshold)
{
    uint32_t totalPixels = width * height;
    for(uint32_t i = 0; i < totalPixels; i++)
    {
        uint32_t idx = i * 3;
        uint8_t intensity = (pInRGB[idx] + pInRGB[idx+1] + pInRGB[idx+2]) / 3;
        pOutMono[i] = (intensity > threshold) ? 255 : 0;
    }
}
#endif
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();

  /* USER CODE BEGIN 2 */

  #ifdef SOLUTION_Q1
  // Q1 Setup: 128x128 Grayscale Input
  // We use pImage for input, pImageOutput for result
  LIB_IMAGE_InitStruct(&img_input, (uint8_t*)pImage, 128, 128, IMAGE_FORMAT_GRAYSCALE);
  LIB_IMAGE_InitStruct(&img_output, pImageOutput, 128, 128, IMAGE_FORMAT_GRAYSCALE);
  #endif

  #ifdef SOLUTION_Q2
  // Q2 Setup: 128x128 Color Input
  // HACK: If your lib doesn't support RGB, we trick it by saying width=128, height=128*3 (Total bytes match)
  LIB_IMAGE_InitStruct(&img_input, (uint8_t*)pImage, 128, 128, 3);
  LIB_IMAGE_InitStruct(&img_output, pImageOutput, 128, 128, IMAGE_FORMAT_GRAYSCALE);
  #endif

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  if (LIB_SERIAL_IMG_Receive(&img_input) == SERIAL_OK)
	  {
          uint8_t optimalThreshold = 0;

          #ifdef SOLUTION_Q1
          // --- Q1 EXECUTION ---
          CalculateHistogram_Gray(img_input.pData, 128*128, histogram);
          optimalThreshold = CalculateOtsuThreshold(histogram, 128*128);
          ApplyThreshold_Gray(img_input.pData, img_output.pData, 128*128, optimalThreshold);
          #endif

          #ifdef SOLUTION_Q2
          // --- Q2 EXECUTION ---
          // Note: img_input.pData contains RGB bytes here
          CalculateHistogram_RGB(img_input.pData, 128, 128, histogram);
          optimalThreshold = CalculateOtsuThreshold(histogram, 128*128);
          ApplyThreshold_RGB(img_input.pData, img_output.pData, 128, 128, optimalThreshold);
          #endif

          // Transmit the Binary Result back to PC
          LIB_SERIAL_IMG_Transmit(&img_output);
	  }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
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

  /** Initializes the CPU, AHB and APB buses clocks
  */
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
  * @param None
  * @retval None
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
  * @param None
  * @retval None
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

/* USER CODE BEGIN 4 */
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  * where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
