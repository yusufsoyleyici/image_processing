/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body (Updated for HW3 Q3 - Complete)
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
//#define SOLUTION_Q1     // Grayscale Otsu
//#define SOLUTION_Q2     // Color Otsu
#define SOLUTION_Q3     // Morphological Ops (Direct Binary Input)

#ifdef SOLUTION_Q3
    // UNCOMMENT ONE OPERATION TO TEST:
    //#define DO_EROSION
    #define DO_DILATION
    //#define DO_OPENING    // (Erosion -> Dilation)
    //#define DO_CLOSING    // (Dilation -> Erosion)
#endif
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

// Input Buffer: Holds the received Binary Image (used as Temp buffer for Q3)
volatile uint8_t pImage[128 * 128 * 3];

// Output Buffer: Holds the result of operations
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

// --- SHARED HELPER FUNCTIONS (Q1/Q2) ---
// (Keeping these so the code compiles if you switch back to Q1/Q2)
uint8_t CalculateOtsuThreshold(uint32_t *pHist, int totalPixels) {
    float sum = 0; for(int i=0; i<256; ++i) sum += i * pHist[i];
    float sumB = 0; int wB = 0, wF = 0; float varMax = 0; uint8_t threshold = 0;
    for(int t=0; t<256; t++) {
        wB += pHist[t]; if(wB==0) continue;
        wF = totalPixels - wB; if(wF==0) break;
        sumB += (float)(t * pHist[t]);
        float mB = sumB/wB; float mF = (sum-sumB)/wF;
        float varBetween = (float)wB * (float)wF * (mB-mF)*(mB-mF);
        if(varBetween > varMax) { varMax = varBetween; threshold = t; }
    }
    return threshold;
}
void CalculateHistogram_Gray(uint8_t *pData, uint32_t size, uint32_t *pHist) {
    for(int i=0; i<256; i++) pHist[i]=0; for(uint32_t i=0; i<size; i++) pHist[pData[i]]++;
}
void ApplyThreshold_Gray(uint8_t *pIn, uint8_t *pOut, uint32_t size, uint8_t threshold) {
    for(uint32_t i=0; i<size; i++) pOut[i] = (pIn[i] > threshold) ? 255 : 0;
}
// ----------------------------------------

// --- Q3: MORPHOLOGICAL FUNCTIONS ---
#ifdef SOLUTION_Q3

void ApplyErosion(uint8_t *pIn, uint8_t *pOut, int width, int height)
{
    // Clear borders
    for(int i=0; i<width*height; i++) pOut[i] = 0;

    for(int y = 1; y < height - 1; y++)
    {
        for(int x = 1; x < width - 1; x++)
        {
            int allWhite = 1;
            // Check 3x3 Window
            for(int ky = -1; ky <= 1; ky++)
            {
                for(int kx = -1; kx <= 1; kx++)
                {
                    // Use >127 check to be safe against minor noise
                    if(pIn[(y + ky) * width + (x + kx)] < 127)
                    {
                        allWhite = 0;
                        break;
                    }
                }
                if(!allWhite) break;
            }
            pOut[y * width + x] = (allWhite) ? 255 : 0;
        }
    }
}

void ApplyDilation(uint8_t *pIn, uint8_t *pOut, int width, int height)
{
    for(int i=0; i<width*height; i++) pOut[i] = 0;

    for(int y = 1; y < height - 1; y++)
    {
        for(int x = 1; x < width - 1; x++)
        {
            int anyWhite = 0;
            for(int ky = -1; ky <= 1; ky++)
            {
                for(int kx = -1; kx <= 1; kx++)
                {
                    if(pIn[(y + ky) * width + (x + kx)] > 127)
                    {
                        anyWhite = 1;
                        break;
                    }
                }
                if(anyWhite) break;
            }
            pOut[y * width + x] = (anyWhite) ? 255 : 0;
        }
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
  // Initialize standard structs
  LIB_IMAGE_InitStruct(&img_input, (uint8_t*)pImage, 128, 128, IMAGE_FORMAT_GRAYSCALE);
  LIB_IMAGE_InitStruct(&img_output, pImageOutput, 128, 128, IMAGE_FORMAT_GRAYSCALE);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    if (LIB_SERIAL_IMG_Receive(&img_input) == SERIAL_OK)
    {
        #ifdef SOLUTION_Q3
            // img_input.pData is already the Binary Image from PC

            #ifdef DO_EROSION
                // Input -> Erode -> Output
                ApplyErosion(img_input.pData, img_output.pData, 128, 128);
                LIB_SERIAL_IMG_Transmit(&img_output);
            #endif

            #ifdef DO_DILATION
                // Input -> Dilate -> Output
                ApplyDilation(img_input.pData, img_output.pData, 128, 128);
                LIB_SERIAL_IMG_Transmit(&img_output);
            #endif

            #ifdef DO_OPENING
                // Opening = Erosion -> Dilation
                // 1. Erode: Input (pImage) -> Temp (pImageOutput)
                ApplyErosion(img_input.pData, img_output.pData, 128, 128);

                // 2. Dilate: Temp (pImageOutput) -> Result (pImage)
                // We overwrite pImage because we need to transmit specific struct
                ApplyDilation(img_output.pData, img_input.pData, 128, 128);

                // Transmit result (which is now in pImage/img_input)
                LIB_SERIAL_IMG_Transmit(&img_input);
            #endif

            #ifdef DO_CLOSING
                // Closing = Dilation -> Erosion
                // 1. Dilate: Input (pImage) -> Temp (pImageOutput)
                ApplyDilation(img_input.pData, img_output.pData, 128, 128);

                // 2. Erode: Temp (pImageOutput) -> Result (pImage)
                ApplyErosion(img_output.pData, img_input.pData, 128, 128);

                LIB_SERIAL_IMG_Transmit(&img_input);
            #endif

        #else
            // Fallback logic for Q1/Q2 (Otsu)
            CalculateHistogram_Gray(img_input.pData, 128*128, histogram);
            uint8_t th = CalculateOtsuThreshold(histogram, 128*128);
            ApplyThreshold_Gray(img_input.pData, img_output.pData, 128*128, th);
            LIB_SERIAL_IMG_Transmit(&img_output);
        #endif
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
void assert_failed(uint8_t *file, uint32_t line)
{
}
#endif /* USE_FULL_ASSERT */
