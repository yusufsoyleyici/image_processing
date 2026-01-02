# EE4065_STM32_nucleo_F446RE-Image-Transfer-STM-to-PC-and-PC-to-STM

## STM32-F446RE: Two-Way Image Transfer over UART

### 1. Project Overview

This project demonstrates a high-speed, bi-directional image transfer system between a host PC and an STM32 Nucleo-F446RE microcontroller.

The primary goal is to address the challenge of handling image buffers on a microcontroller with limited internal memory. The STM32F446RE possesses 128KB of internal SRAM and no external SDRAM. This project is configured to transfer a **128x128 pixel** image in **RGB565** (16-bit) format, which requires a **32KB** buffer (`128*128*2 bytes`). This buffer size is well within the 128KB SRAM limitation, demonstrating a feasible solution for memory-constrained devices.

The system uses a custom serial protocol over UART for handshaking and data transfer, with corresponding libraries on both the STM32 (C) and PC (Python) sides.

## 2. Hardware Requirements

* **STM32 Nucleo-F446RE** Development Board
* **Micro-USB Cable** (Provides power, programming, and UART communication)

## 3. Software & Dependencies

### Firmware (STM32 Side)

* **STM32CubeIDE:** Project is built using the STM32 toolchain.
* **Custom Libraries:** `lib_image` for data structure management and `lib_serialimage` for the serial transfer protocol.
* **Hardware Configuration:**
    * `USART2` is enabled for communication (linked to the ST-Link VCP).
    * Baud Rate is set to **`2000000`** (2 Mbps) for high-speed transfer.
    * `FMC` (SDRAM Controller) is disabled.

### Host (PC Side)

* **Python 3.x**
* **Required Python Libraries:**
    * `pyserial`
    * `opencv-python` (cv2)
    * `numpy`

## 4. How to Use

### Part 1: Firmware (STM32)

1.  **Open Project:** Open the project in STM32CubeIDE. The `.ioc` file contains the necessary `USART2` (at 2,000,000 baud) configuration.
2.  **Review `main.c`:** The core logic in the main loop first waits to receive an image buffer, and upon successful reception, transmits the same buffer back to the host. The 32KB image buffer (`pImage`) is declared globally in SRAM.
3.  **Build and Flash:** Compile the project (Hammer icon) and flash it to the Nucleo-F446RE board (Play icon).

### Part 2: Host (PC)

1.  **Prepare Directory:** Ensure the `py_image.py` (main script), `py_serialimg.py` (protocol library), and a test image (e.g., `mandrill.tif`) are all located in the same directory.
2.  **Configure Script:** Open `py_image.py` in a text editor.
3.  **Set COM Port:** Find the `COM_PORT` variable at the top of the script. Update its value (e.g., `"COM8"`) to match the Virtual COM Port assigned to your Nucleo board (check Windows Device Manager).

### Part 3: Execution Flow

1.  **Run PC Script:** Execute the `py_image.py` script from your terminal (`python py_image.py`). The script will initialize the COM port and enter a listening state, waiting for a signal from the STM32.
2.  **Reset Microcontroller:** Press the **BLACK RESET button** on the Nucleo-F446RE board.
3.  **Observe Transfer:** The STM32 will restart and send a request for an image.
    * The PC script will detect this request, automatically resize the `mandrill.tif` file to 128x128, and transmit it to the STM32.
    * The STM32 will receive the 32KB image into its SRAM.
    * The STM32 will then immediately transmit the 32KB buffer from its SRAM back to the PC.
    * The PC script will receive the buffer, display the image on-screen for 2 seconds, and save it locally as `received_from_f446re.png`.
    * The script will then return to the listening state.

## 5. Troubleshooting

* **Script fails to open COM port:** Ensure the `COM_PORT` variable in `py_image.py` is correct and that no other application (like a serial monitor) is using the port.
* **Script fails with `imread` or `resize` error:** Ensure the test image (e.g., `mandrill.tif`) is present in the same directory as the Python script.
* **Script waits at "STM32'den istek bekleniyor..." indefinitely:**
    1.  Ensure you pressed the **RESET button** on the Nucleo board *after* starting the PC script.
    2.  Verify the **Baud Rate** in STM32CubeIDE (`.ioc` file, `USART2`) is exactly **`2000000`** to match the rate set in the `py_serialimg.py` library.
