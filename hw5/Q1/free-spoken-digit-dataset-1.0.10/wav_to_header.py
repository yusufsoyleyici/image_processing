import numpy as np
import scipy.io.wavfile as wav
import os

# --- CONFIGURATION ---
INPUT_WAV = "recordings/0_jackson_0.wav"  # Change to your specific test file
OUTPUT_HEADER = "test_audio.h"
FFT_SIZE = 1024
# ---------------------

# 1. Read the audio
sample_rate, audio_data = wav.read(INPUT_WAV)

# 2. Normalize to float [-1.0, 1.0] (Matching your mfcc_func.py logic)
# Note: wav.read usually returns int16. We divide by 32768.0 as per your script.
audio_float = audio_data.astype(np.float32) / 32768.0

# 3. Slice the CENTER 1024 samples (Matching your mfcc_func.py logic)
if len(audio_float) > FFT_SIZE:
    center_idx = len(audio_float) // 2
    start = center_idx - (FFT_SIZE // 2)
    end = center_idx + (FFT_SIZE // 2)
    audio_slice = audio_float[start:end]
    print(f"Slicing audio from sample {start} to {end}")
else:
    # Pad if too short (rare for these files but good safety)
    pad_amount = FFT_SIZE - len(audio_float)
    audio_slice = np.pad(audio_float, (0, pad_amount))
    print("Padding audio to fit 1024 samples")

# 4. Generate the C Header File
with open(OUTPUT_HEADER, 'w') as f:
    f.write("#ifndef TEST_AUDIO_H\n")
    f.write("#define TEST_AUDIO_H\n\n")
    f.write(f"// Audio Source: {INPUT_WAV}\n")
    f.write(f"// Sliced: Center {FFT_SIZE} samples\n\n")
    f.write(f"float test_audio_data[{FFT_SIZE}] = {{\n")
    
    # Write the float values comma-separated
    for i, sample in enumerate(audio_slice):
        f.write(f"    {sample:.6f}f")
        if i < len(audio_slice) - 1:
            f.write(",")
        if (i + 1) % 10 == 0:  # Newline every 10 samples for readability
            f.write("\n")
            
    f.write("\n};\n\n")
    f.write("#endif // TEST_AUDIO_H\n")

print(f"Successfully created {OUTPUT_HEADER} with centered audio data!")