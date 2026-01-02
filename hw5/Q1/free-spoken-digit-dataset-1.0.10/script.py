import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack as fft
import scipy.signal as sig
import tensorflow as tf
import os

# --- CONFIGURATION ---
WAV_FILE = "recordings/5_jackson_0.wav" # Ensure this path is correct!
MODEL_FILE = "mlp_fsdd_model.h5"
OUTPUT_HEADER = "golden_data.h"
# ---------------------

def compute_mfcc_single(filename):
    # This logic matches your training script EXACTLY
    sample_rate, audio = wav.read(filename)
    FFTSize = 1024
    numOfMelFilters = 20
    numOfDctOutputs = 13
    
    # 1. Normalize & Slice Center
    audio = audio.astype(np.float32) / 32768.0
    if len(audio) > FFTSize:
        center_idx = len(audio) // 2
        audio_frame = audio[center_idx - FFTSize//2 : center_idx + FFTSize//2]
    else:
        audio_frame = np.pad(audio, (0, FFTSize - len(audio)))
        
    # 2. Window
    window = sig.get_window("hamming", FFTSize)
    audio_frame = audio_frame * window
    
    # 3. FFT -> Power
    half_fft = FFTSize // 2
    fft_out = np.abs(fft.rfft(audio_frame, n=FFTSize))[:half_fft + 1]
    power_spectrum = fft_out ** 2
    
    # 4. Mel Filterbank
    mel_filters = np.zeros((numOfMelFilters, half_fft + 1))
    mel_min = 1127.0 * np.log(1.0 + 20.0 / 700.0)
    mel_max = 1127.0 * np.log(1.0 + 4000.0 / 700.0)
    mel_points = np.linspace(mel_min, mel_max, numOfMelFilters + 2)
    hz_points = 700.0 * (np.exp(mel_points / 1127.0) - 1.0)
    bin_points = np.floor((FFTSize + 1) * hz_points / sample_rate).astype(int)
    
    for i in range(1, numOfMelFilters + 1):
        left, center, right = bin_points[i-1], bin_points[i], bin_points[i+1]
        for j in range(left, center):
            mel_filters[i-1, j] = (j - left) / (center - left)
        for j in range(center, right):
            mel_filters[i-1, j] = (right - j) / (right - center)
            
    mel_energies = np.dot(mel_filters, power_spectrum)
    mel_energies = np.where(mel_energies == 0, 1e-7, mel_energies)
    
    # 5. Log & DCT
    log_mel = np.log(mel_energies)
    dct_matrix = np.zeros((numOfDctOutputs, numOfMelFilters))
    norm_factor = np.sqrt(2.0 / numOfMelFilters)
    for i in range(numOfDctOutputs):
        for j in range(numOfMelFilters):
            dct_matrix[i, j] = norm_factor * np.cos(np.pi * i * (j + 0.5) / numOfMelFilters)
            
    mfcc = np.dot(dct_matrix, log_mel)
    
    # 6. Deltas (Zeros)
    deltas = np.zeros_like(mfcc)
    full_input = np.concatenate((mfcc, deltas))
    return full_input

# Run Calculation
input_features = compute_mfcc_single(WAV_FILE)

# Run Inference
model = tf.keras.models.load_model(MODEL_FILE)
# Reshape for Keras (1, 26)
preds = model.predict(input_features.reshape(1, -1))
predicted_class = np.argmax(preds)

print(f"Expected Prediction: {predicted_class}")
print(f"First 3 MFCCs: {input_features[0]:.4f}, {input_features[1]:.4f}, {input_features[2]:.4f}")

# Write to Header
with open(OUTPUT_HEADER, "w") as f:
    f.write("#ifndef GOLDEN_DATA_H\n#define GOLDEN_DATA_H\n\n")
    f.write(f"// Generated from {WAV_FILE}\n")
    f.write(f"// Python Prediction: {predicted_class}\n\n")
    f.write("float golden_input[26] = {\n")
    for i, val in enumerate(input_features):
        f.write(f"    {val:.6f}f,")
        if (i+1)%5==0: f.write("\n")
    f.write("\n};\n\n#endif")

print(f"Saved {OUTPUT_HEADER}")