import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack as fft
import scipy.signal as sig
import os

def frequencyToMel(freq):
    return 1127.0 * np.log(1.0 + freq / 700.0)

def melToFrequency(mel):
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)

def create_mfcc_features(recordings_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window):
    features = []
    labels = []

    # 1. Precompute Mel Filterbank Matrix
    half_fft_size = FFTSize // 2
    mel_filters = np.zeros((numOfMelFilters, half_fft_size + 1))
    
    mel_min = frequencyToMel(20)
    mel_max = frequencyToMel(4000)
    mel_points = np.linspace(mel_min, mel_max, numOfMelFilters + 2)
    hz_points = melToFrequency(mel_points)
    bin_points = np.floor((FFTSize + 1) * hz_points / sample_rate).astype(int)

    for i in range(1, numOfMelFilters + 1):
        left = bin_points[i-1]
        center = bin_points[i]
        right = bin_points[i+1]

        for j in range(left, center):
            mel_filters[i-1, j] = (j - left) / (center - left)
        for j in range(center, right):
            mel_filters[i-1, j] = (right - j) / (right - center)

    # 2. Precompute DCT Matrix
    dct_matrix = np.zeros((numOfDctOutputs, numOfMelFilters))
    norm_factor = np.sqrt(2.0 / numOfMelFilters)
    for i in range(numOfDctOutputs):
        for j in range(numOfMelFilters):
            dct_matrix[i, j] = norm_factor * np.cos(np.pi * i * (j + 0.5) / numOfMelFilters)

    # 3. Process All Audio Files
    for item in recordings_list:
        # --- FIX FOR TUPLE ERROR STARTS HERE ---
        # The book's script passes a tuple: (folder_name, file_name)
        # We must join them to make a valid path string.
        if isinstance(item, tuple):
            full_path = os.path.join(item[0], item[1])
            filename = item[1]
        else:
            full_path = item
            filename = os.path.basename(item)
        # --- FIX ENDS HERE ---

        try:
            rate, audio = wav.read(full_path)
        except Exception as e:
            print(f"Could not read {full_path}: {e}")
            continue
        
        # Ensure we just take the center 1024 samples
        if len(audio) > FFTSize:
            center_idx = len(audio) // 2
            audio_frame = audio[center_idx - FFTSize//2 : center_idx + FFTSize//2]
        else:
            audio_frame = np.pad(audio, (0, FFTSize - len(audio)))
        
        # A. Normalize
        audio_frame = audio_frame.astype(np.float32) / 32768.0
        
        # B. Windowing
        audio_frame = audio_frame * window
        
        # C. FFT -> Power Spectrum
        fft_out = np.abs(fft.rfft(audio_frame, n=FFTSize))[:half_fft_size + 1]
        power_spectrum = fft_out ** 2
        
        # D. Mel Filtering
        mel_energies = np.dot(mel_filters, power_spectrum)
        mel_energies = np.where(mel_energies == 0, 1e-7, mel_energies)
        
        # E. Log
        log_mel_energies = np.log(mel_energies)
        
        # F. DCT
        mfcc_out = np.dot(dct_matrix, log_mel_energies)
        
        # G. CALCULATE DELTAS (To match input shape 26)
        deltas = np.zeros_like(mfcc_out) 
        combined_features = np.concatenate((mfcc_out, deltas))

        features.append(combined_features)
        
        # Extract Label from filename (e.g., "0_jackson_0.wav" -> Label=0)
        try:
            label = int(filename.split('_')[0])
            labels.append(label)
        except:
            pass

    return np.array(features), np.array(labels)