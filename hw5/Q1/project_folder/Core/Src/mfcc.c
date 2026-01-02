/* mfcc.c */
#include "mfcc.h"

// Helper function to create the DCT matrix
float* create_dct_matrix(const int numOfDctOutputs, const int numOfMelFilters) {
    float *dct_matrix = malloc(sizeof(float) * numOfDctOutputs * numOfMelFilters);
    float norm_mels = sqrtf(2.0f / numOfMelFilters); // sqrtf for float
    for (int mel_idx = 0; mel_idx < numOfMelFilters; mel_idx++) {
        for (int dct_idx = 0; dct_idx < numOfDctOutputs; dct_idx++) {
            float s = (mel_idx + 0.5f) / numOfMelFilters;
            dct_matrix[dct_idx * numOfMelFilters + mel_idx] = (cosf(dct_idx * PI * s) * norm_mels);
        }
    }
    return dct_matrix;
}

float frequencyToMelSpace(float freq) {
    return 1127.0f * logf(1.0f + freq / 700.0f);
}

float melSpaceToFrequency(float mel) {
    return 700.0f * (expf(mel / 1127.0f) - 1.0f);
}

int create_mel_fbank(int FFTSize, int n_mels, uint32_t **filtPos, uint32_t **filtLen, float **packedFilters) {
    int half_fft_size = FFTSize / 2;
    // VLA (Variable Length Arrays) are risky in embedded, but used in book example.
    // Ideally, malloc these, but sticking to book logic for now.
    float spectrogram_mel[half_fft_size];
    float fmin_mel = frequencyToMelSpace(MEL_LOW_FREQ);
    float fmax_mel = frequencyToMelSpace(MEL_HIGH_FREQ);
    float freq_step = (float)SAMP_FREQ / FFTSize;

    for (int freq_idx = 1; freq_idx < half_fft_size + 1; freq_idx++) {
        float linear_freq = freq_idx * freq_step;
        spectrogram_mel[freq_idx - 1] = frequencyToMelSpace(linear_freq);
    }

    float mel_step = (fmax_mel - fmin_mel) / (n_mels + 1);
    int totalLen = 0;
    
    // Allocate temporary arrays for filter creation
    *filtPos = malloc(n_mels * sizeof(uint32_t));
    *filtLen = malloc(n_mels * sizeof(uint32_t));
    *packedFilters = NULL; // Will use realloc

    for (int mel_idx = 0; mel_idx < n_mels; mel_idx++) {
        float mel = fmin_mel + mel_step * (mel_idx + 1); // Fixed book logic slightly for start/end
        bool startFound = false;
        int startPos = 0, endPos = -1; // Initialize endPos
        
        // Find filter points
        for (int freq_idx = 0; freq_idx < half_fft_size; freq_idx++) {
            // Logic matching book exactly:
            float upper = (spectrogram_mel[freq_idx] - (mel - mel_step)) / mel_step; // Left slope
            float lower = ((mel + mel_step) - spectrogram_mel[freq_idx]) / mel_step; // Right slope
            float filter_val = fmaxf(0.0f, fminf(upper, lower)); // Triangle shape

            if (!startFound && filter_val > 0.0f) {
                startFound = true;
                startPos = freq_idx + 1; // 1-based index for FFT bins usually
            } else if (startFound && filter_val == 0.0f) {
                endPos = freq_idx;
                break;
            }
            // If we reach the end and still valid
             if (startFound) endPos = freq_idx;
        }

        int curLen = endPos - startPos + 1;
        (*filtLen)[mel_idx] = curLen;
        (*filtPos)[mel_idx] = startPos;

        // Realloc packed filters
        float* new_packed = realloc(*packedFilters, (totalLen + curLen) * sizeof(float));
        if (new_packed == NULL) {
             return 1; // Fail
        }
        *packedFilters = new_packed;

        // Fill packed filters
        // Recalculate filter vals to store them
         for (int i = 0; i < curLen; i++) {
             int freq_idx = startPos + i - 1;
             float upper = (spectrogram_mel[freq_idx] - (mel - mel_step)) / mel_step;
             float lower = ((mel + mel_step) - spectrogram_mel[freq_idx]) / mel_step;
             float filter_val = fmaxf(0.0f, fminf(upper, lower));
             (*packedFilters)[totalLen + i] = filter_val;
         }

        totalLen += curLen;
    }
    return 0;
}

void init_mfcc_instance(mfcc_instance *S, uint32_t fftLen, uint32_t nbMelFilters, uint32_t nbDctOutputs) {
    // Create window function
    float *window_func = malloc(sizeof(float) * fftLen);
    for (int i = 0; i < fftLen; i++)
        window_func[i] = 0.5f - 0.5f * cosf(M_2PI * ((float)i) / (fftLen)); // Hamming/Hanning style

    // Create mel filterbank
    S->nbMelFilters = nbMelFilters;
    S->nbDctOutputs = nbDctOutputs;
    S->windowCoefs = window_func;
    S->fftLen = fftLen;

    create_mel_fbank(fftLen, nbMelFilters, &S->filterPos, &S->filterLengths, (float**)&S->windowCoefs); 
    // Note: The book reuses windowCoefs variable in a confusing way in listing 5.15. 
    // In strict C, we should use a separate variable for packed filters, but assuming 
    // packedFilters are stored or referenced. 
    // *Correction*: In the book code provided, they pass &S->windowCoefs to create_mel_fbank's packedFilters.
    // This overwrites the Hamming window we just made! This is likely a typo in the book source.
    // However, to strictly follow the "book implementation", I will leave it, 
    // BUT the standard MFCC flow needs the Hamming window applied BEFORE FFT, 
    // and Mel filters applied AFTER FFT. 
    // I will fix it by using a separate member if possible, but let's stick to the book logic:
    // Actually, looking closely at 5.14, `windowCoefs` seems to hold the Mel Filter Packed weights in the struct?
    // Let's assume S->windowCoefs holds the Hamming window.
    
    // Let's fix the logic for safety:
    float* packedFilters_ptr = NULL;
    create_mel_fbank(fftLen, nbMelFilters, &S->filterPos, &S->filterLengths, &packedFilters_ptr);
    // We need to store this packedFilters_ptr somewhere. 
    // The struct in 5.14 has `float * windowCoefs`. 
    // It seems the book implementation might rely on a global or mishandling here. 
    // For now, let's assume we need to add a member to the struct or reuse one.
    // I will add a `packedFilters` member to the struct in my mind, but since I gave you 5.14 already...
    // Let's just create a static global for now to avoid changing the header too much if you want strict book adherence.
    // actually, let's look at `mfcc_compute`.
    // It uses `S->windowCoefs` in the loop `mel_energy += ... S.windowCoefs[...]`. 
    // This confirms `windowCoefs` in the struct IS the MEL FILTER BANK, not the hamming window.
    // So where is the hamming window? 
    // `mfcc_compute`: `frame[i] *= S->windowCoefs[i];` -> Wait, it uses it for BOTH? 
    // That is a bug in the book code. It uses `windowCoefs` for time-domain windowing AND Mel filtering.
    // I will fix this for you: I will add `melFilters` to the struct in the header I gave you? 
    // No, I'll update `mfcc.c` to use `window_func` for the Hamming window and `S->windowCoefs` for the Mel filters.
    
    // Correct logic for init:
    // 1. Calculate Hamming window (keep local or store separately)
    // 2. Calculate Mel Filters (store in S->windowCoefs as per book usage in second loop)
    
    // To make this work without crashing:
    // We will store Mel Filters in `S->windowCoefs` as the book does for the second loop.
    // We will hardcode the Hamming window inside `mfcc_compute` or re-generate it, 
    // OR we modify the struct. 
    // *Recommendation*: Let's stick to the book's `create_mel_fbank` outputting to `S->windowCoefs`.
    S->windowCoefs = packedFilters_ptr; 

    // Create DCT matrix
    S->dct_matrix = create_dct_matrix(nbDctOutputs, nbMelFilters);

    // Initialize FFT
    arm_rfft_fast_init_f32(&S->rfft, fftLen);
}

void mfcc_compute(mfcc_instance *S, const float *audio_data, int frame_len, float *mfcc_out)
{
    int32_t i, j, bin;
    float frame[frame_len]; 
    float buffer[frame_len];
    float mel_energies[S->nbMelFilters];

    // 1. Apply Window (Hamming)
    // CHANGE IS HERE: Input 'audio_data' is already normalized float.
    // We only need to multiply by the window function. Do NOT divide by 32768.
    for (i = 0; i < frame_len; i++) {
        float win_coef = 0.54f - 0.46f * cosf(2 * M_PI * i / (frame_len - 1));
        frame[i] = audio_data[i] * win_coef; 
    }

    // 2. Compute FFT
    // The arm_rfft_fast_f32 computes Real-to-Complex FFT
    arm_rfft_fast_f32(&S->rfft, frame, buffer, 0);

    // 3. Convert to Power Spectrum
    // CMSIS RFFT Output format: {R0, R_Nyquist, R1, I1, R2, I2, ...}
    float first_energy = buffer[0] * buffer[0];
    float last_energy = buffer[1] * buffer[1];
    
    int32_t half_dim = frame_len / 2;
    
    for (i = 1; i < half_dim; i++) {
        float real = buffer[2 * i];
        float im = buffer[2 * i + 1];
        buffer[i] = real * real + im * im;
    }
    buffer[0] = first_energy;
    buffer[half_dim] = last_energy;

    // 4. Apply Mel Filterbanks
    static int current_filter_offset = 0; // Static to remember position across calls is risky if called multiple times in loop, but okay for single-shot.
                                          // BETTER: Reset it every time function enters.
    current_filter_offset = 0;            // RESET OFFSET HERE

    for (bin = 0; bin < S->nbMelFilters; bin++) {
        float mel_energy = 0;
        int32_t first_index = S->filterPos[bin];
        int32_t length = S->filterLengths[bin];
        
        for (i = 0; i < length; i++) {
             float power_val = buffer[first_index + i];
             float filter_val = S->windowCoefs[current_filter_offset + i];
             mel_energy += power_val * filter_val;
        }
        current_filter_offset += length;

        mel_energies[bin] = mel_energy;
        if (mel_energy <= 0.0f) mel_energies[bin] = 1e-7f;
    }

    // 5. Take Log
    for (bin = 0; bin < S->nbMelFilters; bin++)
        mel_energies[bin] = logf(mel_energies[bin]);

    // 6. Take DCT
    float dct_normalization = 0.316227766f;
    for (i = 0; i < S->nbDctOutputs; i++) {
        float sum = 0.0f;
        for (j = 0; j < S->nbMelFilters; j++) {
            sum += S->dct_matrix[i * S->nbMelFilters + j] * mel_energies[j];
        }
        mfcc_out[i] = sum * dct_normalization;
    }
}
