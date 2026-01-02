/* nn_inference.c */
#include "nn_inference.h"
#include "weights_data.h" 
#include <math.h>

// Define layer sizes matching your Python model
#define INPUT_SIZE 26
#define L1_SIZE 100
#define L2_SIZE 100
#define OUTPUT_SIZE 10

// Buffers to hold intermediate results
static float l1_output[L1_SIZE];
static float l2_output[L2_SIZE];
static float final_output[OUTPUT_SIZE];

// Helper: ReLU Activation Function
static void relu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

// Helper: Softmax Activation Function
static void softmax(float* data, int size) {
    float max_val = -100000.0f;
    float sum = 0.0f;

    // Find max for numerical stability
    for (int i = 0; i < size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    // Compute exp and sum
    for (int i = 0; i < size; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }

    // Normalize
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

// Main Inference Function
int neural_network_inference(float* input_features) {
    
    // --- Layer 1: Dense (26 -> 100) ---
    // Using exported weights: w1 (Output x Input)
    for (int j = 0; j < L1_SIZE; j++) {
        float sum = b1[j]; // Use 'b1', not 'layer1_biases'
        for (int i = 0; i < INPUT_SIZE; i++) {
            // Use 'w1', not 'layer1_weights'
            sum += input_features[i] * w1[j * INPUT_SIZE + i];
        }
        l1_output[j] = sum;
    }
    relu(l1_output, L1_SIZE);

    // --- Layer 2: Dense (100 -> 100) ---
    for (int j = 0; j < L2_SIZE; j++) {
        float sum = b2[j]; // Use 'b2'
        for (int i = 0; i < L1_SIZE; i++) {
            sum += l1_output[i] * w2[j * L1_SIZE + i]; // Use 'w2'
        }
        l2_output[j] = sum;
    }
    relu(l2_output, L2_SIZE);

    // --- Layer 3: Output (100 -> 10) ---
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        float sum = b3[j]; // Use 'b3'
        for (int i = 0; i < L2_SIZE; i++) {
            sum += l2_output[i] * w3[j * L2_SIZE + i]; // Use 'w3'
        }
        final_output[j] = sum;
    }
    
    // Softmax
    softmax(final_output, OUTPUT_SIZE);

    // Argmax: Find class with highest probability
    int predicted_class = 0;
    float max_prob = final_output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (final_output[i] > max_prob) {
            max_prob = final_output[i];
            predicted_class = i;
        }
    }

    return predicted_class;
}