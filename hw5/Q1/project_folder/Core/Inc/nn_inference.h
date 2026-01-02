/* nn_inference.h */
#ifndef INC_NN_INFERENCE_H_
#define INC_NN_INFERENCE_H_

// Function Prototype
// Takes an array of 26 floats (MFCC features) and returns the predicted digit (0-9)
int neural_network_inference(float* input_features);

#endif /* INC_NN_INFERENCE_H_ */