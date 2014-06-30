#ifndef NEURALNETWORK_CUH
#define NEURALNETWORK_CUH

void display(cudaSurfaceObject_t cudaSurfaceObject, int width, int height);
void createNN(float* devNeuronWeights, float* devNeuronOutputs, float* devNeuronErrors, int* layerSize, int numLayers);

#endif