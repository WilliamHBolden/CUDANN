#ifndef NEURALNETWORK_CUH
#define NEURALNETWORK_CUH

//void display(cudaSurfaceObject_t cudaSurfaceObject, int width, int height);

//void display(cudaSurfaceObject_t cudaSurfaceObject, int width, int height, float* neuronOutputs, float* neuronError, float* neuronWeights, int* layerSize, int numLayers);
void display(cudaSurfaceObject_t cudaSurfaceObject, int width, int height, int* layerSize, int numLayers);

//void createNN(float* devNeuronWeights, float* devNeuronOutputs, float* devNeuronErrors, int* layerSize, int numLayers);

void createNN(int* layerSize, int numLayers);

#endif