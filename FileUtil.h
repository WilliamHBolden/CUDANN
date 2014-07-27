#ifndef FILEUTIL_H
#define FILEUTIL_H

//#include "NeuralNetwork.cuh"

void loadImage();

void loadMNIST(const char* imageFilepath, const char* labelFilepath, float** inputData, float** outputData, int* numSets, int* setSize, int* outputsPerSet);

//void saveNN(const NNInfo& info);

//NNInfo* loadNN(const char* weightFilepath);

#endif