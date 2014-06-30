#include "NeuralNetwork.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void cuDisplay(cudaSurfaceObject_t cudaSurfaceObject);


float learningWeight;

__device__ float dot(float* v0, float* v1, int length)
{
	float total = 0;

	for(int i = 0; i < length; i++)
	{
		total += v0[i] * v1[i];
	}
	return total;
}

__device__  float activationFunction(float x)
{
	//return 1.0f/ (1.0f + __expf(-x));

	return __frcp_rn(1.0f + __expf(-x));
}

__device__  float activationFunctionDerivative(float activationFunctionOutput)
{
	return activationFunctionOutput * (1.0f - activationFunctionOutput);
}

/*
output * (1 - output) * (difference between actual and target output)
or
output * (1 - output) * (Summation of connection weight * error in next layer)
*/
__device__  float errorGradient(float derivative, float difference)
{
	return derivative * (difference);
}

/*
Given a neuron A connected to a neuron B

	weight = weight on connection between A and B
	error = error of output at neuron B
	output = output of neuron A
*/
__device__ void updateWeight(float* weight, float error, float output, float learningRate)
{
	*weight  = *weight + learningRate*error*output; 
}

__device__ __host__ int getArrayOffset(int* layerSize, int layer)
{
	if(layer == 0)
	{
		return 0;
	}
	else
	{
		int sum = 0;
		for(int i = 0; i  < layer; i++)
		{
			sum += layerSize[i];
		}
		return sum;
	}
}

__device__ __host__ int getWeightOffset(int* layerSize, int layer)
{
	if(layer == 0)
	{
		return 0;
	}
	else
	{
		int sum = 0;
		for(int i = 0; i  < layer; i++)
		{
			sum += layerSize[i] * layerSize[i+1];
		}
		return sum;
	}
}


__device__ void serialNNOutput(float* neuronOutputs, float* neuronWeights, int* layerSize, int numLayers)
{
	//Compute the outputs for the first layer
	//Assumes inputs have already been placed appropriately within the first layer of neuronOutputs
	for(int n =0; n < layerSize[0] - 1; n++)  
	{
		neuronOutputs[n] = activationFunction(neuronOutputs[n]);
	}

	for(int i = 0; i < numLayers -1; i++)
	{
		int offset = getArrayOffset(layerSize, i);

		int weightOffset = getWeightOffset(layerSize, i);

		int biasOffset = 1;
		if(i == numLayers -2)
		{
			biasOffset--;
		}

		for(int n = 0; n < layerSize[i+1] - biasOffset; n++)
		{
			neuronOutputs[offset + layerSize[i] + n] = activationFunction( dot(&neuronOutputs[offset],&neuronWeights[weightOffset + n*(layerSize[i])], layerSize[i]) );
		}
	}
}

__device__ void serialNNBackprop(float* neuronOutputs, float* neuronWeights, float* neuronError, int* layerSize, int numLayers, float* targetOutput, float learningRate)
{
	int offset = getArrayOffset(layerSize, numLayers-1);
	int weightOffset = getWeightOffset(layerSize, numLayers-2);
	int lowerOffset = getArrayOffset(layerSize, numLayers-2);
	
	for(int n = 0; n < layerSize[numLayers-1]; n++)
	{
		neuronError[offset + n] = activationFunctionDerivative(neuronOutputs[offset + n]) * (targetOutput[n] - neuronOutputs[offset + n]);
	}

	//update weights
	for(int x = 0; x < layerSize[numLayers-1]; x++)
	{
		for(int y = 0; y < layerSize[numLayers-2]; y++)
		{
			updateWeight(&neuronWeights[weightOffset + y* layerSize[numLayers -2] + x], neuronError[offset + x], neuronOutputs[lowerOffset + y], learningRate);
		}
	}


	for(int i = numLayers -1; i >1; i--)   /// > 1???? or 0
	{
		offset = getArrayOffset(layerSize, i);
		lowerOffset = getArrayOffset(layerSize, i-1);
		weightOffset = getWeightOffset(layerSize, i-1);

		//find error
		for(int n =0; n < layerSize[i-1]; n++)
		{
			neuronError[lowerOffset + n] = dot(&neuronError[offset], &neuronWeights[weightOffset + n*(layerSize[i-1])], layerSize[i]) * activationFunctionDerivative(neuronOutputs[lowerOffset + n]);
		}

		//update weigths
		for(int x = 0; x < layerSize[i]; x++)
		{
			for(int y = 0; y < layerSize[i-1]; y++)
			{
				updateWeight(&neuronWeights[weightOffset + y* layerSize[i -1] + x], neuronError[offset + x], neuronOutputs[lowerOffset + y], learningRate);
			}
		}
	}
}

__global__ void cuDisplay(cudaSurfaceObject_t cudaSurfaceObject, int width, int height)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x < width && y < height)
	{

		//uchar4 out = make_uchar4(100, 200, 100, 255);

		uchar4 out = make_uchar4(70, 70, 70, 255);

		if(x > 100 && x < 500 && y > 100 && y < 500)
		{
			out = make_uchar4(x, y, 100, 255);
		}

		if(x > 700 && x < 1100 && y > 100 && y < 500)
		{
			out = make_uchar4((char)(activationFunction((x-900)/100.0f)*255), 100, 100, 255);
		}

		surf2Dwrite(out, cudaSurfaceObject, x * sizeof(uchar4), y, cudaBoundaryModeClamp);
	}
}


//extremely bad race condition filled rng
__global__ void initialize(float* neurons, int length ,bool random, unsigned int* seed)
{
	int gtid = blockDim.x * blockIdx.x + threadIdx.x;
		if(gtid < length)
		{
		if(random)
		{
			unsigned int out;
			out = 8253729 * (*seed+ gtid) + 2396403;

			neurons[gtid] = ((float)(out%32767))/32767.0f -0.5f;
			*seed = out;
		}
		else
		{
			neurons[gtid] = 1.0f;
		}
	}
}

void createNN(float* devNeuronWeights, float* devNeuronOutputs, float* devNeuronErrors, int* layerSize, int numLayers)
{
	int arrayLength = getArrayOffset(layerSize,numLayers);
	int weightArrayLength = getWeightOffset(layerSize,numLayers);

	cudaMalloc((void**)&devNeuronOutputs, sizeof(float)* arrayLength);
	cudaMalloc((void**)&devNeuronErrors, sizeof(float)* arrayLength);
	cudaMalloc((void**)&devNeuronWeights, sizeof(float)* weightArrayLength);

	dim3 blockSize = dim3(256);
	dim3 gridSizeA = dim3((arrayLength + blockSize.x -1)/blockSize.x);
	dim3 gridSizeW = dim3((weightArrayLength + blockSize.x -1)/blockSize.x);

	unsigned int* seed;

	*seed = 5865;

	unsigned int* devseed;
	cudaMalloc((void**)&devseed, sizeof(unsigned int));
	cudaMemcpy(devseed, seed, sizeof(unsigned int), cudaMemcpyHostToDevice);


	initialize<<<gridSizeA, blockSize>>>(devNeuronOutputs, arrayLength, true, devseed);
	initialize<<<gridSizeA, blockSize>>>(devNeuronErrors, arrayLength, true, devseed);
	initialize<<<gridSizeW, blockSize>>>(devNeuronWeights, weightArrayLength, true, devseed);
}


void display(cudaSurfaceObject_t cudaSurfaceObject, int width, int height)
{
	dim3 blockSize = dim3(16,16);
	int xGridSize = (width + blockSize.x-1)/blockSize.x; 
	int yGridSize = (height + blockSize.y-1)/blockSize.y;

	dim3 gridSize = dim3(xGridSize, yGridSize);

	cuDisplay<<<gridSize, blockSize>>>(cudaSurfaceObject, width, height);
}

//void trainNN();


for(size_t i = hidden.size() - 1; i > 1; --i) 
{
    thisErr = calculateHiddenErrors(prevErr, hidden.at(i), learningRate);
            
    for(size_t j = 0; j < hidden.at(i - 1).size(); ++j) 
	{
        std::vector<float> deltas;
                
        for(size_t k = 0; k < thisErr.size(); ++k)
		{
             float outputTimesErr = hidden.at(i - 1).getNeuron(j).getOutput(activationFunction) * outErr.at( k );
             deltas.push_back(outputTimesErr);
        }
                
        correctWeights(hidden.at(i - 1).getNeuron( j ), deltas);
     }
            
      prevErr = thisErr;
}