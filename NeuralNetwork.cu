#include "NeuralNetwork.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

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

//valid for layer < totalLayers-1
__device__ __host__ int getWeightOffset(int* layerSize, int layer, int numLayers)
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
			if(i == numLayers -2) //out layer -1
			{
				sum += layerSize[i] * layerSize[i+1];
			}
			else
			{
				sum += layerSize[i] * (layerSize[i+1]-1);
			}
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

		int weightOffset = getWeightOffset(layerSize, i, numLayers);

		int biasOffset = 1;
		if(i == numLayers -2)
		{
			biasOffset--;
		}

		for(int n = 0; n < layerSize[i+1] - biasOffset; n++) //For every item in the layer the the right, excluding the bias
		{
			/*
			Adding layerSize[i] in neuronWeights[] places the offset at the beginning of the layer to the right. n increments through said layer

			The output of the next layer is calculated by passing the dot product of the current weights and outputs through the activation function

			Output(Layer(i+1, neuron n)) = S( Outputs(layer i)   dot   ConnectionWeights(Layer(i+1, neuron n), Layer(i)) )
		
		*/

			neuronOutputs[offset + layerSize[i] + n] = dot(&neuronWeights[weightOffset + n * layerSize[i]], &neuronOutputs[offset], layerSize[i]);
			neuronOutputs[offset + layerSize[i] + n] = activationFunction(neuronOutputs[offset + layerSize[i] + n]);

			neuronOutputs[offset] = neuronOutputs[offset];


			//else
			//neuronOutputs[offset + layerSize[i] + n] = 333;

		}
	}
}

__device__ void serialNNBackprop(float* neuronOutputs, float* neuronWeights, float* neuronError, int* layerSize, int numLayers, float* targetOutput, float learningRate)
{
	int offset = getArrayOffset(layerSize, numLayers-1);
	int weightOffset = getWeightOffset(layerSize, numLayers-2, numLayers);
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
			updateWeight(&neuronWeights[weightOffset + x* layerSize[numLayers -2] + y], neuronError[offset + x], neuronOutputs[lowerOffset + y], learningRate);
		}
	}


	for(int i = numLayers -1; i >0; i--)   /// > 1???? or 0
	{
		offset = getArrayOffset(layerSize, i);
		lowerOffset = getArrayOffset(layerSize, i-1);
		weightOffset = getWeightOffset(layerSize, i-1, numLayers);

		//find error
		for(int n =0; n < layerSize[i-1]; n++)
		{
			neuronError[lowerOffset + n] = dot(&neuronError[offset], &neuronWeights[weightOffset + n*(layerSize[i])], layerSize[i]);

			//Dot of the weight between A and layer K and the error in K

			neuronError[lowerOffset + n]*= activationFunctionDerivative(neuronOutputs[lowerOffset + n]);
		}

		//update weigths
		for(int x = 0; x < layerSize[i]; x++)
		{
			for(int y = 0; y < layerSize[i-1]; y++)
			{
				updateWeight(&neuronWeights[weightOffset + x* layerSize[i -1] + y], neuronError[offset + x], neuronOutputs[lowerOffset + y], learningRate);
			}
		}
	}

	float gerror = 0;
	for(int i =0; i <7; i++)
	{
		gerror += neuronError[i] * neuronError[i];
	}
	printf("Global error: %f\n", gerror);
}

__global__ void cuDisplay(cudaSurfaceObject_t cudaSurfaceObject, int width, int height, float* neuronOutputs, float* neuronError, float* neuronWeights, int* layerSize, int numLayers, int xpixel, int ypixel)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	uchar4 out = make_uchar4(70, 70, 70, 255);
	
	if(x*y == 1)
	{
		
		int learn[8] = {0, 1, 1, 0, 1, 1, 0, 0};
		float out[4] = {0, 0, 1, 0};



		for(int i =0; i < 4; i++)
		{
			neuronOutputs[0] = learn[i*2];
			neuronOutputs[1] = learn[i*2 + 1];

		//	layerSize[0] = 2;
		//	layerSize[1] = 2;
		//	layerSize[2] = 3;

	//		printf("%d %d = %f \n", neuronOutputs[i*2], neuronOutputs[i*2+1], out[i]);

			serialNNOutput(neuronOutputs, neuronWeights, layerSize, numLayers);
			serialNNBackprop(neuronOutputs, neuronWeights, neuronError, layerSize, numLayers, &out[i] , 01.8f);

			int index = getArrayOffset(layerSize, numLayers-1);
			printf("%d %d = %f \n", learn[i*2], learn[i*2+1], neuronOutputs[index]);
		}
		printf("----------------\n");
		

	}

	/*

	if(xpixel ==x && ypixel == y)
	{
		neuronOutputs[0] = x/400.0f;
		neuronOutputs[1] = y/400.0f;

		float out[3];


		out[0] = activationFunction((x-400)/100.0f);
//		out[1] = 100/255.0f;

//		out[0] = x/255.0f;
//		out[1] = y/255.0f;

		serialNNOutput(neuronOutputs, neuronWeights, layerSize, numLayers);
		serialNNBackprop(neuronOutputs, neuronWeights, neuronError, layerSize, numLayers, out , 0.1f);

	//	out[2] = 100/255.0f;

	}

	*/
		/*
		for(int xx = 100; xx < 105; xx++)
		{
			for(int yy = 100; yy < 105; yy++)
			{
				neuronOutputs[0] = xx;
				neuronOutputs[1] = yy;

				float out[3];
				out[0] = xx/255.0f;
				out[1] = yy/255.0f;
				out[2] = 100/255.0f;

				serialNNOutput(neuronOutputs, neuronWeights, layerSize, numLayers);
				serialNNBackprop(neuronOutputs, neuronWeights, neuronError, layerSize, numLayers, out , 0.1f);

			}
		}
		*/
		
	
	
	if(x < width && y < height)
	{

		//uchar4 out = make_uchar4(100, 200, 100, 255);

		

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
__global__ void initialize(float* neurons, int length ,bool random, unsigned int iseed)
{
	int gtid = blockDim.x * blockIdx.x + threadIdx.x;
	if(gtid < length)
	{
		if(random)
		{
			/*
			int seed;
			unsigned int out;
			out = (8253729) * (seed + gtid*seed) + 2396403;
			seed = out;
			neurons[gtid] = ((float)(out%32767))/32767.0f -0.5f;
			*/
			neurons[gtid] = (gtid %20 )/20.0f -0.5f;

		}
		else
		{
			neurons[gtid] = 1.0f;
		}
	}
}


float* devNeuronWeights;
float* devNeuronOutputs;
float* devNeuronErrors;

int* devLayerSize;

void createNN(int* layerSize, int numLayers)
{
	int arrayLength = getArrayOffset(layerSize,numLayers);
	int weightArrayLength = getWeightOffset(layerSize,numLayers-1, numLayers);

	cudaMalloc((void**)&devNeuronOutputs, sizeof(float)* arrayLength);
	cudaMalloc((void**)&devNeuronErrors, sizeof(float)* arrayLength);
	cudaMalloc((void**)&devNeuronWeights, sizeof(float)* weightArrayLength);


	cudaMalloc((void**)&devLayerSize, sizeof(int)*numLayers);
	cudaMemcpy(devLayerSize, layerSize, sizeof(int)*numLayers, cudaMemcpyHostToDevice);

	
	dim3 blockSize = dim3(256);
	dim3 gridSizeA = dim3((arrayLength + blockSize.x -1)/blockSize.x);
	dim3 gridSizeW = dim3((weightArrayLength + blockSize.x -1)/blockSize.x);

	unsigned int seed = 5323;

	initialize<<<gridSizeA, blockSize>>>(devNeuronOutputs, arrayLength, false, seed);

	initialize<<<gridSizeW, blockSize>>>(devNeuronWeights, weightArrayLength, true, seed);
	
}

int xxx = 0;
int yyy = 0;

void display(cudaSurfaceObject_t cudaSurfaceObject, int width, int height, int* layerSize, int numLayers)
{
	dim3 blockSize = dim3(16,16);
	int xGridSize = (width + blockSize.x-1)/blockSize.x; 
	int yGridSize = (height + blockSize.y-1)/blockSize.y;

	dim3 gridSize = dim3(xGridSize, yGridSize);

	if(xxx% 400 ==0)
	{
		yyy++;
	}
	else
	{
		xxx++;
	}
	yyy = yyy%400;

	 cudaThreadSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);

	cuDisplay<<<gridSize, blockSize>>>(cudaSurfaceObject, width, height, devNeuronOutputs,  devNeuronErrors,  devNeuronWeights,  devLayerSize, numLayers, xxx, yyy);
}
