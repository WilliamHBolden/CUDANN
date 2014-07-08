#include "NeuralNetwork.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h> 

__global__ void cuDisplay(cudaSurfaceObject_t cudaSurfaceObject);


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

__device__  float FSCactivationFunctionDerivative(float activationFunctionOutput)
{
	return activationFunctionOutput * (1.0f - activationFunctionOutput) + 0.1f;
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


	/*
	for(int n =0; n < layerSize[0] - 1; n++)  
	{
		neuronOutputs[n] = activationFunction(neuronOutputs[n]);
	}

	*/

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

			neuronOutputs[offset + layerSize[i] + n] = activationFunction(dot(&neuronWeights[weightOffset + n * layerSize[i]], &neuronOutputs[offset], layerSize[i]));
		}
	}
}

/*
	x dim minimum = layerSize[currentLayer+1]
	y dim minumum = layerSize[currentLayer]
*/
__global__ void atomicParallelNNOutput(float* neuronOutputs, float* neuronWeights, int* layerSize, int numLayers, int currentLayer)
{

	//@TODO: shared memory, first segment for neuron outputs in next layer, second segment for neuron outputs in current layer

	int n = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	int biasOffset = 1;
	if(currentLayer == numLayers -2)
	{
		biasOffset--;
	}

	if(n < (layerSize[currentLayer+1]-biasOffset))
	{
		if(i < layerSize[currentLayer])
		{
			int offset = getArrayOffset(layerSize, currentLayer);
			int weightOffset = getWeightOffset(layerSize, currentLayer, numLayers);

			atomicAdd(&neuronOutputs[offset + layerSize[currentLayer] + n], neuronWeights[weightOffset + n * layerSize[currentLayer] + i] * neuronOutputs[offset + i]);
		}
	}
}

/*
	x dim minimum: layerSize[currentLayer+1]
*/
__global__ void applyActivationFunction(float* neuronOutputs, int* layerSize, int numLayers, int currentLayer)
{
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	int biasOffset = 1;
	if(currentLayer == numLayers -2)
	{
		biasOffset--;
	}

	if(n < (layerSize[currentLayer+1]-biasOffset))
	{
		int offset = getArrayOffset(layerSize, currentLayer);
		neuronOutputs[offset + layerSize[currentLayer] + n] = activationFunction(neuronOutputs[offset + layerSize[currentLayer] + n]);
	}
}

__global__ void parallelNNOutput(float* neuronOutputs, float* neuronWeights, int* layerSize, int numLayers, int currentLayer)
{
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	int biasOffset = 1;
	if(currentLayer == numLayers -2)
	{
		biasOffset--;
	}

	if(n < (layerSize[currentLayer+1]-biasOffset))
	{
		int offset = getArrayOffset(layerSize, currentLayer);
		int weightOffset = getWeightOffset(layerSize, currentLayer, numLayers);

		neuronOutputs[offset + layerSize[currentLayer] + n] = activationFunction( dot(&neuronWeights[weightOffset + n * layerSize[currentLayer]], &neuronOutputs[offset], layerSize[currentLayer]) );
	}
}

/*
	xdim: min of layerSize[numLayers - 1]
*/
__global__ void calculateOutputLayerError(float* neuronOutputs, float* neuronError, float* targetOutput, int* layerSize, int numLayers)
{
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	if(n < layerSize[numLayers-1])
	{
		int offset = getArrayOffset(layerSize, numLayers-1);
		neuronError[offset + n] = FSCactivationFunctionDerivative(neuronOutputs[offset + n]) * (targetOutput[n] - neuronOutputs[offset + n]);
	}
}

/*
	xdim: min of layerSize[currentLayer - 1]
	ydim: min of layerSize[currentLayer]
*/
__global__ void calculateError(float* neuronOutputs, float* neuronError, float* neuronWeights, int* layerSize, int numLayers, int currentLayer)
{
	//@TODO shared memory for neuronOutputs and neuronError

	int n = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	int offset = getArrayOffset(layerSize, currentLayer);
	int lowerOffset = getArrayOffset(layerSize, currentLayer-1);
	int	weightOffset = getWeightOffset(layerSize, currentLayer-1, numLayers);

	if(n < layerSize[currentLayer -1] && i < layerSize[currentLayer])
	{

		if(true)
		{
			if(currentLayer > 0)
			{
				atomicAdd(&neuronError[lowerOffset + n], activationFunctionDerivative(neuronOutputs[lowerOffset + n])* (neuronError[offset+i] * neuronWeights[weightOffset + n*(layerSize[currentLayer])+i]));
			}
			else
			{
				atomicAdd(&neuronError[lowerOffset + n], activationFunctionDerivative(activationFunction(neuronOutputs[lowerOffset + n]))* (neuronError[offset+i] * neuronWeights[weightOffset + n*(layerSize[currentLayer])+i]));
			}
		}
		else
		{
			//The same only 1 neuronError and n layer sizes are used in these calculations. This is terribly incorrect but the neural network still manages to learn if this code is used!
			// Conclusion: Neural Networks are magic
			if(currentLayer > 0)
			{
				atomicAdd(&neuronError[lowerOffset + n], activationFunctionDerivative(neuronOutputs[lowerOffset + n])* (neuronError[offset] * neuronWeights[weightOffset + n*(layerSize[currentLayer])]));
			}
			else
			{
				atomicAdd(&neuronError[lowerOffset + n], activationFunctionDerivative(activationFunction(neuronOutputs[lowerOffset + n]))* (neuronError[offset] * neuronWeights[weightOffset + n*(layerSize[currentLayer])]));
			}
		}


	}
}

/*
	ydim: minimum of layerSize[currentLayer - 1]
	xdim: minimum of layerSize[currentLayer]
*/
__global__ void updateWeights(float* neuronOutputs, float* neuronWeights, float* neuronError, int* layerSize, int numLayers, int currentLayer, float learningRate)
{
	int y = blockDim.x * blockIdx.x + threadIdx.x;
	int x = blockDim.y * blockIdx.y + threadIdx.y;

	int offset = getArrayOffset(layerSize, currentLayer);
	int lowerOffset = getArrayOffset(layerSize, currentLayer-1);
	int	weightOffset = getWeightOffset(layerSize, currentLayer-1, numLayers);

	if(x < layerSize[currentLayer] && y < layerSize[currentLayer-1])
	{
		updateWeight(&neuronWeights[weightOffset + x* layerSize[currentLayer -1] + y], neuronError[offset + x], neuronOutputs[lowerOffset + y], learningRate);
	}
}

__global__ void cuDisplay(cudaSurfaceObject_t cudaSurfaceObject, int width, int height, float* neuronOutputs, float* neuronError, float* neuronWeights, int* layerSize, int numLayers, int count)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	uchar4 out = make_uchar4(70, 70, 70, 255);
	
	if(x < width && y < height)
	{
		
		if(x > 100 && x < 500 && y > 100 && y < 500)
		{
			out = make_uchar4(x, y, 100, 255);
		}

		if(x > 700 && x < 1100 && y > 100 && y < 500)
		{
			
			float fx = (float)(x-900);

			float fy = (float)(y-300);

			fx /= 25.0f;
			fy /= 25.0f;



			char ocolx = (char)(200.0f*(((__sinf(fx*fy)+1)/2)/1.25f +0.1f));

		//	int tx = (x-300);
		//	int ty = (x-300);

		//	no[0]= (float)((tx*ty)%314)/25.0f;

		//	char ocolx = (char)(200.0f*(((__sinf((float)((tx*ty)%314)/(25.0f*25.0f))+1)/2)/1.25f +0.1f));

		//	char ocoly = (char)(fy*200.0f);

		//	((__sinf(x * y)+1)/2)/1.25f +0.1f;


			out = make_uchar4(ocolx, ocolx, ocolx, 255);
			
			//out = make_uchar4((char)(activationFunction((x-900)/100.0f)*255), 100, 100, 255);
		}
		


		if(x > 100 && x < 500 && y > 100 && y < 500)
		//if (x > 600 && x < 620 && y >200 && y < 220)
		{

			float fx = (float)(x-300);

			float fy = (float)(y-300);




			fx /= 25.0f;
			fy /= 25.0f;

			int arrayLength = getArrayOffset(layerSize,numLayers);
			float* no = (float*)malloc(arrayLength * sizeof(float));
			for(int i =0; i < arrayLength; i++)
			{
				no[i] = neuronOutputs[i];
			}



			no[0]= fx;
			no[1]= fy;
			
			serialNNOutput(no, neuronWeights, layerSize, numLayers);

			int index = getArrayOffset(layerSize, numLayers-1);

			char ocol = (char)(no[index]*200.0f);

			out = make_uchar4(ocol, ocol, ocol, 255);


			if((fx == 0 || fx == 1.0f) && (fy == 0 || fy == 1.0f))
			{
			//	out = make_uchar4(200, 10, 10, 255);
			}

			free(no);
		}

		surf2Dwrite(out, cudaSurfaceObject, x * sizeof(uchar4), y, cudaBoundaryModeClamp);

	}
}

/*
	Sets the output of bias neurons to 1 and the output of other neurons to 0
*/
__global__ void initOutputs(float* neurons, int length , int* layerSize, int numLayers)
{
	int gtid = blockDim.x * blockIdx.x + threadIdx.x;
	if(gtid < length)
	{
		neurons[gtid] = 0.0f;

		for(int i = 0; i < numLayers - 1; i++)
		{
			if(gtid == getArrayOffset(layerSize, i) + layerSize[i]-1)
			{
				neurons[gtid] = 1.0f;
			}
		}
	}
}


float* devNeuronWeights;
float* devNeuronOutputs;
float* devNeuronErrors;
float* devDefaultOutputs;
int* devLayerSize;

void createNN(int* layerSize, int numLayers)
{
	int arrayLength = getArrayOffset(layerSize,numLayers);
	int weightArrayLength = getWeightOffset(layerSize,numLayers-1, numLayers);

	cudaMalloc((void**)&devNeuronOutputs, sizeof(float)* arrayLength);
	cudaMalloc((void**)&devDefaultOutputs, sizeof(float)* arrayLength);
	cudaMalloc((void**)&devNeuronErrors, sizeof(float)* arrayLength);
	cudaMalloc((void**)&devNeuronWeights, sizeof(float)* weightArrayLength);
	cudaMalloc((void**)&devLayerSize, sizeof(int)*numLayers);

	cudaMemcpy(devLayerSize, layerSize, sizeof(int)*numLayers, cudaMemcpyHostToDevice);

	dim3 blockSize = dim3(256);
	dim3 gridSize = dim3((arrayLength + blockSize.x -1)/blockSize.x);
	initOutputs<<<gridSize, blockSize>>>(devNeuronOutputs, arrayLength, devLayerSize, numLayers);
	cudaMemcpy(devDefaultOutputs, devNeuronOutputs, sizeof(float)*arrayLength, cudaMemcpyDeviceToDevice);

	srand(time(NULL));

	float* hostNeuronWeights = (float*)malloc(weightArrayLength*sizeof(float));
	for(int i =0; i < weightArrayLength; i++)
	{
		hostNeuronWeights[i] = (rand()%10000-5000)/10000.0f;  //random weights between -0.5 and 0.5
	}
	cudaMemcpy(devNeuronWeights, hostNeuronWeights, sizeof(float)*weightArrayLength, cudaMemcpyHostToDevice);
	free(hostNeuronWeights);
}


float* devTrainingInputs;
float* devTrainingOutputs;
int devInputSize;
int devNumItems;
int devNumOutputs;

/*
	inputs : a flat array of input data
	inputSize : number of floats in the array making up one input
	numItems : total number of inputs
	numOutputs : number of outputs per input
*/
void setTrainingData(float* inputs, float* outputs,int inputSize, int numItems, int numOutputs)
{
	cudaMalloc((void**)&devTrainingInputs, sizeof(float)*inputSize*numItems);
	cudaMalloc((void**)&devTrainingOutputs, sizeof(float)*numOutputs*numItems);

	cudaMemcpy(devTrainingInputs, inputs, sizeof(float)*inputSize*numItems, cudaMemcpyHostToDevice);
	cudaMemcpy(devTrainingOutputs, outputs, sizeof(float)*numOutputs*numItems, cudaMemcpyHostToDevice);

	devInputSize = inputSize;
	devNumItems = numItems;
	devNumOutputs = numOutputs;
}

int getGridSize(unsigned int minimumSize, unsigned int blockSize)
{
	return (minimumSize + blockSize - 1)/blockSize;
}

/*
	layerSize : host array of layer sizes
	numLayers : number of layers
	iterations : the number of iterations through the input set
	learningRate : how fast the neural net learns, set between 0.3 and 0.8
*/
void orderedTrain(int* hostLayerSize, int numLayers, int iterations, float learningRate)
{
	int outputArrayLength = getArrayOffset(hostLayerSize,numLayers);

	for(int setItertion =0; setItertion < iterations; setItertion++)
	{
		dim3 blockSize;
		dim3 gridSize;

		for(int n = 0; n < devNumItems; n++)
		{
			cudaMemcpy(devNeuronOutputs, devDefaultOutputs, sizeof(float)*outputArrayLength, cudaMemcpyDeviceToDevice); //set devNeuronOutputs to defaults
			cudaMemcpy(devNeuronOutputs, devTrainingInputs + n*devInputSize, sizeof(float)*devInputSize, cudaMemcpyDeviceToDevice); //Set the inputs

			for(int i = 0; i < numLayers -1; i++)
			{
				blockSize = dim3(16, 16);
				gridSize = dim3(getGridSize(hostLayerSize[i+1], blockSize.x), getGridSize(hostLayerSize[i], blockSize.y));

				atomicParallelNNOutput<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronWeights, devLayerSize, numLayers, i);
				applyActivationFunction<<<getGridSize(hostLayerSize[i+1], 128), 128>>>(devNeuronOutputs, devLayerSize, numLayers, i);
			}

			cudaMemset(devNeuronErrors, 0, sizeof(float)*outputArrayLength);

			blockSize = dim3(128);
			gridSize = dim3(getGridSize(hostLayerSize[numLayers-1], blockSize.x));

			calculateOutputLayerError<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronErrors, &devTrainingOutputs[n*devNumOutputs], devLayerSize, numLayers);
	
			for(int i = numLayers - 1;i > 0; i--)
			{
				blockSize = dim3(16, 16);
				gridSize = dim3(getGridSize(hostLayerSize[i-1], blockSize.x), getGridSize(hostLayerSize[i], blockSize.y));

				calculateError<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronErrors, devNeuronWeights, devLayerSize, numLayers, i);
				updateWeights<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronWeights, devNeuronErrors, devLayerSize, numLayers, i, learningRate);
			}
		}
	}
}



/*
	layerSize : host array of layer sizes
	numLayers : number of layers
	iterations : the number of iterations through random items in the input set
	learningRate : how fast the neural net learns, set between 0.3 and 0.8
*/
void randomTrain(int* hostLayerSize, int numLayers, int iterations, float learningRate)
{
	int outputArrayLength = getArrayOffset(hostLayerSize,numLayers);

	for(int setItertion =0; setItertion < iterations; setItertion++)
	{
		dim3 blockSize;
		dim3 gridSize;

		//RAND_MAX
		int n = (rand() * (RAND_MAX+1) + rand())% devNumItems;

		cudaMemcpy(devNeuronOutputs, devDefaultOutputs, sizeof(float)*outputArrayLength, cudaMemcpyDeviceToDevice); //set devNeuronOutputs to defaults
		cudaMemcpy(devNeuronOutputs, devTrainingInputs + n*devInputSize, sizeof(float)*devInputSize, cudaMemcpyDeviceToDevice); //Set the inputs

		for(int i = 0; i < numLayers -1; i++)
		{
			blockSize = dim3(16, 16);
			gridSize = dim3(getGridSize(hostLayerSize[i+1], blockSize.x), getGridSize(hostLayerSize[i], blockSize.y));

			atomicParallelNNOutput<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronWeights, devLayerSize, numLayers, i);
			applyActivationFunction<<<getGridSize(hostLayerSize[i+1], 128), 128>>>(devNeuronOutputs, devLayerSize, numLayers, i);
		}

		cudaMemset(devNeuronErrors, 0, sizeof(float)*outputArrayLength);

		blockSize = dim3(128);
		gridSize = dim3(getGridSize(hostLayerSize[numLayers-1], blockSize.x));

		calculateOutputLayerError<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronErrors, &devTrainingOutputs[n*devNumOutputs], devLayerSize, numLayers);
	
		for(int i = numLayers - 1;i > 0; i--)
		{
			blockSize = dim3(16, 16);
			gridSize = dim3(getGridSize(hostLayerSize[i-1], blockSize.x), getGridSize(hostLayerSize[i], blockSize.y));

			calculateError<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronErrors, devNeuronWeights, devLayerSize, numLayers, i);
			updateWeights<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronWeights, devNeuronErrors, devLayerSize, numLayers, i, learningRate);
		}
		
	}
}


int count = 0;

void display(cudaSurfaceObject_t cudaSurfaceObject, int width, int height, int* hostLayerSize, int numLayers)
{
	dim3 blockSize = dim3(16,16);
	int xGridSize = (width + blockSize.x-1)/blockSize.x; 
	int yGridSize = (height + blockSize.y-1)/blockSize.y;

	dim3 gridSize = dim3(xGridSize, yGridSize);

    cudaThreadSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);

	count++;

	printf("====%d====\n", count);

	if(count % 10 == 0)
	{
		cuDisplay<<<gridSize, blockSize>>>(cudaSurfaceObject, width, height, devNeuronOutputs,  devNeuronErrors,  devNeuronWeights,  devLayerSize, numLayers, count);
	}
}

