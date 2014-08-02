#include "NeuralNetwork.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h> 
__device__ __host__ float dot(float* v0, float* v1, int length)
{
	float total = 0;

	for(int i = 0; i < length; i++)
	{
		total += v0[i] * v1[i];
	}
	return total;
}

__device__ __host__ float activationFunction(float x)
{
	#ifdef __CUDA_ARCH__
		return __frcp_rz(1.0f + __expf(-x));
	#else
		return 1.0f/(1.0f + expf(-x));
	#endif
}

__device__  float activationFunctionDerivative(float activationFunctionOutput)
{
	return activationFunctionOutput * (1.0f - activationFunctionOutput);
}

__device__  float FSCactivationFunctionDerivative(float activationFunctionOutput)
{
	return activationFunctionOutput * (1.0f - activationFunctionOutput) + 0.00f;
}

/*
Given a neuron A connected to a neuron BparallelNNOutput

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

__device__ __host__ void serialNNOutput(float* neuronOutputs, float* neuronWeights, int* layerSize, int numLayers)
{
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
	Parallel backpropagation
*/

/*
	x dim minimum = layerSize[currentLayer+1]
	y dim minumum = layerSize[currentLayer]
	shared: layerSize[currentLayer]
*/
__global__ void parallelNNOutput(float* neuronOutputs, float* neuronWeights, int* layerSize, int numLayers, int currentLayer)
{
	int n = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	int biasOffset = 1;
	if(currentLayer == numLayers -2)
	{
		biasOffset--;
	}

	if(n < (layerSize[currentLayer+1]-biasOffset) && i < layerSize[currentLayer])
	{
		int offset = getArrayOffset(layerSize, currentLayer);
		int weightOffset = getWeightOffset(layerSize, currentLayer, numLayers);

		atomicAdd(&neuronOutputs[offset + layerSize[currentLayer] + n], neuronWeights[weightOffset + n * layerSize[currentLayer] + i] * neuronOutputs[offset + i]);
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
	shared: layerSize[currentLayer] + layerSize[currentLayer - 1]
*/
__global__ void calculateError(float* neuronOutputs, float* neuronError, float* neuronWeights, int* layerSize, int numLayers, int currentLayer)
{
	int n = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	if(n < layerSize[currentLayer -1] && i < layerSize[currentLayer])
	{

		int offset = getArrayOffset(layerSize, currentLayer);
		int lowerOffset = getArrayOffset(layerSize, currentLayer-1);
		int	weightOffset = getWeightOffset(layerSize, currentLayer-1, numLayers);
		

		if(true)
		{
			if(currentLayer > 0)
			{
				atomicAdd(&neuronError[lowerOffset + n], FSCactivationFunctionDerivative(neuronOutputs[lowerOffset + n])* (neuronError[offset+i] * neuronWeights[weightOffset + n*(layerSize[currentLayer])+i]));
			}
			else
			{
				atomicAdd(&neuronError[lowerOffset + n], FSCactivationFunctionDerivative(activationFunction(neuronOutputs[lowerOffset + n]))* (neuronError[offset+i] * neuronWeights[weightOffset + n*(layerSize[currentLayer])+i]));
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
__global__ void updateWeights(float* neuronOutputs, float* neuronWeights, float* neuronError, float* previousDeltas, int* layerSize, int numLayers, int currentLayer, float learningRate)
{
	int y = blockDim.x * blockIdx.x + threadIdx.x;
	int x = blockDim.y * blockIdx.y + threadIdx.y;

	int offset = getArrayOffset(layerSize, currentLayer);
	int lowerOffset = getArrayOffset(layerSize, currentLayer-1);
	int	weightOffset = getWeightOffset(layerSize, currentLayer-1, numLayers);

	if(x < layerSize[currentLayer] && y < layerSize[currentLayer-1])
	{
		int weightIndex = weightOffset + x* layerSize[currentLayer -1] + y;


		float weightDelta = neuronError[offset + x] * neuronOutputs[lowerOffset + y] * learningRate  + previousDeltas[weightIndex] * 0.3; 
		neuronWeights[weightIndex] += weightDelta;

//		float weightDelta = neuronError[offset + x] * neuronOutputs[lowerOffset + y] * learningRate; 
//		neuronWeights[weightIndex] += weightDelta + previousDeltas[weightIndex] * 0.3;
		

		previousDeltas[weightIndex] = weightDelta;
	}
}






__global__ void cuDisplay(cudaSurfaceObject_t cudaSurfaceObject, int width, int height, float* neuronOutputs, float* neuronWeights, int* layerSize, int numLayers, float* inputs)
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

			int index = (y-100) * 1200 + (x-700)*3;

			char ocolx = (char)(inputs[index]*255);
			char ocoly = (char)(inputs[index + 1]*255);
			char ocolz = (char)(inputs[index + 2]*255);


			out = make_uchar4(ocolx, ocoly, ocolz, 255);

		}
		


		if(x > 100 && x < 500 && y > 100 && y < 500)
		//if (x > 600 && x < 620 && y >200 && y < 220)
		{

			float fx = (float)(x-300);

			float fy = (float)(y-300);




			fx /= 200.0f;
			fy /= 200.0f;

			int arrayLength = getArrayOffset(layerSize,numLayers);
			float* no = (float*)malloc(arrayLength * sizeof(float));
			for(int i =0; i < arrayLength; i++)
			{
				no[i] = neuronOutputs[i];
			}



			no[0]= fx/1.0f;
			no[1]= fy/1.0f;

			for(int i =2; i < 16; i+=2)
			{
				no[i] = __sinf(fx * (i/2) *2* 3.14159);
				no[i + 1] = __sinf(fy * (i/2)*2 * 3.14159);
			}
			
			serialNNOutput(no, neuronWeights, layerSize, numLayers);

			int index = getArrayOffset(layerSize, numLayers-1);
			

			char ocol = (char)(no[index]*255);
			char ocoly = (char)(no[index+1]*255);
			char coolz = (char)(no[index+2]*255);

			out = make_uchar4(ocol, ocoly, coolz, 255);

			

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
float* devWeightDeltas;
int* devLayerSize;

void createNN(int* layerSize, int numLayers)
{
	int arrayLength = getArrayOffset(layerSize,numLayers);
	int weightArrayLength = getWeightOffset(layerSize,numLayers-1, numLayers);

	cudaMalloc((void**)&devNeuronOutputs, sizeof(float)* arrayLength);
	cudaMalloc((void**)&devDefaultOutputs, sizeof(float)* arrayLength);
	cudaMalloc((void**)&devNeuronErrors, sizeof(float)* arrayLength);
	cudaMalloc((void**)&devNeuronWeights, sizeof(float)* weightArrayLength);
	cudaMalloc((void**)&devWeightDeltas, sizeof(float)* weightArrayLength);
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

	cudaMemset(devWeightDeltas, 0, sizeof(float)*weightArrayLength);
}


float* devTrainingInputs;
float* devTrainingOutputs;
int devSetSize;
int devNumItems;
int devNumOutputs;

/*
	inputs : an array of input data
	inputs : an array of ouput data
	setSize : number of items in the input array making up one set of inputs
	numItems : total number of input sets
	numOutputs : number of outputs per input
*/
void setTrainingData(float* inputs, float* outputs,int setSize, int numItems, int numOutputs)
{
	cudaMalloc((void**)&devTrainingInputs, sizeof(float)*setSize*numItems);
	cudaMalloc((void**)&devTrainingOutputs, sizeof(float)*numOutputs*numItems);

	cudaMemcpy(devTrainingInputs, inputs, sizeof(float)*setSize*numItems, cudaMemcpyHostToDevice);
	cudaMemcpy(devTrainingOutputs, outputs, sizeof(float)*numOutputs*numItems, cudaMemcpyHostToDevice);

	devSetSize = setSize;
	devNumItems = numItems;
	devNumOutputs = numOutputs;
}

int getGridSize(unsigned int minimumSize, unsigned int blockSize)
{
	return (minimumSize + blockSize - 1)/blockSize;
}

void forwardPropagation()
{

}

void backPropagation()
{

	
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
			cudaMemcpy(devNeuronOutputs, devTrainingInputs + n*devSetSize, sizeof(float)*devSetSize, cudaMemcpyDeviceToDevice); //Set the inputs

			for(int i = 0; i < numLayers -1; i++)
			{
				blockSize = dim3(32, 32);
				gridSize = dim3(getGridSize(hostLayerSize[i+1], blockSize.x), getGridSize(hostLayerSize[i], blockSize.y));

				parallelNNOutput<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronWeights, devLayerSize, numLayers, i);
				applyActivationFunction<<<getGridSize(hostLayerSize[i+1], 128), 128>>>(devNeuronOutputs, devLayerSize, numLayers, i);
			}

			cudaMemset(devNeuronErrors, 0, sizeof(float)*outputArrayLength);

			blockSize = dim3(128);
			gridSize = dim3(getGridSize(hostLayerSize[numLayers-1], blockSize.x));

			calculateOutputLayerError<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronErrors, &devTrainingOutputs[n*devNumOutputs], devLayerSize, numLayers);
	
			for(int i = numLayers - 1;i > 0; i--)
			{
				blockSize = dim3(32, 32);
				gridSize = dim3(getGridSize(hostLayerSize[i-1], blockSize.x), getGridSize(hostLayerSize[i], blockSize.y));

				calculateError<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronErrors, devNeuronWeights, devLayerSize, numLayers, i);
				updateWeights<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronWeights, devNeuronErrors, devWeightDeltas, devLayerSize, numLayers, i, learningRate);
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
		unsigned int n = (unsigned int)((rand() * (RAND_MAX+1) + rand()))% devNumItems;

		//n = 30;


		cudaMemcpy(devNeuronOutputs, devDefaultOutputs, sizeof(float)*outputArrayLength, cudaMemcpyDeviceToDevice); //set devNeuronOutputs to defaults
		cudaMemcpy(devNeuronOutputs, devTrainingInputs + n*devSetSize, sizeof(float)*devSetSize, cudaMemcpyDeviceToDevice); //Set the inputs

		for(int i = 0; i < numLayers -1; i++)
		{
			blockSize = dim3(32, 32);
			gridSize = dim3(getGridSize(hostLayerSize[i+1], blockSize.x), getGridSize(hostLayerSize[i], blockSize.y));


			parallelNNOutput<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronWeights, devLayerSize, numLayers, i);
			applyActivationFunction<<<getGridSize(hostLayerSize[i+1], 128), 128>>>(devNeuronOutputs, devLayerSize, numLayers, i);
		}

		cudaMemset(devNeuronErrors, 0, sizeof(float)*outputArrayLength);

		blockSize = dim3(128);
		gridSize = dim3(getGridSize(hostLayerSize[numLayers-1], blockSize.x));

		calculateOutputLayerError<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronErrors, &devTrainingOutputs[n*devNumOutputs], devLayerSize, numLayers);

		if(rand() % 1000 == 0)
		{
			printf("Index: %d\n", n);

			printf("Size: %d\n", devNumItems);

			float* expected = (float*)malloc(sizeof(float) * 10);
			float* actual = (float*)malloc(sizeof(float) * 10);

			cudaMemcpy(expected, &devTrainingOutputs[n*devNumOutputs], sizeof(float)*10, cudaMemcpyDeviceToHost);
			cudaMemcpy(actual, &devNeuronOutputs[getArrayOffset(hostLayerSize,numLayers-1)], sizeof(float)*10, cudaMemcpyDeviceToHost);

			for(int i =0; i < 10; i++)
			{
				printf("%d|	e(%f)	:	a(%f)\n", i, expected[i], actual[i]);
			}

			printf("\n");

			free(expected);
			free(actual);

		}

	//	updateWeights<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronWeights, devNeuronErrors, devWeightDeltas, devLayerSize, numLayers, numLayers, learningRate);
	
		for(int i = numLayers - 1;i > 0; i--)
		{
			blockSize = dim3(32, 32);
			gridSize = dim3(getGridSize(hostLayerSize[i-1], blockSize.x), getGridSize(hostLayerSize[i], blockSize.y));

			calculateError<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronErrors, devNeuronWeights, devLayerSize, numLayers, i);
			updateWeights<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronWeights, devNeuronErrors, devWeightDeltas, devLayerSize, numLayers, i, learningRate);
		}
		
	}
}

float* getWeightArray(int* hostLayerSize, int numLayers)
{
	int weightArrayLength = getWeightOffset(hostLayerSize, numLayers-1, numLayers);

	float* weights = (float*) malloc(sizeof(float) * weightArrayLength);
	cudaMemcpy(weights, devNeuronWeights, sizeof(float) * weightArrayLength, cudaMemcpyDeviceToHost);

	return weights;
}

void display(cudaSurfaceObject_t cudaSurfaceObject, int width, int height, int* hostLayerSize, int numLayers)
{
	dim3 blockSize = dim3(16,16);
	int xGridSize = (width + blockSize.x-1)/blockSize.x; 
	int yGridSize = (height + blockSize.y-1)/blockSize.y;

	dim3 gridSize = dim3(xGridSize, yGridSize);

    cudaThreadSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);

	cuDisplay<<<gridSize, blockSize>>>(cudaSurfaceObject, width, height, devNeuronOutputs, devNeuronWeights,  devLayerSize, numLayers, devTrainingOutputs);
}

void displayClassification()
{

}

