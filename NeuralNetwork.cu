#include "NeuralNetwork.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>


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

__device__  float FSCactivationFunctionDerivative(float activationFunctionOutput)
{
	return activationFunctionOutput * (1.0f - activationFunctionOutput) + 0.1f;
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

__device__ void serialNNBackprop(float* neuronOutputs, float* neuronWeights, float* neuronError, int* layerSize, int numLayers, float* targetOutput, float learningRate)
{
	int offset = getArrayOffset(layerSize, numLayers-1);
	int weightOffset = getWeightOffset(layerSize, numLayers-2, numLayers);
	int lowerOffset = getArrayOffset(layerSize, numLayers-2);
	
	for(int n = 0; n < layerSize[numLayers-1]; n++) //For every neuron in the final layer
	{
		neuronError[offset + n] = FSCactivationFunctionDerivative(neuronOutputs[offset + n]) * (targetOutput[n] - neuronOutputs[offset + n]);
	}

	//update weights
	/*
	for(int x = 0; x < layerSize[numLayers-1]; x++)
	{
		for(int y = 0; y < layerSize[numLayers-2]; y++)
		{
			updateWeight(&neuronWeights[weightOffset + x* layerSize[numLayers -2] + y], neuronError[offset + x], neuronOutputs[lowerOffset + y], learningRate);
		}
	}
	
	*/
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
			if(i > 0)
			neuronError[lowerOffset + n]*= activationFunctionDerivative(neuronOutputs[lowerOffset + n]);
			else
			neuronError[lowerOffset + n]*= activationFunctionDerivative(activationFunction(neuronOutputs[lowerOffset + n]));
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
	/*
	float gerror = 0;
	for(int i =0; i <7; i++)
	{
		gerror += neuronError[i] * neuronError[i];
	}
	*/
	//printf("Global error: %f\n", gerror);
}


/*
	x dim minimum = # neurons in next layer
	y dim minumum = # neurons in current layer
*/
__global__ void atomicParallelNNOutput(float* neuronOutputs, float* neuronWeights, int* layerSize, int numLayers, int currentLayer)
{
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
	int n = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	int offset = getArrayOffset(layerSize, currentLayer);
	int lowerOffset = getArrayOffset(layerSize, currentLayer-1);
	int	weightOffset = getWeightOffset(layerSize, currentLayer-1, numLayers);

	if(n < layerSize[currentLayer -1] && i < layerSize[currentLayer])
	{
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

/*
	xdim: minimum of layerSize[i]
	ydim: minimum of layerSize[i - 1]
*/
__global__ void updateWeights(float* neuronOutputs, float* neuronWeights, float* neuronError, int* layerSize, int numLayers, int currentLayer, float learningRate)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int offset = getArrayOffset(layerSize, currentLayer);
	int lowerOffset = getArrayOffset(layerSize, currentLayer-1);
	int	weightOffset = getWeightOffset(layerSize, currentLayer-1, numLayers);

	if(x < layerSize[currentLayer] && y < layerSize[currentLayer-1])
	{
		updateWeight(&neuronWeights[weightOffset + x* layerSize[currentLayer -1] + y], neuronError[offset + x], neuronOutputs[lowerOffset + y], learningRate);
	}
}



__global__ void cuTrain(float* neuronOutputs, float* neuronError, float* neuronWeights, int* layerSize, int numLayers, int count, float adx)
{
	float learn[16] = {0, 1, 1, 0, 1, 1, 0, 0, 0.2, 1, -3, 1, -2, 2.5, 0, 2.5};
		float out[8] = {1, 1, 0, 0, 0, 0, 0, 0};

		float actual;

	//	for(float x =-4.140*0+0+0/2.0f; x < 4.1400+0+0/2.0f; x+=0.22)
	//	for(float y =-4.140+0+0/2.0f; y < 4.1400+0+0/2.0f; y+=0.22)
		for(int i =0; i < 4; i++)
		{

		//	neuronOutputs[0] = i;

		//	actual = __sinf(i);
			
		//	actual = ((tanh(i)+1)/2)/1.25f +0.1f;

		//	actual = __cosf(__sinf(i))*__sinf(i)*__sinf(i);

	//		actual = activationFunctionDerivative(activationFunction(i));

		//	actual = ((__sinf(x * y)+1)/2)/1.25f +0.1f;


		//	neuronOutputs[0] = i ;

			//if(count > 2000)
			{
		//		actual = ((__sinf(i)+1)/2)/1.25f +0.1f;
				//actual = ((tanh(i)+1)/2)/1.25f +0.1f;
			}

				//activationFunction(i);

			neuronOutputs[0] = learn[i*2];
			neuronOutputs[1] = learn[i*2 + 1];


		//	neuronOutputs[0] = x;
		//	neuronOutputs[1] = y;

		//	actual = __sinf(x+y);

			serialNNOutput(neuronOutputs, neuronWeights, layerSize, numLayers);
			serialNNBackprop(neuronOutputs, neuronWeights, neuronError, layerSize, numLayers, &out[i], 0.8f);

			int index = getArrayOffset(layerSize, numLayers-1);

		
			printf("%f %f = %f \n", learn[i*2], learn[i*2+1], neuronOutputs[index]);
	//		printf("S(%f) = %f   (%f) \n", i, neuronOutputs[index], actual);


		//	printf("S(%f) = %f   (%f) \n", i, neuronOutputs[index], actual);
		}
		printf("--------%d--------\n", count);
	//	printf()
}

__global__ void cuDisplay(cudaSurfaceObject_t cudaSurfaceObject, int width, int height, float* neuronOutputs, float* neuronError, float* neuronWeights, int* layerSize, int numLayers, int count, float adx,int* devValues, int* funcVals)
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

			fx /= 50.0f;
			fy /= 50.0f;

			char ocolx = (char)(200.0f*(((__sinf(fx * fy)+1)/2)/1.25f +0.1f));
		//	char ocoly = (char)(fy*200.0f);

		//	((__sinf(x * y)+1)/2)/1.25f +0.1f;


			out = make_uchar4(ocolx, ocolx, ocolx, 255);
			
			//out = make_uchar4((char)(activationFunction((x-900)/100.0f)*255), 100, 100, 255);
		}
		

	//	if(count %10 == 0)
		

		


	//	float fx = (float)x;

	//	float fy = (float)y;

		if(x > 100 && x < 500 && y > 100 && y < 500)
		//if (x > 600 && x < 620 && y >200 && y < 220)
		{

			float fx = (float)(x-300);

			float fy = (float)(y-300);

			fx /= 100.0f;
			fy /= 100.0f;


		


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
				out = make_uchar4(200, 10, 10, 255);


			free(no);
		}

		surf2Dwrite(out, cudaSurfaceObject, x * sizeof(uchar4), y, cudaBoundaryModeClamp);


/*

		if(y == 500)
		{
			int arrayLength = getArrayOffset(layerSize,numLayers);


			float* no = (float*)malloc(arrayLength * sizeof(float));

			
			for(int i =0; i < arrayLength; i++)
			{
				no[i] = neuronOutputs[i];
			}
			

			//no[0] = ((fx-width/2)/50.0f);

		//	no[0] = ((float)((x%628)-314))/100.0f;

		//	no[0] = ((fx-width/2)/100.0f);

			

			if(x < width /3)
			{
				//[0][1] ... [1][1]

				no[0]= fx/(width/3);
				no[1]= 1;
			}
			else if(x < 2*width /3)
			{
				//[0][1] ... [0][0]

				no[0]= 0;
				no[1]= ((width/3)-(fx - width/3))/(width/3);
			}
			else
			{
				//[0][0] ... [1][1]

				no[0]= (fx - 2*width/3)/(width/3);
				no[1]= (fx - 2*width/3)/(width/3);
			}


			serialNNOutput(no, neuronWeights, layerSize, numLayers);
			int index = getArrayOffset(layerSize, numLayers-1);

			devValues[x] = 200 + (int)(no[index]*200.0f);


			free(no);
			

		}

		


		if(y == devValues[x])
		{
			out = make_uchar4(255, 00, 0, 255);
			surf2Dwrite(out, cudaSurfaceObject, x * sizeof(uchar4), devValues[x], cudaBoundaryModeClamp);
		}

		*/

	}
}



__global__ void initialize(float* neurons, int length , int* layerSize, int numLayers)
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


int* devValues;
int* funcVals;

int* devLayerSize;



void createNN(int* layerSize, int numLayers)
{

	

	cudaMalloc((void**)&devValues, sizeof(int)* 1500);
	cudaMalloc((void**)&funcVals, sizeof(int)* 1500);



	int arrayLength = getArrayOffset(layerSize,numLayers);
	int weightArrayLength = getWeightOffset(layerSize,numLayers-1, numLayers);



	float* wvals = (float*)malloc(weightArrayLength*sizeof(float));

	for(int i =0; i < weightArrayLength; i++)
	{
		wvals[i] = (rand()%10000-5000)/10000.0f;
	}


	cudaMalloc((void**)&devNeuronOutputs, sizeof(float)* arrayLength);
	cudaMalloc((void**)&devDefaultOutputs, sizeof(float)* arrayLength);


	cudaMalloc((void**)&devNeuronErrors, sizeof(float)* arrayLength);
	cudaMalloc((void**)&devNeuronWeights, sizeof(float)* weightArrayLength);


	cudaMalloc((void**)&devLayerSize, sizeof(int)*numLayers);
	cudaMemcpy(devLayerSize, layerSize, sizeof(int)*numLayers, cudaMemcpyHostToDevice);

	cudaMemcpy(devNeuronWeights, wvals, sizeof(float)*weightArrayLength, cudaMemcpyHostToDevice);

	
	dim3 blockSize = dim3(256);
	dim3 gridSizeA = dim3((arrayLength + blockSize.x -1)/blockSize.x);
	dim3 gridSizeW = dim3((weightArrayLength + blockSize.x -1)/blockSize.x);

	//unsigned int seed = 5323;

	initialize<<<gridSizeA, blockSize>>>(devNeuronOutputs, arrayLength, devLayerSize, numLayers);
	initialize<<<gridSizeA, blockSize>>>(devDefaultOutputs, arrayLength, devLayerSize, numLayers);


//	initialize<<<gridSizeW, blockSize>>>(devNeuronWeights, weightArrayLength, true, seed);
	
}




void train(int* hostLayerSize, int numLayers)
{
	#define ARGSIZE 2



	int arrayLength = getArrayOffset(hostLayerSize,numLayers);


//	cudaMemcpy(devNeuronOutputs, devDefaultOutputs, sizeof(float)*arrayLength, cudaMemcpyDeviceToDevice);

//	float startingArgs[ARGSIZE] = {1, 0};


//	cudaMemcpy(devNeuronOutputs, startingArgs, sizeof(float)*ARGSIZE, cudaMemcpyHostToDevice);

//	cudaMemcpy(startingArgs, devNeuronOutputs , sizeof(float)*ARGSIZE, cudaMemcpyDeviceToHost);

//	printf("%f    -     %f\n\n", startingArgs[0], startingArgs[1]);
	

	float startingArgs[ARGSIZE] = {1, 0};

	//Forward prop through neural net

	//for(int x = 0; x < 10000; x++)
	{

	cudaMemcpy(devNeuronOutputs, devDefaultOutputs, sizeof(float)*arrayLength, cudaMemcpyDeviceToDevice);
	cudaMemcpy(devNeuronOutputs, startingArgs, sizeof(float)*ARGSIZE, cudaMemcpyHostToDevice);

	for(int i = 0; i < numLayers -1; i++)
	{
		dim3 blockSize = dim3(16, 16);

		dim3 gridSize = dim3((hostLayerSize[i+1] + blockSize.x-1) / blockSize.x, (hostLayerSize[i] + blockSize.y-1) / blockSize.y );

		//dim3 tgs = (20, 20);

		//dim3 tgs = dim3((numLayers + tbs.x -1)/tbs.x);

	//	parallelNNOutput<<<1, 128>>>(devNeuronOutputs, devNeuronWeights, devLayerSize, numLayers, i);

		atomicParallelNNOutput<<<gridSize, blockSize>>>(devNeuronOutputs, devNeuronWeights, devLayerSize, numLayers, i);
		applyActivationFunction<<<(hostLayerSize[i+1] + 127)/128, 128>>>(devNeuronOutputs, devLayerSize, numLayers, i);
	}

	}

	
	int index = getArrayOffset(hostLayerSize, numLayers-1);
	float* out = (float*)malloc(sizeof(float));
	cudaMemcpy(out, devNeuronOutputs +index, sizeof(float)*1, cudaMemcpyDeviceToHost);
	printf("%f\n\n", *out);

	for(int i = 0; i < numLayers -1; i++)
	{
		parallelNNOutput<<<1, 128>>>(devNeuronOutputs, devNeuronWeights, devLayerSize, numLayers, i);
	}

	cudaMemcpy(out, devNeuronOutputs +index, sizeof(float)*1, cudaMemcpyDeviceToHost);
	printf("%f\n\n", *out);
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

	cuTrain<<<1,1>>>(devNeuronOutputs,  devNeuronErrors,  devNeuronWeights,  devLayerSize, numLayers, count, (rand()%10000-5000)/10000.0f);

	printf("====%d====\n", count);

	train(hostLayerSize, numLayers);
	

	//if(count % 10 == 0)
	{
		cuDisplay<<<gridSize, blockSize>>>(cudaSurfaceObject, width, height, devNeuronOutputs,  devNeuronErrors,  devNeuronWeights,  devLayerSize, numLayers, count, (rand()%10000-5000)/10000.0f, devValues, funcVals);
	}
}

