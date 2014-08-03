#ifndef NEURALNETWORK_CUH
#define NEURALNETWORK_CUH


struct NNInfo
{
	int numSets;
	int setSize;
	int numOutputs;
	float* weights;
	int numWeights;
};

void display(cudaSurfaceObject_t cudaSurfaceObject, int width, int height, int* layerSize, int numLayers);
void createNN(int* layerSize, int numLayers);

/*
	inputs : a flat array of input data
	setSize : number of floats in the array making up one input
	numItems : total number of inputs
	numOutputs : number of outputs per input
*/
void setTrainingData(float* inputs, float* outputs,int setSize, int numItems, int numOutputs);

void setTrainingData(float* inputs, float* outputs, NNInfo info);

/*
	layerSize : host array of layer sizes
	numLayers : number of layers
	iterations : the number of iterations through the input set
	learningRate : how fast the neural net learns, set between 0.3 and 0.8
*/
void orderedTrain(int* hostLayerSize, int numLayers, int iterations, float learningRate);

/*
	layerSize : host array of layer sizes
	numLayers : number of layers
	iterations : the number of iterations through the input set
	learningRate : how fast the neural net learns, set between 0.3 and 0.8
*/
void randomTrain(int* hostLayerSize, int numLayers, int iterations, float learningRate);

class FFNet
{
public:
	FFNet(void);

	void randomTrain(int iterations);
	void orderedTrain(int iterations);
	float validate();

	void createNN(int* layerSize, int numLayers, float learningRate, float momentum);

	void setLearningRate(float learningRate);
	void setMomentum(float momentum);

	void setTrainingData(float* inputs, float* outputs, int setSize, int numSets, int outputsPerSet);
	void setValidationData(float* inputs, float* outputs, int setSize, int numSets, int outputsPerSet);

	~FFNet(void);

private:
	float learningRate;
	int numLayers;
	int* hostLayerSize;

	float* devNeuronWeights;
	float* devNeuronOutputs;
	float* devNeuronErrors;
	float* devDefaultOutputs;
	float* devWeightDeltas;
	int* devLayerSize;

	float* devTrainingInputs;
	float* devTrainingOutputs;

	int setSize;
	int numItems;
	int numOutputs;

	void forwardPropagation(unsigned int index);
	void backPropagation(unsigned int index);
};

class RecurrentNet
{


};

#endif