#include "Display.h"
#include "NeuralNetwork.cuh"
#include "Timer.h"
#include "FileUtil.h"

#include <math.h> 
#include <cstdio>



#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "glu32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glfw3.lib")

#define WIDTH 1200
#define HEIGHT 600


int main()
{
	GLFWwindow* window;

	glfwInit();
	
	window = glfwCreateWindow(WIDTH, HEIGHT, "Neural Network", NULL, NULL);
	glfwMakeContextCurrent(window);

	Display displayWindow = Display(window);
	displayWindow.initRenderTexture();

	float* inputs = nullptr;
	float* outputs = nullptr;
	int numSets;
	int numInputs;
	int numOutputs;

	loadMNIST("Data/timages.mnist", 
		"Data/tlabels.mnist",
		&inputs, &outputs, 
		&numSets, &numInputs, &numOutputs);

	printf("%f", outputs[5]);

	FFNet network = FFNet();

//	network.createNN(layerSize, numLayers);
//	network.setTrainingData(inputs, outputs, numInputs, numSets, numOutputs);

//	network.randomTrain(1000);

	
	#define NUM 3
	
	int numLayers = NUM;
	int layerSize[NUM];

	layerSize[0] = numInputs;
	layerSize[1] = 400;
	layerSize[2] = numOutputs;

	for(int i =0; i < numLayers-1; i++) //+1 for biases
	{
		layerSize[i]++;
	}
	
//	createNN(layerSize, numLayers);

//	setTrainingData(inputs, outputs, numInputs, numSets, numOutputs);

	Timer timer = Timer();
	int iterations = 5000;
	int count = 0;
	int amt = 200;

	while(true)
	{
		timer.start();
//		randomTrain(layerSize, numLayers, iterations, 0.5f);
		timer.stop();

		count++;
		printf("%f\n", iterations*timer.getFPS());
		printf("====%d====\n", count);

		if(count % amt == -1)
		{
//			display(displayWindow.getCUDASurfaceObject(), WIDTH, HEIGHT, layerSize, numLayers);
			displayWindow.destroySO();
		}
		displayWindow.displayFrame(window);
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}