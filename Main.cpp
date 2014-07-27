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
	int setSize;
	int outputsPerSet;

	
	loadMNIST("C:\\Users\\Will\\Desktop\\timages.mnist", 
		"C:\\Users\\Will\\Desktop\\tlabels.mnist",
		&inputs, &outputs, 
		&numSets, &setSize, &outputsPerSet);

	printf("%f", outputs[5]);

	
	#define NUM 3
	#define NUMIN 16
	
	int numLayers = NUM;
	int layerSize[NUM];

	layerSize[0] = setSize;
	layerSize[1] = 400;
	layerSize[2] = outputsPerSet;

	for(int i =0; i < NUM-1; i++) //+1 for biases
	{
		layerSize[i]++;
	}
	
	createNN(layerSize, numLayers);

	setTrainingData(inputs, outputs, setSize, numSets, outputsPerSet);

	Timer timer = Timer();
	int iterations = 5000;
	int count = 0;
	int amt = 200;

	while(true)
	{
		timer.start();
		randomTrain(layerSize, numLayers, iterations, 0.5f);
		timer.stop();

		count++;
		printf("%f\n", iterations*timer.getFPS());
		printf("====%d====\n", count);

		if(count % amt == -1)
		{
			display(displayWindow.getCUDASurfaceObject(), WIDTH, HEIGHT, layerSize, numLayers);
			displayWindow.destroySO();
		}
		displayWindow.displayFrame(window);
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}