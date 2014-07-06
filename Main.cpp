#include "Display.h"
#include "NeuralNetwork.cuh"

/*
#include <iostream>
#include <stdlib.h>
#include <cmath>
*/


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
	displayWindow.createRenderer();


//	float* devNeuronOutpus;
//	float* devNeuronErrors;
//	float* devNeuronWeights;

	#define NUM 3

	int numLayers = NUM;

	int layerSize[NUM] = {2 + 1, 600 + 1, 1};  //+1 for biases

	createNN(layerSize, numLayers);
	

	while(true)
	{


		display(displayWindow.getCUDASurfaceObject(), WIDTH, HEIGHT, layerSize, numLayers);
		displayWindow.destroySO();
		displayWindow.displayFrame(window);
	}




	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}