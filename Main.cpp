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
	int numLayers = 3;

	int layerSize[3] = {2 + 1, 5 + 1 ,1};  //+1 for biases

	createNN(layerSize, numLayers);
	

	while(true)
	{

	//	cudaSurfaceObject_t temp = displayWindow.getCUDASurfaceObject();

		display(displayWindow.getCUDASurfaceObject(), WIDTH, HEIGHT, layerSize, numLayers);




		displayWindow.destroySO();
		displayWindow.displayFrame(window);

//		std::cout << "hi";
	}




	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}