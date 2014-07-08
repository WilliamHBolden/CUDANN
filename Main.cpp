#include "Display.h"
#include "NeuralNetwork.cuh"
#include "Timer.h"

#include <math.h> 
#include <cstdio>

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


	//glfwRead





	
	#define NUM 4
	int numLayers = NUM;
	int layerSize[NUM] = {2 + 1, 128 + 1, 128 +1, 1};  //+1 for biases
	
	createNN(layerSize, numLayers);
	
	/*
	float inputs[8] = {1, 0, 0, 1, 1, 1, 0, 0};
	float outputs[4] = {1, 1, 0, 0};

	setTrainingData(inputs, outputs, 2, 4, 1);

	*/
	
	//sin(x*y) training set
	

	
//	float inputs[1*400*2];
//	float outputs[1*400];
	
	
	float* inputs = (float*)malloc(sizeof(float)*(400*400*2));
	float* outputs = (float*)malloc(sizeof(float)*400*400);

	for(int x = -200; x < 200; x++)
	{
		for(int y = -200; y < 200; y++)
		{
			float fx = (float)x/25.0f;
			float fy = (float)y/25.0f;

//			printf("%d     %d", x ,y);

			int index = (y+200) * 800 + (x+200)*2;

			inputs[index] = fx;
			inputs[index+1] = fy;

			outputs[(y+200) * 400 + (x+200)] = (((sin(fx * fy)+1)/2)/1.25f +0.1f);

		}
	}

	

	setTrainingData(inputs, outputs, 2, 400*400, 1);
	
	
	//sin(x) training set
	/*
	float inputs[6280];
	float outputs[6280];

	for(int x = -3140; x <= 3140; x++)
	{
		float fx = (float)x/1000.0f;

		inputs[x+3140] = fx;
		outputs[x+3140] = (((sin(fx)+1)/2)/1.25f +0.1f);
	}

	setTrainingData(inputs, outputs, 1, 6280, 1);
	*/

	Timer timer = Timer();
	int iterations = 50000;

	while(true)
	{
		timer.start();
		

		randomTrain(layerSize, numLayers, iterations, 0.8f);

		timer.stop();
//		if(timer.getFrameCount() % 10 == 0)
		{
			printf("%f\n", iterations*timer.getFPS());
		}
		


		display(displayWindow.getCUDASurfaceObject(), WIDTH, HEIGHT, layerSize, numLayers);
		displayWindow.destroySO();
		displayWindow.displayFrame(window);
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	
	return 0;
}