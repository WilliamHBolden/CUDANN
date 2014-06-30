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


	/*
	int testset[4][3] =
	{
		{0, 0, 0},
		{1, 0, 1},
		{0, 1, 1},
		{1, 1, 0}
	};

	float input[2][2];

	float bias0 = (rand()%10000 - 5000)/10000.0f;

	for(int x = 0; x < 2; x++)
	{
		for(int y = 0; y < 2; y++)
		{
			input[x][y] = (rand()%10000 - 5000)/10000.0f;   //-0.5 0.5
		}
	}

	float hidden[2][1];
	float bias1 = (rand()%10000 - 5000)/10000.0f;

	for(int x = 0; x < 2; x++)
	{
		for(int y = 0; y < 1; y++)
		{
			input[x][y] = (rand()%10000 - 5000)/10000.0f;   //-0.5 0.5
		}
	}
	
	float out;


	float inputOutput[2];
	float hiddenOutput[2];
	float finalOutput;

	while(true)
	{
		for(int i = 0; i < 4; i++)
		{
			inputOutput[0] = 


		}



	}
	*/





	while(true)
	{

		display(displayWindow.getCUDASurfaceObject(), WIDTH, HEIGHT);

		displayWindow.destroySO();
		displayWindow.displayFrame(window);

//		std::cout << "hi";
	}




	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}