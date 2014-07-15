#include "Display.h"
#include "NeuralNetwork.cuh"
#include "Timer.h"

#include <math.h> 
#include <cstdio>

#include <fstream>


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

#include <Windows.h>

int main()
{
	GLFWwindow* window;

	glfwInit();
	
	window = glfwCreateWindow(WIDTH, HEIGHT, "Neural Network", NULL, NULL);
	glfwMakeContextCurrent(window);

	Display displayWindow = Display(window);
	displayWindow.initRenderTexture();


	std::ifstream image;
	image.open("C:\\Users\\Will\\Desktop\\gaben3.bmp", std::ios::binary);

	UINT8* dbuf[2] = {nullptr, nullptr};

	BITMAPFILEHEADER* bhead = nullptr;
	BITMAPINFOHEADER* binf = nullptr;


	dbuf[0] = new UINT8[sizeof(BITMAPFILEHEADER)];
	dbuf[1] = new UINT8[sizeof(BITMAPINFOHEADER)];

	image.read((char*)dbuf[0], sizeof(BITMAPFILEHEADER));
	image.read((char*)dbuf[1], sizeof(BITMAPINFOHEADER));

	bhead =(BITMAPFILEHEADER*) dbuf[0];
	binf =(BITMAPINFOHEADER*) dbuf[1];

	//image.seekg(0, std::ios::end);
	//int n = image.tellg();
	UINT8* res = new UINT8[binf->biSizeImage];

	//image.seekg(0);
	image.seekg(bhead->bfOffBits);

	image.read((char*)res, binf->biSizeImage);

	


	//image.eof();

	//image.read(res, n);

	//printf("%ui\n%ui\n%ui\n%ui\n%ui\n%ui\n", (unsigned int)res[0], (unsigned int)res[1], (unsigned int)res[2], (unsigned int)res[3], (unsigned int)res[4], (unsigned int)res[5]);

	
	#define NUM 4
#define NUMIN 16
	int numLayers = NUM;
	int layerSize[NUM] = {NUMIN + 1, 128 + 1, 128 + 1, 3};  //+1 for biases
	
	createNN(layerSize, numLayers);


	float* inputs = (float*)malloc(sizeof(float)*(400*400*NUMIN));
	float* outputs = (float*)malloc(sizeof(float)*binf->biSizeImage);


	
	for(unsigned int i = 0; i < binf->biSizeImage; i+=3)
	{
		
		outputs[i] = (float)res[i+2] / 255.0f;
		outputs[i+1] = (float)res[i+1] / 255.0f;
		outputs[i+2] = (float)res[i] / 255.0f;

		
	//	outputs[i/3] = (float)(res[i]) / 255.0f;

	//	outputs[i/3] = 0.5;
		
		//outputs[i] = (float)res[i] / 255.0f;
		
		/*
		outputs[i/3] = 1.0;
		if(i > 400*400)
		{
			outputs[i/3] = 0;
		}
		*/
		

	}

	//printf("%f  %f  %f", outputs[0], outputs[1], outputs[2]);

#define PNUM 400
#define HPNUM (PNUM/2)

	for(int x = 0; x < PNUM; x++)
	{
		for(int y = 0; y < PNUM; y++)
		{
			float fx = (float)(x - HPNUM);
			float fy = (float)(y - HPNUM);


			int index = y * PNUM*NUMIN + x*NUMIN;

			float tfx = fx/((float)HPNUM);
			float tfy = fy/((float)HPNUM);

			inputs[index] = tfx;
			inputs[index+1] = tfy;

			for(int i =2; i < NUMIN; i+=2)
			{

				inputs[index + i] = sinf(tfx * (i/2) *2 * 3.14159);
				inputs[index + i + 1] = sinf(tfy * (i/2) *2* 3.14159);
			}
		}
	}


	setTrainingData(inputs, outputs, NUMIN, PNUM*PNUM, 3);

	/*
	for(int x = -200; x < 200; x++)
	{
		for(int y = -200; y < 200; y++)
		{
			float fx = (float)x/1;
			float fy = (float)y/1;


			int index = (y+200) * 800 + (x+200)*2;

			inputs[index] = fx/100.0f;
			inputs[index+1] = fy/100.0f;
		}
	}

	

	setTrainingData(inputs, outputs, 2, 400*400, 1);
	*/

	
	/*
	float inputs[8] = {1, 0, 0, 1, 1, 1, 0, 0};
	float outputs[4] = {1, 1, 0, 0};

	setTrainingData(inputs, outputs, 2, 4, 1);

	*/
	
	//sin(x*y) training set
	

	
//	float inputs[1*400*2];
//	float outputs[1*400];
	
	/*
	float* inputs = (float*)malloc(sizeof(float)*(400*400*2));
	float* outputs = (float*)malloc(sizeof(float)*400*400);

	for(int x = -200; x < 200; x++)
	{
		for(int y = -200; y < 200; y++)
		{
			float fx = (float)x/35.0f;
			float fy = (float)y/35.0f;

//			printf("%d     %d", x ,y);

			int index = (y+200) * 800 + (x+200)*2;

			outputs[(y+200) * 400 + (x+200)] = (((sin(fx * fy)+1)/2)/1.25f +0.1f);

			inputs[index] = fx/1.0f;
			inputs[index+1] = fy/1.0f;

			

		}
	}

	

	setTrainingData(inputs, outputs, 2, 400*400, 1);
	
	*/
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
	int count = 0;
	int amt = 25;

	while(true)
	{
		timer.start();
		
		randomTrain(layerSize, numLayers, iterations, 0.5f);
	//	orderedTrain(layerSize, numLayers, iterations, 0.3f);

		timer.stop();
		
		printf("%f\n", iterations*timer.getFPS());
		
		count++;

		printf("====%d====\n", count);

		if(count % amt == 0)
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