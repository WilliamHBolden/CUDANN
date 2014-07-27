#ifndef DISPLAY_H
#define DISPLAY_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <GL/glu.h>

class Display
{
public:
	Display(GLFWwindow* window);
	void initRenderTexture();
	void resizeWindow(GLFWwindow* window);
	void displayFrame(GLFWwindow* window);
	cudaSurfaceObject_t getCUDASurfaceObject();
	void destroySO();

	//void draw();

	~Display(void);

private:
	int width, height;
	GLuint vtex;
	cudaGraphicsResource_t cudavResource;
	cudaSurfaceObject_t cudavSurfaceObject;
};

#endif
