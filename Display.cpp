#include "Display.h"
#include <cuda_gl_interop.h>

Display::Display(GLFWwindow* window)
{
	glfwGetWindowSize(window, &width, &height);
}

void Display::initRenderTexture()
{
	glewInit();

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &vtex);

	glBindTexture(GL_TEXTURE_2D, vtex);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	//GL_RGB
	glBindTexture(GL_TEXTURE_2D, 0);

	cudaGraphicsGLRegisterImage(&cudavResource, vtex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

cudaSurfaceObject_t Display::getCUDASurfaceObject()
{
	cudaGraphicsMapResources(1, &cudavResource);

	cudaArray_t cudavArray;
	cudaGraphicsSubResourceGetMappedArray(&cudavArray, cudavResource, 0, 0);

	cudaResourceDesc cudavArrayResourceDesc;
	cudavArrayResourceDesc.resType = cudaResourceTypeArray;
	cudavArrayResourceDesc.res.array.array = cudavArray;
	
	//cudaSurfaceObject_t cudavSurfaceObject;
	cudaCreateSurfaceObject(&cudavSurfaceObject, &cudavArrayResourceDesc);

	cudaGraphicsUnmapResources(1, &cudavResource);

	return cudavSurfaceObject;
}

void Display::destroySO()
{
	cudaDestroySurfaceObject(cudavSurfaceObject);
	cudaStreamSynchronize(0);
}

void Display::displayFrame(GLFWwindow* window)
{
	glBindTexture(GL_TEXTURE_2D,vtex);
	
	glBegin(GL_QUADS);
		
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
		
	glEnd();
	
	glBindTexture(GL_TEXTURE_2D, 0);

	glfwSwapBuffers(window);
	glfwPollEvents();
}

Display::~Display(void)
{
}
