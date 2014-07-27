#include "FileUtil.h"
#include <fstream>
#include <iostream>
#include <string.h>

union bits
{
	unsigned int n;
	unsigned char c[4];
};

int reverseBytes(int i)
{
	bits n;
	n.n = i;
	unsigned char swap;

	swap = n.c[0];
	n.c[0] = n.c[3];
	n.c[3] = swap;

	swap = n.c[1];
	n.c[1] = n.c[2];
	n.c[2] = swap;

	return n.n;
}

void loadMNIST(const char* imageFilepath, const char* labelFilepath, float** inputData, float** outputData, int* numSets, int* setSize, int* outputsPerSet)
{
	std::ifstream imageFile(imageFilepath, std::ios::binary);
	std::ifstream labelFile(labelFilepath, std::ios::binary);

	int magicNumber;
	int numItems;

	labelFile.read((char*)&magicNumber, sizeof(int));
	magicNumber = reverseBytes(magicNumber);
	labelFile.read((char*)&numItems, sizeof(int));
	numItems = reverseBytes(numItems);

	char* labels = (char*)malloc(sizeof(char)*numItems);
	labelFile.read(labels, sizeof(char)*numItems);
	labelFile.close();

	*numSets = numItems;
	*outputsPerSet = 10; //For the digits [0, 9]

	*outputData = (float*)malloc(sizeof(float)**numSets**outputsPerSet);
	memset(*outputData, 0, sizeof(float)**numSets**outputsPerSet);

	for(int i =0; i < *numSets; i++)
	{
		(*outputData)[i * 10 + (int)labels[i]] = 1.0f; 
	}
	
	free(labels);


	int rows;
	int cols;

	imageFile.read((char*)&magicNumber, sizeof(int));
	magicNumber = reverseBytes(magicNumber);
	imageFile.read((char*)&numItems, sizeof(int));
	numItems = reverseBytes(numItems);

	imageFile.read((char*)&rows, sizeof(int));
	rows = reverseBytes(rows);
	imageFile.read((char*)&cols, sizeof(int));
	cols = reverseBytes(cols);

	unsigned char* pixels = (unsigned char*)malloc(sizeof(char)*numItems*rows*cols);
	imageFile.read((char*)pixels, sizeof(char)*numItems*rows*cols);
	imageFile.close();

	*setSize = rows*cols;
	*inputData = (float*)malloc(sizeof(float)*numItems*rows*cols);

	for(int i = 0; i < numItems*rows*cols; i++)
	{
		(*inputData)[i] = (float)pixels[i]/255.0f;
	}

	free(pixels);


}

/*
void saveNN(const NNInfo& info)
{
	std::ofstream output;

	output.open("network.NN");

	output << info.numOutputs << std::endl
		   << info.numSets << std::endl
		   << info.setSize << std::endl;


	output.close();
}

*/
/*
NNInfo* loadNN(const char* weightFilepath)
{

}
*/