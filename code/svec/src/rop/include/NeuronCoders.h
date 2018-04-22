
#pragma once
#ifndef NEURON_CODERS_H
#define NEURON_CODERS_H

#include <ostream>

namespace NeuronCoders {	
	
	
	void uniformPeakDistribution(double rangeMin, double rangeMax, unsigned int numOfNeurons, double peaksPositions[]);

	double gaussian(double x, double height, double peekPosition, double widthCoef);
	void inverseGaussian(double y, double height, double peekPosition, double widthCoef, double &x1, double &x2);


	void valueToNeuronArray(double value, unsigned int numberOfNeurons, double peaks[], double *output, double gaussian_height, double gaussian_width);	
	void valueToNeuronArray(double value, unsigned int numberOfNeurons, double peaks[], std::ostream &outputStream, double gaussian_height, double gaussian_width);

	//void valueToNeuronArray(double value, double min, double max, unsigned int numberOfNeurons, double *output, double gaussian_width = 1.0, double gaussian_height = 1.0);			
	//void valueToNeuronArray(double value, double min, double max, unsigned int numberOfNeurons, std::ostream &outputStream, double gaussian_width = 1.0, double gaussian_height = 1.0);

	double neuronArrayToValue(double* arr, unsigned int numberOfNeurons, double peaks[], double gaussian_height, double gaussian_width);
	double neuronArrayToValue1(double* arr, unsigned int numberOfNeurons, double peaks[], double gaussian_height, double gaussian_width);
	double neuronArrayToValue2(double* arr, unsigned int numberOfNeurons, double peaks[], double gaussian_height, double gaussian_width);
	double neuronArrayToValue3(double* arr, unsigned int numberOfNeurons, double peaks[], double gaussian_height, double gaussian_width);
	
}


#endif