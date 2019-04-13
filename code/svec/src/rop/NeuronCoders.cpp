
#include "NeuronCoders.h"
#include "MyUtils.h"

#include <iostream>
#include <math.h>

using namespace std;


//v rangeMin aj v rangeMax bude neuron
void NeuronCoders::uniformPeakDistribution(double rangeMin, double rangeMax, unsigned int numOfNeurons, double peaksP ositions[]) {
	double step = ( rangeMax - rangeMin) / (numOfNeurons - 1);
	double val = rangeMin;
	for (unsigned int i=0; i<numOfNeurons; i++) {
		peaksPositions[i] = val;
		val += step;
	}	
}


double NeuronCoders::gaussian(double x, double height, double peakPosition, double widthCoef) {	
	return pow(height*MU::EULER, (-1)*((x-peakPosition)*(x-peakPosition))/(2*widthCoef*widthCoef));	
}

void NeuronCoders::inverseGaussian(double y, double height, double peakPosition, double widthCoef, double &x1, double &x2) {
	double a = 1.0;
	double b = -2.0*peakPosition;
	double c = peakPosition*peakPosition + 2.0*widthCoef*widthCoef*log(y/height);

	double determinant = b*b - 4.0*a*c;
	x1 = (-b - sqrt(determinant)) / (2.0*a);	
	x2 = (-b + sqrt(determinant)) / (2.0*a);
}


void NeuronCoders::valueToNeuronArray(double value, unsigned int numberOfNeurons, double peaks[], double *output, double gaussian_height, double gaussian_width) {	
	for (unsigned int i=0; i<numberOfNeurons; i++) {				
		output[i] = gaussian(value, gaussian_height, peaks[i], gaussian_width); 		 	
	}	
}

           
void NeuronCoders::valueToNeuronArray(double value, unsigned int numberOfNeurons, double peaks[], ostream &outputStream, double gaussian_height, double gaussian_width) {	
	for (unsigned int i=0; i<numberOfNeurons; i++) {
		double res = gaussian(value, gaussian_height, peaks[i], gaussian_width); 
		if (i>0) { outputStream << " "; };		
		outputStream << res;
	}
	outputStream << endl;
}

/*
void NeuronCoders::valueToNeuronArray(double value, double min, double max, unsigned int numberOfNeurons, double *output, double gaussian_width, double gaussian_height) {
	double step = (max - min)/(numberOfNeurons-1);
	for (unsigned int i=0; i<numberOfNeurons; i++) {
		double x = min + (double)i*step;		
		double res = gaussian(value, gaussian_height, x, gaussian_width); 
		output[i] = res;		
	}	
}

           
void NeuronCoders::valueToNeuronArray(double value, double min, double max, unsigned int numberOfNeurons, ostream &outputStream, double gaussian_width, double gaussian_height) {
	double step = (max - min)/(numberOfNeurons-1);
	for (unsigned int i=0; i<numberOfNeurons; i++) {
		double x = min + (double)i*step;		
		double res = gaussian(value, gaussian_height, x, gaussian_width); 
		if (i>0) { outputStream << " "; };		
		outputStream << res;
	}
	outputStream << endl;
}
*/


double NeuronCoders::neuronArrayToValue1(double* arr, unsigned int numberOfNeurons, double peaks[], double gaussian_height, double gaussian_width) {	
	double res = 0;
	unsigned int peekIndex = -1;
	double max = 0;
	for (unsigned int i=0; i<numberOfNeurons; i++) {		
		if (arr[i] > max) {
			peekIndex = i;
			max = arr[i];
		}
	}
	double x1=0, x2=0;		
	inverseGaussian(arr[peekIndex], gaussian_height, peaks[peekIndex], gaussian_width, x1, x2);	
		
	if ((peekIndex == 0) || ((peekIndex < numberOfNeurons-1) && (arr[peekIndex+1]>arr[peekIndex-1]))) {
		return x2;
	} else {
		return x1;
	}	
}

double NeuronCoders::neuronArrayToValue2(double* arr, unsigned int numberOfNeurons, double peaks[], double gaussian_height, double gaussian_width) {	
	double res = 0;
	unsigned int peekIndex = -1;
	double max = 0;
	for (unsigned int i=0; i<numberOfNeurons; i++) {		
		if (arr[i] > max) {
			peekIndex = i;
			max = arr[i];
		}
	}	
	double x1=0, x2=0;		
		
	if ((peekIndex == 0) || ((peekIndex < numberOfNeurons-1) && (arr[peekIndex+1]>arr[peekIndex-1]))) {
		inverseGaussian(arr[peekIndex+1], gaussian_height, peaks[peekIndex+1], gaussian_width, x1, x2);	
		return x1;
	} else {
		inverseGaussian(arr[peekIndex-1], gaussian_height, peaks[peekIndex-1], gaussian_width, x1, x2);	
		return x2;		
	}	
}

double NeuronCoders::neuronArrayToValue3(double* arr, unsigned int numberOfNeurons, double peaks[], double gaussian_height, double gaussian_width) {	
	double res = 0;
	unsigned int peekIndex = -1;
	double max = 0;
	for (unsigned int i=0; i<numberOfNeurons; i++) {		
		if (arr[i] > max) {
			peekIndex = i;
			max = arr[i];
		}
	}	
	double x1=0, x2=0;		
		
	if ((peekIndex == 0) || (peekIndex == numberOfNeurons-1)) {
		return -1000;
	} else if (arr[peekIndex+1]>arr[peekIndex-1]) {
		inverseGaussian(arr[peekIndex-1], gaussian_height, peaks[peekIndex-1], gaussian_width, x1, x2);	
		return x2;
	} else {
		inverseGaussian(arr[peekIndex+1], gaussian_height, peaks[peekIndex+1], gaussian_width, x1, x2);	
		return x1;		
	}	
}

double NeuronCoders::neuronArrayToValue(double* arr, unsigned int numberOfNeurons, double peaks[], double gaussian_height, double gaussian_width) {	
	double res = 0;
	unsigned int peakIndex = -1;
	double max = 0;
	for (unsigned int i=0; i<numberOfNeurons; i++) {		
		if (arr[i] > max) {
			peakIndex = i;
			max = arr[i];
		}
	}
	double x1=0, x2=0, x=0;		
	int num = 1;
	inverseGaussian(arr[peakIndex], gaussian_height, peaks[peakIndex], gaussian_width, x1, x2);		
	if ((peakIndex == 0) || ((peakIndex < numberOfNeurons-1) && (arr[peakIndex+1]>arr[peakIndex-1]))) {
		x = x2;
	} else {
		x = x1;
	}
	if (peakIndex>0) {
		inverseGaussian(arr[peakIndex-1], gaussian_height, peaks[peakIndex-1], gaussian_width, x1, x2);
		x += x2;
		num++;	
	}
	if (peakIndex<numberOfNeurons-1) {
		inverseGaussian(arr[peakIndex+1], gaussian_height, peaks[peakIndex+1], gaussian_width, x1, x2);
		x += x1;
		num++;
	}
	x = x/(double)num;
	return x;
}