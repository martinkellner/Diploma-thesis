#pragma once
#ifndef MY_COMMON_H
#define MY_COMMON_H

#include <string>
using namespace std;

struct Parameters {
	unsigned int tiltNum;
	double tiltGaussianHeight, tiltGaussianWidth;
	double *tiltPeaks;

	unsigned int versionNum;
	double versionGaussianHeight, versionGaussianWidth;
	double *versionPeaks;
	
	unsigned int xNum;
	double xGaussianHeight, xGaussianWidth;
	double *xPeaks;

	unsigned int yNum;
	double yGaussianHeight, yGaussianWidth;
	double *yPeaks;

	unsigned int retinaWidth, retinaHeight;
};

Parameters loadParameters(string file);

void printParameters(ostream &output, Parameters p);


#endif