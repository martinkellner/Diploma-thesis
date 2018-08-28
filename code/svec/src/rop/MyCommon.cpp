#include "MyCommon.h"
#include "MyUtils.h"

#include <string>
#include <fstream>
using namespace std;

Parameters loadParameters(string file) {
	Parameters r;
	ifstream ifs;
	ifs.open(file.c_str());
	
	ifs >> r.tiltNum >> r.tiltGaussianHeight >> r.tiltGaussianWidth;
	r.tiltPeaks = new double[r.tiltNum];
	for (unsigned int i=0; i<r.tiltNum; i++) {
		ifs >> r.tiltPeaks[i];
	}
	
	ifs >> r.versionNum >> r.versionGaussianHeight >> r.versionGaussianWidth;
	r.versionPeaks = new double[r.versionNum];
	for (unsigned int i=0; i<r.versionNum; i++) {
		ifs >> r.versionPeaks[i];
	}

	ifs >> r.xNum >> r.xGaussianHeight >> r.xGaussianWidth;
	r.xPeaks = new double[r.xNum];
	for (unsigned int i=0; i<r.xNum; i++) {
		ifs >> r.xPeaks[i];
	}

	ifs >> r.yNum >> r.yGaussianHeight >> r.yGaussianWidth;
	r.yPeaks = new double[r.yNum];
	for (unsigned int i=0; i<r.yNum; i++) {
		ifs >> r.yPeaks[i];
	}
	
	ifs >> r.retinaWidth >> r.retinaHeight;

	return r;
}

void printParameters(ostream &output, Parameters p) {

	MU::printlnValues(output, "idd", p.tiltNum, p.tiltGaussianHeight, p.tiltGaussianWidth);
	MU::printlnArr(output, p.tiltPeaks, p.tiltNum);
	output << endl;
	
	MU::printlnValues(output, "idd", p.versionNum, p.versionGaussianHeight, p.versionGaussianWidth);
	MU::printlnArr(output, p.versionPeaks, p.versionNum);
	output << endl;
	
	MU::printlnValues(output, "idd", p.xNum, p.xGaussianHeight, p.xGaussianWidth);
	MU::printlnArr(output, p.xPeaks, p.xNum);
	output << endl;
	
	MU::printlnValues(output, "idd", p.yNum, p.yGaussianHeight, p.yGaussianWidth);
	MU::printlnArr(output, p.yPeaks, p.yNum);
	output << endl;

	MU::printlnValues(output, "ii", p.retinaWidth, p.retinaHeight);
}