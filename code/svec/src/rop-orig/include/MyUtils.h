#pragma once
#ifndef MY_UTILS_H
#define MY_UTILS_H


#include <ostream>
#include <string>

using namespace std;

namespace MU {
		
	extern const double PI; 
	extern const double EULER; 

	bool theSame(double x, double y);
	double fRand(double fMin, double fMax);		

	void printArr(std::ostream &output, double *arr, unsigned int length, std::string del = " ");	
	void printlnArr(std::ostream &output, double *arr, unsigned int length, std::string del = " ");
	
	void printValues(std::ostream &output, char *szTypes, ...);		

	void printlnValues(std::ostream &output, char *szTypes, ...);			

	void printlnCSValues(std::ostream &output, char *szTypes, ...);		

	void pause(char* msg = " press ENTER to continue...");
}


#endif