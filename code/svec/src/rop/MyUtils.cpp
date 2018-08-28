#include "MyUtils.h"

#include <limits>
#include <algorithm>
#include <ostream>
#include <iostream>
#include <stdarg.h>
#include <cmath>
#include <iomanip>
#include <string>
using namespace std;


const double MU::PI = atan(1.0)*4;
const double MU::EULER = 2.71828182845904523536;

bool MU::theSame(double x, double y) {
	const double EPSILON = numeric_limits<double>::epsilon();	
	return (abs(x - y) <= EPSILON * max(max(1.0, abs(x)), abs(y)));
}


double MU::fRand(double fMin, double fMax) {
    double f = (double)rand() / (double)RAND_MAX;
    return fMin + f * (fMax - fMin);
}


void MU::printArr(ostream &output, double *arr, unsigned int length, string del) {
	for (unsigned int i=0; i<length; i++) {
		if (i>0) output << del;
		output << arr[i];
	}	
}

void MU::printlnArr(ostream &output, double *arr, unsigned int length, string del) {
	printArr(output, arr, length, del);
	output << endl;
}

void MU::printValues(ostream &output, char *szTypes, ...) {
	va_list vl;	
	va_start(vl, szTypes);
	char * del = " ";
	for (int i=0; szTypes[i] != '\0'; i++) {
		if (i>0) output << del;
		switch( szTypes[i] ) {
			case '_' :
				del = va_arg( vl, char * );				 
				break; 
			case 'i':
				output << va_arg( vl, int);				 
				break;
			case 'f':
				output << va_arg( vl, float );				
				break;			
			case 'd':
				output << va_arg( vl, double );				
				break;			
			case 's':
				output << va_arg( vl, char * );				 
				break;
			default:
				output << " unknow arg " << szTypes[i];
				break;
		  }		
	}
}


void MU::printlnValues(ostream &output, char *szTypes, ...) {
	va_list vl;	
	va_start(vl, szTypes);
	for (int i=0; szTypes[i] != '\0'; i++) {
		output << " ";
		switch( szTypes[i] ) {   
			case 'i':
				output << va_arg( vl, int);				 
				break;
			case 'f':
				output << va_arg( vl, float );				
				break;			
			case 'd':
				output << setw(12) << va_arg( vl, double );				
				break;			
			case 's':
				output << va_arg( vl, char * );				 
				break;
			default:
				output << " unknown arg " << szTypes[i];
				break;
		  }		
	}
	output << endl;
}

void MU::printlnCSValues(ostream &output, char *szTypes, ...) {
	va_list vl;	
	va_start(vl, szTypes);
	for (int i=0; szTypes[i] != '\0'; i++) {
		if (i>0) output << ",";
		switch( szTypes[i] ) {   
			case 'i':
				output << va_arg( vl, int);				 
				break;
			case 'f':
				output << va_arg( vl, float );				
				break;			
			case 'd':
				output << va_arg( vl, double );				
				break;			
			case 's':
				output << va_arg( vl, char * );				 
				break;
			default:
				output << " unknown arg " << szTypes[i];
				break;
		  }		
	}
	output << endl;
}

void MU::pause(char* msg) {
	cout << msg;	
	cin.get();
}
