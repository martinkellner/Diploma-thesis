#include "MatrixOperations.h"

#include <math.h>


void MatrixOp::addM31(double a[], double b[], double result[]) {
	for (int i=0; i<3; i++) {
		result[i] = a[i] + b[i];
	}	
}

void MatrixOp::subtractM31(double a[], double b[], double result[]) {	
	for (int i=0; i<3; i++) {
		result[i] = a[i] - b[i];
	}	
}

void MatrixOp::multiplyM33M31(double m33[], double m31[], double res[]) {		
	for (int i=0; i<3; i++) {		//apply rotation (multiply matrix)
		res[i] = 0;
		for (int j=0; j<3; j++) {			
			res[i] += m33[i*3+j] * m31[j]; 
		}		
	}	
}

void MatrixOp::rotX(double alfa, double a[], double res[]) {
	double rtx[9] = {
		1.0, 0.0, 0.0, 
		0.0, cos(alfa), -sin(alfa),
		0.0, sin(alfa), cos(alfa)
	};
	MatrixOp::multiplyM33M31(rtx, a, res);	
}

void MatrixOp::rotY(double beta, double a[], double res[]) {
	double rty[9] = {
			cos(beta), 0 , sin(beta),
			0, 1, 0,
			-sin(beta), 0, cos(beta)
	};
	MatrixOp::multiplyM33M31(rty, a, res);	
}

