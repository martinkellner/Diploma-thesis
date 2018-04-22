#pragma once
#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H



namespace MatrixOp {	
	void addM31(double a[], double b[], double result[]);
	void subtractM31(double a[], double b[], double result[]);
	void multiplyM33M31(double m33[], double m31[], double res[]);
	void rotX(double alfa, double a[], double res[]);
	void rotY(double beta, double a[], double res[]);
}


#endif