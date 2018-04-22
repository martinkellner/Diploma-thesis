
#pragma once
#ifndef BOTTLE_WRAPPER_H
#define BOTTLE_WRAPPER_H

#include <yarp/os/Bottle.h>
using namespace yarp::os;


class BottleWrapper {

	//Bottle getBottle(char* s);	
	
public: 
	static Bottle prepareBottle(char *szTypes, ...);	
	
	//creation
	static Bottle box(double sx, double sy, double sz, double px, double py, double pz, double r=1, double g=0, double b=0);
	static Bottle cyl(double radius, double length, double px, double py, double pz, double r=1, double g=0, double b=0);
	static Bottle sph(double radius, double px, double py, double pz, double r=1, double g=0, double b=0);
	
	//rotation
	static Bottle r_box(int index, double rx, double ry, double rz);
	static Bottle r_cyl(int index, double rx, double ry, double rz);
	static Bottle r_sph(int index, double rx, double ry, double rz);	

};






#endif