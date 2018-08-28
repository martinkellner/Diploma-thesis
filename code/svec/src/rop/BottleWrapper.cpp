#include "BottleWrapper.h"

#include <stdio.h>
#include <stdarg.h>

#include <yarp/os/Bottle.h>
using namespace yarp::os;

Bottle BottleWrapper::prepareBottle(char *szTypes, ...) {
	va_list vl;	
	Bottle bottle;
	va_start(vl, szTypes);
	for (int i=0; szTypes[i] != '\0'; i++) {
		switch( szTypes[i] ) {   
			case 'i':
				bottle.addInt(va_arg( vl, int ));				 
				break;
			case 'f':
				bottle.addDouble(va_arg( vl, double ));				
				break;			
			case 's':
				bottle.addString(va_arg( vl, char * ));				 
				break;
			default:
				break;
		  }
	}
	return bottle;
}

Bottle BottleWrapper::box(double sx, double sy, double sz, double px, double py, double pz, double r, double g, double b) {
	return prepareBottle("sssfffffffff", "world", "mk", "sbox", sx, sy, sz, px, py, pz, r, g, b);
}

Bottle BottleWrapper::cyl(double radius, double length, double px, double py, double pz, double r, double g, double b) {
	return prepareBottle("sssffffffff", "world", "mk", "scyl", radius, length, px, py, pz, r, g, b);
}

Bottle BottleWrapper::sph(double radius, double px, double py, double pz, double r, double g, double b) {
	return prepareBottle("sssfffffff", "world", "mk", "ssph", radius, px, py, pz, r, g, b);	
}

Bottle BottleWrapper::r_box(int index, double rx, double ry, double rz) {
	return prepareBottle("sssifff", "world", "rot", "sbox", index, rx, ry, rz);	
}

Bottle BottleWrapper::r_cyl(int index, double rx, double ry, double rz) {
	return prepareBottle("sssifff", "world", "rot", "scyl", index, rx, ry, rz);
}

Bottle BottleWrapper::r_sph(int index, double rx, double ry, double rz) {
	return prepareBottle("sssifff", "world", "rot", "ssph", index, rx, ry, rz);
}

/*
Bottle MySim::getBottle(char* ss) {		
	Bottle bottle;	
	char *s = ss;
	while(1) {	
		const char *begin = s;
    	while(*s != ' ' && *s!='\0' && *s) {
    		s++;
		}		
		string str = string(begin, s);
		if (str!="") {					
			bottle.addString(str.c_str());
		}		
    	if(s==NULL || *s=='\0')
    		break;
		s++;
    }	
	return bottle;	
}*/