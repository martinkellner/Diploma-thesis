#pragma once
#ifndef MY_SIM_H
#define MY_SIM_H

#include <string.h>

#include <yarp/os/Bottle.h>
#include <yarp/os/RpcClient.h>
#include <yarp/os/BufferedPort.h>
using namespace yarp::os;

#include <yarp/sig/Image.h>
using namespace yarp::sig;

#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IPositionControl.h>
using namespace yarp::dev;

using namespace std;

class MySim {

private:	
	RpcClient *rpcWorld;	
	IPositionControl* eyesPosControl;
	PolyDriver* headDriver;
	BufferedPort<ImageOf<PixelRgb>> *leftEye, *rightEye;
	ostream *output;

protected:
	double
		eyeLx, eyeLy, eyeLz,
		eyeRx, eyeRy, eyeRz,				
		eyesTilt, eyesVersion,
		bodyCenterY,
		EYE_F,
		ZF_WIDTH, ZF_HEIGHT;	
	string 
		robotName,
		ownPort,
		worldPort,
		headPort,
		leftCamPort,
		rightCamPort;	
	bool verbose;
		
	double myScale(double minz, double maxz, double sminz, double smaxz, double z);
	void putInFOV(double &x, double &y, double &z);

	string getPortFullName(string port, bool own=true);
	bool connect(string port, bool write);
	bool disconnect(string port);

	RpcClient *getRpcWorld();	

	PolyDriver *getHeadDriver();
	IPositionControl *getEyesPositionControl();
	BufferedPort<ImageOf<PixelRgb>> *getLeftEye();
	BufferedPort<ImageOf<PixelRgb>> *getRightEye();

public:
	MySim(string robotName="/icubSim", string ownPortName="/mysim");
	~MySim();

	Bottle deleteAllObjects();
	Bottle putObjectBeforeEye_left();
	Bottle putRandomObjectWRotation(bool inFOV=true, double z=-1);
	
	void turnEyes(double tilt, double version, bool waitForDone=true);		
	ImageOf<PixelRgb> *getLeftEyeImage();
	ImageOf<PixelRgb> *getRightEyeImage();
	void disconnectLeftEye();
	void disconnectRightEye();
	
	void setOutput(ostream *output);
};




#endif