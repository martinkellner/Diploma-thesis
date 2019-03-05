#include "MyUtils.h"
#include "BottleWrapper.h"
#include "MatrixOperations.h"

#include <yarp/os/Bottle.h>
#include <yarp/os/RpcClient.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/Network.h>
using namespace yarp::os;
STATIC DECLARATIONS _________________________

MySim::MySim(string robotName, string ownPortName) {
	eyeLx =  0.034;
	eyeLy =  0.93825;
	eyeLz =  0.039;
	eyeRx = -0.034;
	eyeRy =  0.93825;
	eyeRz =  0.039;	
	eyesTilt = 0.0,
	eyesVersion = 0.0,
	bodyCenterY = 0.6,
	//TODO resource finder!
	EYE_F = 0.25734;	
	ZF_WIDTH = 0.32;
	ZF_HEIGHT = 0.24;	
	this->robotName = robotName;
#include <yarp/sig/Image.h>
#include <yarp/sig/Vector.h>
	using namespace yarp::sig;

#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IPositionControl.h>
#include <yarp/dev/ControlBoardInterfaces.h>
	using namespace yarp::dev;

#include <iostream>
#include <string>
	using namespace std;

#include <Windows.h>

#include "MySim.h"


//_____________________________________________________________________________
//______ CONSTRUCTOR, DESTRUCTOR,
	this->ownPort = ownPortName;
	worldPort = "/world";
	headPort = "/head";
	leftCamPort = "/cam/left";
	rightCamPort = "/cam/right";
	rpcWorld = NULL;	
	headDriver = NULL;
	eyesPosControl = NULL;
	leftEye = NULL;	
	rightEye = NULL;
	Network yarp;	
	verbose = true;
	this->output = &cout;
	*(this->output) << "icub is starting" << endl;
};

MySim::~MySim() {
	if (rpcWorld!=NULL) {
		rpcWorld->close();
		this->disconnect(worldPort);				
		delete rpcWorld;	
		rpcWorld = NULL;
	}	
	if (headDriver!=NULL) {
		headDriver->close();
		delete headDriver;		
		headDriver = NULL;
	}
	eyesPosControl = NULL;
	if (leftEye!=NULL) {
		disconnectLeftEye();
	}
	if (rightEye!=NULL) {
		disconnectRightEye();
	}
}


//_____________________________________________________________________________
//______ PRIVATE DECLARATIONS _________________________________________________


//------- NETWORK stuff ------

string MySim::getPortFullName(string port, bool own) {
	string s = own ? ownPort : robotName;
	s += port;	
	return s;
}

bool MySim::connect(string port, bool write = true) {
	string s1 = getPortFullName(port, write);
	string s2 = getPortFullName(port, !write);	
	return Network::connect(s1.c_str(), s2.c_str(), (const char*)0, !verbose);	
}

bool MySim::disconnect(string port) {
	string s1 = getPortFullName(port, true);
	string s2 = getPortFullName(port, false);		
	return Network::disconnect(s1.c_str(), s2.c_str()) || Network::disconnect(s2.c_str(), s1.c_str());
}

RpcClient* MySim::getRpcWorld() {
	if (rpcWorld==NULL) {
		rpcWorld = new RpcClient();				
		rpcWorld->open(getPortFullName(worldPort, true).c_str());
		this->connect(worldPort);		
	}	
	return rpcWorld;
}

//------ controlling HEAD and EYES -------

PolyDriver *MySim::getHeadDriver() {
	if (headDriver==NULL) {
		Property options;
		options.put("device", "remote_controlboard");		
		options.put("local", getPortFullName(headPort, true).c_str());
		options.put("remote", getPortFullName(headPort, false).c_str());		
		headDriver = new PolyDriver(options);				
		if (!headDriver->isValid()) {
			printf("MYSIM ERROR: Cannot connect to robot head\n");						
		}						
	}
	return headDriver;
}

IPositionControl *MySim::getEyesPositionControl() {
	if (eyesPosControl==NULL) {
		PolyDriver *headDriver = getHeadDriver();
		headDriver->view(eyesPosControl);
		if (eyesPosControl==NULL) {
			printf("MYSIM ERROR: Cannot get interface to robot head\n");						
		}	
	}
	return eyesPosControl;	
}

BufferedPort<ImageOf<PixelRgb>> *MySim::getLeftEye() {
	if (leftEye == NULL) {
		leftEye = new BufferedPort<ImageOf<PixelRgb>>();				
		leftEye->open(getPortFullName(leftCamPort, true).c_str());	
		this->connect(leftCamPort, false);		
	}
	return leftEye;	
}

BufferedPort<ImageOf<PixelRgb>> *MySim::getRightEye() {
	if (rightEye == NULL) {
		rightEye = new BufferedPort<ImageOf<PixelRgb>>();				
		rightEye->open(getPortFullName(rightCamPort, true).c_str());	
		this->connect(rightCamPort, false);		
	}
	return rightEye;	
}

//------ OBJECTS SCALE and POSITION ------

/// objects that are put randomly by simulator will be smaller according to this scale
double MySim::myScale(double minz, double maxz, double sminz, double smaxz, double z) {
	// y = a/(x-c)
	// y1 = sminz, y2 = smaxz, x1 = minz, x2 = maxz
	// yx = y1*x1 + y2*x2
	// y_c = (y1 + y2)	
	double yx = (minz*sminz + maxz*smaxz) / 2.0;	
	double y_c = (sminz + smaxz) / 2.0;
	double c = (minz*sminz - yx) / (sminz - y_c); 
	double a = yx - y_c*c;
	return (a / (z - c));	
}

void MySim::putInFOV(double &x, double &y, double &z) {
	double alfa = -(MU::PI/180.0)*eyesTilt;	  //x-axis of eye is oriented opposite to world x-axis
	double beta = -(MU::PI/180.0)*eyesVersion;  //the same for y-axis
	
	double t[3] = { (eyeLx+eyeRx)/2, eyeLy, eyeLz};	 //alebo {eyeLx, eyeLy, eyeLz}
	double a[3] = {x, y, z};		
	double n[] = {0.0, 0.0, 0.0};	
	MatrixOp::subtractM31(a, t, a);	//translation to left eye / or in the middle of eyes
	MatrixOp::rotX(alfa, a, n);	    //rot X
	MatrixOp::rotY(beta, n, a);		//rot Y
	MatrixOp::addM31(a, t, a);		//translation back
	
	if (a[1]>0.0) {
		x = a[0];
		y = a[1];
		z = a[2];
	} else {
		//posunutie naspat po priamke od objektu k ociam
		double eyeX = (eyeLx+eyeRx)/2.0;
		double eyeY = (eyeLy+eyeRy)/2.0;
		double eyeZ = (eyeLz+eyeRz)/2.0;
		double t = -eyeY / (a[1] - eyeY);		
		x = eyeX + (a[0] - eyeX)*t;
		y = eyeY + (a[1] - eyeY)*t;
		z = eyeZ + (a[2] - eyeZ)*t;		
	}
}



//_____________________________________________________________________________
//______ PUBLIC DECLARATIONS __________________________________________________



/// deletes all objects from scene
Bottle MySim::deleteAllObjects() {						
	Bottle response;		 		
	getRpcWorld()->write(BottleWrapper::prepareBottle("sss","world","del","all"), response);	
//	if (verbose) cout << "deleting all objects: " << response.toString() << endl;	
	return response;
}

// put an object that fills the whole field of view of left eye
Bottle MySim::putObjectBeforeEye_left() {
	Bottle b = BottleWrapper::box(ZF_WIDTH, ZF_HEIGHT, 0.1, eyeLx, eyeLy, eyeLz+EYE_F+0.05);
	Bottle response;
	getRpcWorld()->write(b, response);
//	if (verbose) cout << "putting object before eye " << response.toString() << endl;
	return response;
}

Bottle MySim::putRandomObjectWRotation(bool inFOV, double z) {		
	//random size	
	double minz = 1; 
	//double maxz = (eyesTilt < -10.0) ? 2.5 : 9.0;	
	double maxz = 9.0;

	double scale_at_minz = 1.0;
	double scale_at_maxz = 0.6;			
	double pz = MU::fRand(minz, maxz);  //position of z is necessary for determining size
	if (z>0) {
		pz = z;
	}	
	//pz = 2;
	double specialZ = eyeLz+EYE_F;  //at this distance will object of size ZF_WIDTH, ZF_HEIGHT fill whole FOV	
	double koef = 1.0;
	if (!MU::theSame(pz, specialZ)) {
		koef = (pz/specialZ) * myScale(minz, maxz, scale_at_minz, scale_at_maxz, pz);	
	}			
	double sz = MU::fRand(ZF_WIDTH/9.0, ZF_WIDTH/3.0)*koef; 
	double sx = MU::fRand(ZF_WIDTH/9.0, ZF_WIDTH/3.0)*koef;
	double sy = MU::fRand(ZF_WIDTH/9.0, ZF_WIDTH/3.0)*koef;

	sz = 0.25;
	sx = 0.25;
	sy = 0.25;
		
	//random position	
	double mostx = ((eyeLx+eyeRx)/2.0) + (ZF_WIDTH/2.0) * (pz / specialZ); //the most left position so as object is still in FOV of left eye
	double mosty = eyeLy + (ZF_HEIGHT/2.0)* (pz / specialZ); //the most top 
	double padding = max(max(sx, sy),sz);
	double px = MU::fRand(-(mostx - padding), mostx - padding);
	double py = MU::fRand(max(padding,-(mosty - padding)), mosty - padding);	

	//put in FOV if necessary	
	if (inFOV && (!MU::theSame(eyesTilt,0.0) || !MU::theSame(eyesVersion,0.0))) {		
		double oldPz = pz;
		putInFOV(px, py, pz);
		//adjust size?
		if (!MU::theSame(oldPz, pz)) {
			koef = (pz/specialZ) * myScale(minz, maxz, scale_at_minz, scale_at_maxz, pz);	
			double sz = MU::fRand(ZF_WIDTH/9.0, ZF_WIDTH/3.0)*koef; 
			double sx = MU::fRand(ZF_WIDTH/9.0, ZF_WIDTH/3.0)*koef;
			double sy = MU::fRand(ZF_WIDTH/9.0, ZF_WIDTH/3.0)*koef;
		}		
	}
				
	//random rotation
	double rx = MU::fRand(0.0, 360.0);
	double ry = MU::fRand(0.0, 360.0);
	double rz = MU::fRand(0.0, 360.0);
	//generate random object
	Bottle obj, rot;
	double d = MU::fRand(0.0,3.0);		
	
	////////////////
	//d = 2.5;	
	
	*(this->output) << px << " " << py << " " << pz << " ";
	if ((d>1.0) && (false)) { //ONLY SPHERE NOW
		if (d>2.0) {	
			obj = BottleWrapper::box(sx, sy, sz, px, py, pz);
			//cout << "box" << sx << " " << sy << " "<< sz << " "<< px << " "<< py << " "<< pz << endl;
			rot = BottleWrapper::r_box(1,  rx, ry, rz);
			*(this->output) << "box " << sx << " " << sy << " " << sz << " ";
		} else {	
			obj = BottleWrapper::cyl(sz, sy, px, py, pz);
			//cout << "cyl" << sz << " " << sy << " "<< " "<< px << " "<< py << " "<< pz << endl;
			rot = BottleWrapper::r_cyl(1,  rx, ry, rz);
			*(this->output) << "cyl " << sz << " " << sy;
		}		
	} else {	
		obj = BottleWrapper::sph(sy, px, py, pz);			
		//cout << "sph" << sz << " " << sy << " "<< " "<< px << " "<< py << " "<< pz << endl;
		//rot = BottleWrapper::r_sph(1,  rx, ry, rz);
		*(this->output) << "sph " << sy;
	}	
	*(this->output) << endl;
	Bottle response;
	getRpcWorld()->write(obj, response);				
	getRpcWorld()->write(rot, response);			
	return response;
}

void MySim::turnEyes(double tilt, double version, bool waitForDone) {
	IPositionControl *pos = getEyesPositionControl();
	int jnts = 0;
	pos->getAxes(&jnts);
	Vector positions;	
	positions.resize(jnts);
	for (int i=0; i<jnts; i++) {
		positions[i] = 0;
	}
	positions[3] = tilt;
	positions[4] = version;
	eyesTilt = tilt;
	eyesVersion = version;
	pos->positionMove(positions.data());	
	if (waitForDone) {
		bool done = false;		
		while (!done) {		
			pos->checkMotionDone(&done);
			Sleep(10);
		}
	}	
}

ImageOf<PixelRgb>* MySim::getLeftEyeImage() {	
	return getLeftEye()->read();	
}

ImageOf<PixelRgb>* MySim::getRightEyeImage() {	
	return getRightEye()->read();	
}

void MySim::disconnectLeftEye(){
	if (leftEye != NULL) {
		leftEye->close();
		this->disconnect(leftCamPort);		
		delete leftEye;
		leftEye = NULL;
	}
}

void MySim::disconnectRightEye(){
	if (rightEye != NULL) {
		rightEye->close();
		this->disconnect(rightCamPort);		
		delete rightEye;
		rightEye = NULL;
	}
}

void MySim::setOutput(ostream *output) {
	this->output = output;
}
