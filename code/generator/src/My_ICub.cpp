#include <iostream>
#include <string>
#include <stdio>
using namespace std;

#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IPositionControl.h>
#include <yarp/dev/ControlBoardInterfaces.h>
using namespace yarp::dev;

#include <yarp/sig/Vector.h>
using namespace yarp::sig;

#include <yarp/os/Network.h>
using namespace yarp::os;

#include <Windows.h>
#include "My_ICub.h"
//_____________________________________________________________________________
//______ CONSTRUCTOR, DESTRUCTOR, STATIC DECLARATIONS _________________________

My_ICub::My_ICub(string robot_name, string own_port_name) {
    //Names
    this->robot_name = robot_name;
    this->own_port_name = own_port_name;
    //Ports
    head_port = "/head";
    left_cam_port = "/cam/left";
    right_cam_port = "/cam/right";
    //Drivers
    head_driver = NULL;
    head_controller = NULL;
};

My_ICub::~My_ICub() {
    if ( !(head_driver==NULL) ) {
        head_driver->close();
        delete head_driver;
        head_driver = NULL;
    };
};

string My_ICub::getFullPortName(string port, bool own) {
    if (own) {
	printf((own_port_name + port).c_str());
        return own_port_name + port;
    };

    printf((robot_name + port).c_str());
    return robot_name + port;
};

bool My_ICub::connectToPort(string port, bool write) {
    string port_1 = getFullPortName(port, write);
    string port_2 = getFullPortName(port, !write);
    return Network::connect(port_1.c_str(), port_2.c_str(), (const char*)0, false);
};

PolyDriver *My_ICub::getRobotHeadDriver() {
    if (head_driver==NULL) {
        Property options;
        options.put("device", "remote_controlboard");
        options.put("local", getFullPortName(head_port, true).c_str());
        options.put("remote", getFullPortName(head_port, false).c_str());

        head_driver = new PolyDriver(options);
        if (!(head_driver->isValid())) {
            printf("Device not available.  Here are the known devices:\n");
            printf("%s", Drivers::factory().toString().c_str());
        };
    };
    return head_driver;
};

IPositionControl *My_ICub::getHeadController() {
    if (head_controller==NULL) {
        PolyDriver *head_driver = getRobotHeadDriver();
        head_driver->view(head_controller);
        if (head_controller==NULL) {
            printf("Cannot get interface to robot head\n");
        };
    };
    return head_controller;
};


void My_ICub::headMovement(double angle, int axis=0, bool wait=false) {
    IPositionControl *head_controller = getHeadController();
    if (head_controller==NULL) {
        printf("Problems acquiring interfaces\n");
        return;
    };

    int jnts = 0;
    head_controller->getAxes(&jnts);
    Vector position;
    position.resize(jnts);

    for (int i=0; i < jnts; i++) {
        position[i] = 0;
    };
    position[axis] = angle;
    head_controller->positionMove(position.data());
    if (wait) {
        bool is_done = false;
        while(!is_done) {
            head_controller->checkMotionDone(&is_done);
            Sleep(10);
        };
    };
};