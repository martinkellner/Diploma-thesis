#include <iostream>
#include <string>
using namespace std;

#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/ControlBoardInterfaces.h>
#include <yarp/dev/IPositionControl.h>
#include <yarp/dev/IEncoders.h>
using namespace yarp::dev;

#include <yarp/sig/Vector.h>
using namespace yarp::sig;

#include <yarp/os/Network.h>
using namespace yarp::os;

#include "My_ICub.h"
//_____________________________________________________________________________
//______ CONSTRUCTOR, DESTRUCTOR, STATIC DECLARATIONS _________________________

My_ICub::My_ICub(string robot_name, string own_port_name) {
    //Names
    this->robot_name = robot_name;
    this->own_port_name = own_port_name;
    //Ports
    this->head_port = '/head';
    this->left_cam_port = '/cam/left';
    this->right_cam_port = '/cam/right';
    //Drivers
    this->head_driver = NULL;
};

My_ICub::~My_ICub() {
    if ( !(this->head_driver == NULL) ) {
        this->head_driver->close();
        delete this->head_driver;
        this->head_driver = NULL;
    };
};

string My_ICub::getFullPortName(string port, bool own) {
    if (own) {
        return this->own_port_name + port;
    };
    return this->robot_name + port;
};

bool My_ICub::connectToPort(string port, bool write) {
    string port_1 = getFullPortName(port, write);
    string port_2 = getFullPortName(port, !write);
    return Network::connect(port_1.c_str(), port_2.c_str(), (const char*)0, false);
};

PolyDriver *My_ICub::getRobotHeadDriver() {
    if (this->head_driver == NULL) {
        Property options;
        options.put("device", "remote_controlboard");
        options.put("local", this->getFullPortName(this->head_port, true).c_str());
        options.put("remote", this->getFullPortName(this->head_port, false).c_str());

        this->head_driver = new PolyDriver(options);
        if (!(this->head_driver->isValid() == false)) {
            printf("Device not available.  Here are the known devices:\n");
            printf("%s", Drivers::factory().toString().c_str());
        };
    };
    return this->head_driver;
};

void My_ICub::headMovement() {
    IPositionControl *pos;
    IEncoders *encs;
    bool correct;
    PolyDriver robot_head_driver = this.getRobotHeadDriver();

    if (!(robot_head_driver == NULL)) {
        correct = robot_head_driver.view(pos);
        correct = correct && robot_head_driver.view(encs);
    } else {
        printf("Problems acquiring interfaces\n");

    };
    // TODO: Change icub's head position by given parameters
};