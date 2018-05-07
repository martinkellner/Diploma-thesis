#pragma once
#ifndef MY_ICUB_H
#define MY_ICUB_H

#include <string.h>
using namespace std;

#include <yarp/dev/PolyDriver.h>
using namespace yarp::dev;


class My_ICub {
    
public:
    My_ICub(string robot_name="/icubSim", string own_port_name="/mysim");
    ~My_ICub();

    string getFullPortName(string port, bool own);
    void headMovement();

   
protected:
    string
        robot_name,
        own_port_name;

    bool connectToPort(string port, bool write=true);
    PolyDriver *getRobotHeadDriver();

private:
    string
        head_port,
        left_cam_port,
        right_cam_port;

    PolyDriver *head_driver;
};

#endif
