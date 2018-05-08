#pragma once
#ifndef MY_ICUB_H
#define MY_ICUB_H

#include <string.h>
using namespace std;

#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IPositionControl.h>
using namespace yarp::dev;

class My_ICub {
    
    public:
        My_ICub(string robot_name="/icubSim", string own_port_name="/mysim");
        ~My_ICub();

        string getFullPortName(string port, bool own);
        void headMovement(double angle, int axis, bool wait);
        void rightArmMovement(double angle, int axis, bool wait);

    protected:
        string
            robot_name,
            own_port_name;
        bool connectToPort(string port, bool write);

    private:
        string
            head_port,
            left_cam_port,
            right_cam_port,
            right_arm_port;

        PolyDriver* head_driver;
        PolyDriver* right_arm_driver;
        IPositionControl* head_controller;
        IPositionControl* right_arm_controller;

        IPositionControl* getHeadController();
        IPositionControl* getRightArmController();
        PolyDriver *getRobotHeadDriver();
        PolyDriver *getRobotRightArmDriver();
};

#endif
