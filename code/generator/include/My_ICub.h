#pragma once
#ifndef MY_ICUB_H
#define MY_ICUB_H

#include <string.h>
using namespace std;

#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IPositionControl.h>
using namespace yarp::dev;

#include <yarp/sig/Image.h>
#include <yarp/sig/Vector.h>
using namespace yarp::sig;

#include <yarp/os/BufferedPort.h>
#include <yarp/os/RpcClient.h>
using namespace yarp::os;

class My_ICub {
    
    public:
        My_ICub(string robot_name="/icubSim", string own_port_name="/mysim");
        ~My_ICub();

        string getFullPortName(string port, bool own);
        void headMovement(double angle, string path);
        void rightArmMovement(Vector &position, bool wait);
        int getRightArmJoints();

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

        void setHeadPosition(Vector position, bool wait);
        void takeAndSaveImages(int number, string path);
        ImageOf<PixelRgb> *getRobotRightEyeImage();
        ImageOf<PixelRgb> *getRobotLeftEyeImage();


    PolyDriver* head_driver;
        PolyDriver* right_arm_driver;

        IPositionControl* head_controller;
        IPositionControl* right_arm_controller;

        BufferedPort<ImageOf<PixelRgb>> *left_cam;
        BufferedPort<ImageOf<PixelRgb>> *right_cam;

        IPositionControl* getHeadController();
        IPositionControl* getRightArmController();

        PolyDriver *getRobotHeadDriver();
        PolyDriver *getRobotRightArmDriver();

        BufferedPort<ImageOf<PixelRgb>> *getRobotRightEyeDriver();
        BufferedPort<ImageOf<PixelRgb>> *getRobotLeftEyeDriver();
};

#endif