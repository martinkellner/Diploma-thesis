#pragma once
#ifndef MY_ICUB_H
#define MY_ICUB_H

#include <string.h>
#include <fstream>
#include <vector>
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
        enum Way {RANDOM};
        void collectingData(string path, int number, Way way);
        void setRightArmPosition(Vector &position, bool wait);
        int getRightArmJoints();
        void closeDataFile();
        void prepareDatasetFile(string path, double const angle);


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
            right_arm_port,
            last_head_position,
            last_arm_position;

        ofstream datafile;
        int getDataFile(string path);
        void setHeadPosition(Vector position, bool wait);
        void takeAndSaveImages(string path, string name);
        void writeToDataFile(string str);
        double randomAngle(double minAngle, double maxAngle);
        void randomCollecting(string path, int startFrom, int total, int imagesCount, bool armSeen);
        void randomRightArmPosition(Vector position, bool wait);
        void randomHeadPosition(Vector position, bool wait);

        ImageOf<PixelRgb> *getRobotRightEyeImage();
        ImageOf<PixelRgb> *getRobotLeftEyeImage();

        PolyDriver* head_driver;
        PolyDriver* right_arm_driver;

        IPositionControl* head_controller;
        IPositionControl* right_arm_controller;

        BufferedPort<ImageOf<PixelRgb>> *left_cam;
        BufferedPort<ImageOf<PixelRgb>> *right_cam;

        IPositionControl *getHeadController();
        IPositionControl *getRightArmController();

        PolyDriver *getRobotHeadDriver();
        PolyDriver *getRobotRightArmDriver();

        BufferedPort<ImageOf<PixelRgb>> *getRobotRightEyeDriver();
        BufferedPort<ImageOf<PixelRgb>> *getRobotLeftEyeDriver();
};

#endif