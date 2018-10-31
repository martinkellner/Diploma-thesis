#pragma once
#ifndef MY_ICUB_H
#define MY_ICUB_H

#include <string.h>
#include <fstream>
#include <vector>
using namespace std;

#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/GazeControl.h>
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
        enum Hand {RIGHT, LEFT};
        void collectingData(string path, int number, Way way);

        int getRightArmJoints();
        void closeDataFile();
        void prepareDatasetFile(string path, double const angle);
        void lookAtPosition(Vector worldVct);

        void test();

protected:
        string
            robot_name,
            own_port_name;
        bool connectToPort(string port, bool write);

    private:
        string
            world_port,
            head_port,
            left_cam_port,
            right_cam_port,
            right_arm_port,
            left_arm_port,
            last_head_position,
            last_arm_position;

        ofstream datafile;
        int getDataFile(string path);
        //void setHeadPosition(Vector position, bool wait);
        void takeAndSaveImages(string path, string name);
        void writeToDataFile(string str);
        double randomAngle(double minAngle, double maxAngle);
        void randomCollecting(string path, int startFrom, int total, int imagesCount, bool armSeen);
        void randomRightArmPosition(bool wait);
        void generateObjectOnPosition(Vector position);

        //void randomHeadPosition(Vector position, bool wait);

        ImageOf<PixelRgb> *getRobotRightEyeImage();
        ImageOf<PixelRgb> *getRobotLeftEyeImage();

        PolyDriver *head_driver;
        PolyDriver *right_arm_driver;
        PolyDriver *left_arm_driver;
        RpcClient  *world_client;

        IPositionControl *head_controller;
        IPositionControl *right_arm_controller;
        IPositionControl *left_arm_controller;

        BufferedPort<ImageOf<PixelRgb>> *left_cam;
        BufferedPort<ImageOf<PixelRgb>> *right_cam;

        void getHeadController();
        void getArmController(Hand hand);

        PolyDriver *getRobotHeadDriver();
        PolyDriver *getRobotArmDriver(Hand hand);

        BufferedPort<ImageOf<PixelRgb>> *getRobotRightEyeDriver();
        BufferedPort<ImageOf<PixelRgb>> *getRobotLeftEyeDriver();

        PolyDriver *gaze_driver;
        IGazeControl *iGaze;

        void getRobotGazeInteface();
        void getWorldRpcClient();
        void putObjectToPosition(Vector worldVct);
        Vector getRightPalmWorldPosition();
        void deleteAllObject();
        void setArmToDefaultPosition(Hand hand);
        void setRightArmVector();

        Vector right_arm_vector;
        void setArmPosition(Hand hand, bool wait);
        void saveHandAngles();
        void saveRightHandAngles();
        void saveObjectPosition(Vector objPose);
};

#endif