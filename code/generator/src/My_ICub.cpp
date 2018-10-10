#include <iostream>
#include <string>
#include <cmath>
#include <stdlib.h>

using namespace std;

#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IPositionControl.h>
#include <yarp/dev/ControlBoardInterfaces.h>
using namespace yarp::dev;

#include <yarp/sig/Vector.h>
#include <yarp/sig/Image.h>
#include <yarp/sig/ImageFile.h>
using namespace yarp::sig;

#include <yarp/os/Network.h>
#include <yarp/os/BufferedPort.h>
using namespace yarp::os;

#include <unistd.h>
#include <My_ICub.h>

#include "My_ICub.h"
//_____________________________________________________________________________
//______ CONSTRUCTOR, DESTRUCTOR, STATIC DECLARATIONS _________________________

My_ICub::My_ICub(string robot_name, string own_port_name) {
    //Names
    this->robot_name = robot_name;
    this->own_port_name = own_port_name;
    //Ports
    head_port           = "/head";
    left_cam_port       = "/cam/left";
    right_cam_port      = "/cam/right";
    right_arm_port      = "/right_arm";
    last_arm_position;
    last_head_position;
    //Drivers
    head_driver             = NULL;
    right_arm_driver        = NULL;
    head_controller         = NULL;
    right_arm_controller    = NULL;
    left_cam                = NULL;
    right_cam               = NULL;
    //Others
    datafile;
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

PolyDriver *My_ICub::getRobotRightArmDriver() {
    if (right_arm_driver==NULL) {
        Property options;
        options.put("device", "remote_controlboard");
        options.put("local", getFullPortName(right_arm_port, true).c_str());
        options.put("remote", getFullPortName(right_arm_port, false).c_str());

        right_arm_driver = new PolyDriver(options);
        if (!(right_arm_driver->isValid())) {
            printf("Device not available.  Here are the known devices:\n");
            printf("%s", Drivers::factory().toString().c_str());
        };
    };
    return right_arm_driver;
};

IPositionControl *My_ICub::getHeadController() {
    if (head_controller==NULL) {
        PolyDriver *head_driver = getRobotHeadDriver();
        head_driver->view(head_controller);
        if (head_controller==NULL) {
            printf("Problem acquiring interfaces\n");
        };
    };
    return head_controller;
};

IPositionControl *My_ICub::getRightArmController() {
    if (right_arm_controller==NULL) {
        PolyDriver *right_arm_driver = getRobotRightArmDriver();
        right_arm_driver->view(right_arm_controller);
        if (right_arm_controller==NULL) {
            printf("Problem acquiring interfaces\n");
        };
    };
    return right_arm_controller;
};

void My_ICub::collectingData(string path, int number, Way way) {
    if(!getDataFile(path)) {
        return;
    };

    if (way == RANDOM) {
        randomCollecting(path, 0, number, 10, false); //TODO: set the argument according to datafile's last line!
    };

    /*int itr = 0;
    int axs;

    while (true) {
        axs = itr%3;
        cout << " \t" << itr + 1 << ". head movement, joints values -> " << "pos[0]: " << position[0] << " pos[1]: " << position[1] << " pos[2]: " << position[2] << endl;
        if ((position[0] + angle < 22) || (position[1] + angle < 20) || (position[2] + angle < 45)) {
            if (axs == 0 && position[0] + angle < 22) {
                position[0] = position[0] + angle;
            } else if (axs == 1 && position[1] + angle < 20) {
                position[1] = position[1] + angle;
            } else if (axs == 2 && position[2] + angle < 45) {
                position[2] = position[2] + angle;
            };
            setHeadPosition(position, true);
            last_head_position = to_string(position[0]) + " " + to_string(position[1]) + " " + to_string(position[2]);
            takeAndSaveImages(number, itr, path);
            saveAngles(number, itr, path);
        }
        else {
            position[0] = 22;
            position[1] = 20;
            position[2] = 45;
            setHeadPosition(position, true);
            cout << " \t" << itr + 1 << ". head movement, joints values -> " << "pos[0]: " << position[0] << " pos[1]: " << position[1] << " pos[2]: " << position[2] << endl;
            takeAndSaveImages(number, itr, path);
            break;
        };
        itr = itr + 1;
    };*/
};

void My_ICub::setHeadPosition(Vector position, bool wait) {
    if (head_controller==NULL) {
        getHeadController();
    };
    head_controller->positionMove(position.data());
    if (wait) {
        bool is_done = false;
        while(!is_done) {
            head_controller->checkMotionDone(&is_done);
            usleep(10);
        };
    };
};

void My_ICub::setRightArmPosition(Vector &position, bool wait) {
    if (right_arm_controller==NULL) {
        getRightArmController();
    };
    right_arm_controller->positionMove(position.data());
    if (wait) {
        bool is_done = false;
        while (!is_done) {
            right_arm_controller->checkMotionDone(&is_done);
            usleep(10);
        };
    };
};

BufferedPort<ImageOf<PixelRgb>> *My_ICub::getRobotRightEyeDriver() {
    if (right_cam == NULL) {
        right_cam = new BufferedPort<ImageOf<PixelRgb>>();
        right_cam->open(getFullPortName(right_cam_port, true).c_str());
        this->connectToPort(right_cam_port, false);
    };
    return right_cam;
};

BufferedPort<ImageOf<PixelRgb>> *My_ICub::getRobotLeftEyeDriver() {
    if (left_cam == NULL) {
        left_cam = new BufferedPort<ImageOf<PixelRgb>>();
        left_cam->open(getFullPortName(left_cam_port, true).c_str());
        this->connectToPort(left_cam_port, false);
    };

    return left_cam;
};

ImageOf<PixelRgb> *My_ICub::getRobotRightEyeImage() {
    return getRobotRightEyeDriver()->read();
};

ImageOf<PixelRgb> *My_ICub::getRobotLeftEyeImage() {
    return getRobotLeftEyeDriver()->read();
};

int My_ICub::getRightArmJoints() {
    int joints = 0;
    getRightArmController()->getAxes(&joints);
    return joints;
};

void My_ICub::takeAndSaveImages(string path, string name) {
    ImageOf<PixelRgb> *leftImg  = getRobotLeftEyeImage();
    if (leftImg!=NULL) {
        yarp::sig::file::write(*leftImg, path + name + "_L.ppm");
        datafile << name + "_L.ppm ";
    };
    ImageOf<PixelRgb> *rightImg = getRobotRightEyeImage();
    if (rightImg!=NULL) {
        yarp::sig::file::write(*rightImg, path + name + "_R.ppm");
        datafile << name + "_R.ppm";
    };
};

int My_ICub::getDataFile(string path) {
    if (!datafile.is_open()) {
        datafile.open(path + "dataset.txt");
        if (!datafile.is_open()) {
            return 0;
        };
    };
    cout << "Datafile successufly opened!" << endl;
    return 1;
};


void My_ICub::closeDataFile() {
    datafile.close();
};

double My_ICub::randomAngle(double minAngle, double maxAngle) {
    double rang = (double) rand() / RAND_MAX;
    return minAngle + rang * (maxAngle - minAngle);
};

void My_ICub::randomCollecting(string path, int startFrom, int total, int imagesCount, bool armSeen) {

    getRightArmController();  getHeadController();
    int armJnts, headJnts;
    right_arm_controller->getAxes(&armJnts); head_controller->getAxes(&headJnts);
    Vector armPos, headPos;
    armPos.resize(armJnts); headPos.resize(headJnts);

    for (int i = startFrom; i < total; i++) {
        randomHeadPosition(headPos, true);
        writeToDataFile(last_head_position + '\n');
        for (int j = 0; j < imagesCount; j++) {
            randomRightArmPosition(armPos, true);
            writeToDataFile('\t' + last_arm_position + ' ');
            takeAndSaveImages(path, to_string(i) + "_" + to_string(j));
            writeToDataFile("\n");
        };
        // TODO: Implement collecting images that include a part of the right arm! (VARIABLE: armSeen)
    };
};

void My_ICub::randomHeadPosition(Vector position, bool wait) {
    // Range of head's joints
    // ----------------------
    // Neck pitch <-30, 22>
    // Neck roll  <-20, 20>
    // Neck yaw   <-45, 45>
    // ----------------------

    position[0] = randomAngle(-30, 22);
    position[1] = randomAngle(-20, 20);
    position[2] = randomAngle(-45, 45);
    //setHeadPosition(position, wait);
    last_head_position = to_string(position[0]) + " " + to_string(position[1]) + " " + to_string(position[2]);
};

void My_ICub::randomRightArmPosition(Vector position, bool wait) {
    // Ranges of arm's joint
    // ----------------------
    // Shoulder pitch <-95.5, 8>
    // Shoulder roll  <0, 160>
    // Shoulder yaw   <-32, 80>
    // Elbow          <15, 106>

    position[0] = randomAngle(-95.5, 8);
    position[1] = randomAngle(0, 160);
    position[2] = randomAngle(-32, 80);
    position[3] = randomAngle(15, 106);
    //setRightArmPosition(position, wait);
    last_arm_position = to_string(position[0]) + " " + to_string(position[1]) + " " + to_string(position[2]) + " " + to_string(position[3]);
};

void My_ICub::writeToDataFile(string str) {
    datafile << str;
}

