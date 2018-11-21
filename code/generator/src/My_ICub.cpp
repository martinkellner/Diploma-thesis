#include <iostream>
#include <string>
#include <cmath>
#include <stdlib.h>

using namespace std;

#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IPositionControl.h>
#include <yarp/dev/ControlBoardInterfaces.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/dev/CartesianControl.h>
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
#include "matrix_operations.h"
#include "yarp_world_rpc.h"
using namespace MatrixOperations;

//_____________________________________________________________________________
//______ CONSTRUCTOR, DESTRUCTOR, STATIC DECLARATIONS _________________________

My_ICub::My_ICub(string robot_name, string own_port_name) {
    //Names
    this->robot_name = robot_name;
    this->own_port_name = own_port_name;
    //Ports
    world_port          = "/world";
    head_port           = "/head";
    left_cam_port       = "/cam/left";
    right_cam_port      = "/cam/right";
    right_arm_port      = "/right_arm";
    left_arm_port       = "/left_arm";
    last_arm_position;
    last_head_position;
    //Drivers
    head_driver             = NULL;
    headLimit               = NULL;
    right_arm_driver        = NULL;
    left_arm_driver         = NULL;
    head_controller         = NULL;
    right_arm_controller    = NULL;
    left_arm_controller     = NULL;
    left_cam                = NULL;
    right_cam               = NULL;
    gaze_driver             = NULL;
    iGaze                   = NULL;
    iCarCtrl                = NULL;
    world_client            = NULL;
    //Others
    datafile;
    head_limit_vector       = NULL;

    //this->setRightArmVector();
    //this->setArmToDefaultPosition(RIGHT);
    //this->setArmToDefaultPosition(LEFT);
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

void My_ICub::getRobotGazeInteface() {
    if (iGaze == NULL) {
        Property option;
        option.put("device", "gazecontrollerclient");
        option.put("remote", "/iKinGazeCtrl");
        option.put("local", getFullPortName("/igaze", true));
        gaze_driver = new PolyDriver(option);
        if (gaze_driver->isValid()) {
            gaze_driver->view(iGaze);
        } else {
            cout << "Received iGaze controller is not valid!" << endl;
            return;
        }
    }
}

PolyDriver *My_ICub::getRobotArmDriver(Hand hand) {
    if (hand == RIGHT) {
        if (right_arm_driver == NULL) {
            Property options;
            options.put("device", "remote_controlboard");
            options.put("local", getFullPortName(right_arm_port, true).c_str());
            options.put("remote", getFullPortName(right_arm_port, false).c_str());
            right_arm_driver = new PolyDriver(options);

            if (!(right_arm_driver->isValid())) {
                printf("Device not available.  Here are the known devices:\n");
                printf("%s", Drivers::factory().toString().c_str());
            }
        }

        return right_arm_driver;

    } else if (hand == LEFT) {
        if (left_arm_driver == NULL) {
            Property options;
            options.put("device", "remote_controlboard");
            options.put("local", getFullPortName(left_arm_port, true).c_str());
            options.put("remote", getFullPortName(left_arm_port, false).c_str());
            left_arm_driver = new PolyDriver(options);

            if (!(left_arm_driver->isValid())) {
                printf("Device not available.  Here are the known devices:\n");
                printf("%s", Drivers::factory().toString().c_str());
            }
        }

        return left_arm_driver;
    }
}

void My_ICub::getHeadController() {
    if (head_controller==NULL) {
        PolyDriver *head_driver = getRobotHeadDriver();
        head_driver->view(head_controller);
        if (head_controller==NULL) {
            printf("Head controller: Problem acquiring interfaces\n");
        };
        head_driver->view(headLimit);
        if (headLimit==NULL) {
            printf("Head Limit controller: Problem acquiring intefaces\n");
        } else {
            int jnts;
            head_controller->getAxes(&jnts);
            head_limit_vector.resize(jnts*2);
            for (int i=0; i<jnts; ++i) {
                double min, max;
                headLimit->getLimits(i, &min, &max);
                head_limit_vector[2*i] = min; head_limit_vector[(2*i)+1] = max;
            }
        }
    };
};

void My_ICub::getArmController(Hand hand) {
    if (hand == RIGHT && right_arm_controller==NULL) {
        right_arm_driver = getRobotArmDriver(RIGHT);
        right_arm_driver->view(right_arm_controller);
        if (right_arm_controller==NULL) {
            fprintf(stderr, "Problem acquiring interfaces\n");
        };
    } else if (hand == LEFT && left_arm_controller==NULL) {
        left_arm_driver = getRobotArmDriver(LEFT);
        left_arm_driver->view(left_arm_controller);
        if (left_arm_controller==NULL) {
            fprintf(stderr, "Problem acquiring interfaces\n");
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

int My_ICub::getRightArmJoints() {
    int joints;
    right_arm_controller->getAxes(&joints);
    printf("JOINTS: %d\n", joints);
    return joints;
};

int My_ICub::getDataFile(string path) {
    if (!datafile.is_open()) {
        path = path[path.size()-1] == '/' ? path : path + '/';
        datafile.open((path + "dataset.txt").c_str(), fstream::out | fstream::app);
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

void My_ICub::getWorldRpcClient() {
    if (world_client==NULL) {
        world_client = new RpcClient();
        world_client->open(getFullPortName(world_port, true));
        connectToPort(world_port, true);
    }
}

void My_ICub::getRightPalmWorldPosition(Vector &vector) {
    getWorldRpcClient();
    Bottle rqs, res;

    rqs = WorldYaprRpc::getRightHandWorldPosition();
    world_client->write(rqs, res);

    vector.resize(4);
    vector[0] = res.get(0).asFloat32();
    vector[1] = res.get(1).asFloat32();
    vector[2] = res.get(2).asFloat32();
    vector[3] = 1;
}

void My_ICub::setRightArmVector() {
    getArmController(RIGHT);
    int jnts = getRightArmJoints();
    right_arm_vector = Vector(jnts);
    double refs;

    for (int i=0; i < jnts; i++) {
        right_arm_controller->getTargetPosition(i, &refs);
        right_arm_vector[i] = refs;
    }
    cout << "RIGHT ARM VECTOR: "; printVector(right_arm_vector);
}

void My_ICub::setHeadAnglesAndMove(Vector pose) {
    getHeadController();
    head_controller->positionMove(pose.data());
    bool is_done = false;
    while (!is_done) {
        head_controller->checkMotionDone(&is_done);
        usleep(0.1);
    }
}

void My_ICub::randomLookWayCollecting(string path, int startFrom, int total) {
    getHeadController();
    getRobotGazeInteface();
    getArmController(RIGHT);
    getDataFile(path);
    setRandomVergenceAngle();
    const double maxError = 0.03;
    double maxAngle = 15;
    double minAngle = 3;
    int direction, steps;
    int doneCorrect = -1;
    for (int i = startFrom; i < total; i++) {
        if (i%10 == 0) setRandomVergenceAngle();
        direction = (doneCorrect == -1 || doneCorrect == 10 || doneCorrect == 9 || doneCorrect == 8) ? (rand() % 9) : doneCorrect;
        if (doneCorrect != -1) i--;
        steps = (rand() % 10) + 4;
        doneCorrect = randomHeadMotions(direction, steps, minAngle, maxAngle, maxError);
    }
}

void My_ICub::getCurrentFixPoint(Vector &vector) {
    getRobotGazeInteface();
    iGaze->getFixationPoint(vector);
}

// Use for testing
void My_ICub::test() {
    getHeadController();
    setRandomVergenceAngle();
    getArmController(RIGHT);
    getRobotGazeInteface();

    Vector res;
    Vector test(4); test[0] = -0.0193448571615013;  test[1] = 0.802296807676238; test[2] =  0.201241938140468;
    MatrixOperations::rotoTransfWorldRoot(test, res);
    printVector(res);
}

int My_ICub::randomHeadMotions(int direction, int steps, double minAng, double maxAngle, double maxError) {
    Property option;
    option.put("device", "cartesiancontrollerclient");
    option.put("remote", "/icubSim/cartesianController/right_arm");
    option.put("local", "/client/right_arm");

    PolyDriver clientCartCtrl(option);
    ICartesianControl *icart = NULL;
    if (clientCartCtrl.isValid()) {
        clientCartCtrl.view(icart);
    }
    icart->setTrajTime(.1);  // given in seconds

    Vector headAngles; getHeadCurrentVector(headAngles);

    double xDir, xDiff, yDir, yDiff;
    double directions[] = {0, 1, 0, -1, 1, 0, -1, 0, 1, 1, -1, -1, -1, 1, 1, -1};
    xDir = directions[(direction*2)];
    yDir = directions[(direction*2)+1];
    Vector fixPoint(3), headVector, armVector, error(3);
    Vector err(3); Vector wHandY, rHandY, xd, od, jointConf;
    int errChck;
    for (int i=0; i<steps; i++) {
        xDiff = randomAngle(minAng, maxAngle)*xDir;
        yDiff = randomAngle(minAng, maxAngle)*yDir;
        headAngles[0] += xDiff; headAngles[2] += yDiff;
        if (!checkHeadAngles(headAngles)) {
            fprintf(stderr, "Return false because angles check!\n");
            return 10; // Break the cycle if a joint angle is out of range!
        }
        setHeadAnglesAndMove(headAngles);
        getCurrentFixPoint(fixPoint);
        icart->askForPosition(fixPoint, xd, od, jointConf);
        getArmJoints(jointConf);
        //setArmJoints(RIGHT, jointConf);

        for (int j = 0; j < 3; ++j) {
            error[j] = fixPoint[j] - xd[j];
        }

        if ((errChck = checkError(error, maxError)) != -1) {
            fprintf(stderr, "Return error code (bigger than -1) if  error is bigger as limit! (%d recommeded direction)\n", errChck);
            return errChck;
        } else {
            datafile << vectorDataToString(headAngles) << vectorDataToString(jointConf) << vectorDataToString(fixPoint) << vectorDataToString(xd) << vectorDataToString(error) << '\n';
            datafile.flush();
        }
        //getRightPalmWorldPosition(wHandY); MatrixOperations::rotoTransfWorldRoot(wHandY, rHandY);
        // print data to the file
        cout << "Head joints: "; printVector(headAngles); cout << "\nArm joints: "; printVector(jointConf); cout << "\nFix point: "; printVector(fixPoint); cout << "\nXD: "; printVector(xd); /*cout << "\nYArm: "; printVector(rHandY);*/ cout << "\nErr: "; printVector(err); cout << endl;
    }
    return -1;
}

void My_ICub::getHeadCurrentVector(Vector &headAngles) {
    getHeadController();
    int jnts; head_controller->getAxes(&jnts);
    double *angs = new double[jnts];
    head_controller->getTargetPositions(angs);

    headAngles.resize(jnts);
    for (int i=0; i<jnts; i++) {
        headAngles[i] = angs[i];
    }
}

void My_ICub::printVector(Vector vec) {
    for (int i=0; i<vec.size(); i++) {
        cout << vec[i] << " ";
    }
    cout << endl;
}

bool My_ICub::checkHeadAngles(Vector headAngles) {
    // check if 0., 2., 5. angle is out of the range
    // TODO: check vertage joint limitsbecause
    return head_limit_vector[0] < headAngles[0] && headAngles[0] < head_limit_vector[1] && head_limit_vector[4] < headAngles[2] && headAngles[2] < head_limit_vector[5];
}

void My_ICub::setRandomVergenceAngle() {
    // random angle from 24 to 44
    int randAng = (rand() % 21) + 24;
    head_controller->positionMove(5, randAng);
    bool is_done = false;
    while (!is_done) {
        head_controller->checkMotionDone(&is_done);
        usleep(10);
    }
}

string My_ICub::vectorDataToString(Vector vector) {
    string res = "";
    for (int i=0; i<vector.size(); i++) {
        res += to_string(vector[i]) + " ";
    }
    return res;
}

int My_ICub::checkError(Vector error, const double maxErr) {
    cout << "Print Error: "; printVector(error);
    if (error[0] < maxErr*-1) return 8;
    if (error[0] > maxErr)    return 9;
    if (error[1] > maxErr && error[2] < maxErr*-1) return 4;
    if (error[1] < maxErr*-1 && error[2] > maxErr) return 5;
    if (error[1] > maxErr && error[2] > maxErr)    return 6;
    if (error[1] < maxErr*-1 && error[2] < maxErr*-1) return 7;
    if (error[1] < maxErr*-1) return 1;
    if (error[1] > maxErr)    return 0;
    if (error[2] > maxErr)    return 3;
    if (error[2] < maxErr*-1) return 2;
    return -1;
}

void My_ICub::getArmJoints(Vector &armJoints) {
    Vector newArmJoints(armJoints.size()-3);
    for (int i=3; i<armJoints.size(); i++) {
        newArmJoints[i-3] = armJoints[i];
    }
    armJoints = newArmJoints;
}

void My_ICub::setArmJoints(My_ICub::Hand hand, Vector joints) {
    getArmController(hand);
    if (hand == RIGHT) {
        for (int i=0; i<joints.size(); i++) {
            right_arm_controller->positionMove(i, joints[i]);
        }
        bool is_done = false;
        while (!is_done) {
            right_arm_controller->checkMotionDone(&is_done);
            usleep(5);
        }
    }
}
