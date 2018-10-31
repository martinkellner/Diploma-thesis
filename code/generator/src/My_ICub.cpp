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
    right_arm_driver        = NULL;
    left_arm_driver         = NULL;
    head_controller         = NULL;
    right_arm_controller    = NULL;
    left_arm_controller     = NULL;
    left_cam                = NULL;
    right_cam               = NULL;
    gaze_driver             = NULL;
    iGaze                   = NULL;
    world_client            = NULL;
    //Others
    datafile;

    this->setRightArmVector();
    this->setArmToDefaultPosition(RIGHT);
    this->setArmToDefaultPosition(LEFT);
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
            printf("Problem acquiring interfaces\n");
        };
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

void My_ICub::setArmPosition(Hand hand, bool wait) {
    // TODO: discover constrains for arms' angles

    if (hand == RIGHT) {
        getArmController(RIGHT);
        right_arm_controller->positionMove(right_arm_vector.data());
        if (wait) {
            bool is_done = false;
            while (!is_done) {
                right_arm_controller->checkMotionDone(&is_done);
                usleep(10);
            };
        };
    } else if (hand == LEFT) {
        getArmController(LEFT);
        left_arm_controller->positionMove(right_arm_vector.data());
        if (wait) {
            bool is_done = false;
            while (!is_done) {
                right_arm_controller->checkMotionDone(&is_done);
                usleep(10);
            };
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
    int joints;
    right_arm_controller->getAxes(&joints);
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

    getArmController(RIGHT);
    getHeadController();
    getWorldRpcClient();

    for (int i = startFrom; i < total; i++) {
        writeToDataFile(to_string(i) + ". ");
        deleteAllObject();                              // delete all objects that was created
        randomRightArmPosition(true);                   // random hand position
        saveRightHandAngles();                          // save current right hand angles
        Vector handWPos = getRightPalmWorldPosition();  // receive hand position in world reference frame
        cout << handWPos.toString() << endl;
        cout << "objects deleted" << endl;
        setArmToDefaultPosition(RIGHT);                 // move right hand to default position
        cout << "set to defalut" << endl;
        putObjectToPosition(handWPos);                  // put an object to previous right hand pose
        lookAtPosition(handWPos);                       // look at the object
        saveHandAngles();                               // save current head angles
        saveObjectPosition(handWPos);                   // save current object's pose
        takeAndSaveImages(path, to_string(i));          // save images from icub cameras
        writeToDataFile("\n");
    };
};

void My_ICub::randomRightArmPosition(bool wait) {
    // Ranges of arm's joint
    // ----------------------
    // Shoulder pitch <-95.5, 8>
    // Shoulder roll  <0, 160>
    // Shoulder yaw   <-32, 80>
    // Elbow          <15, 106>

    right_arm_vector[0] = randomAngle(-95.5, 8);
    right_arm_vector[1] = randomAngle(0, 160);
    right_arm_vector[2] = randomAngle(-32, 80);
    right_arm_vector[3] = randomAngle(15, 106);
    setArmPosition(RIGHT, wait);

};

void My_ICub::writeToDataFile(string str) {
    datafile << str;
}

void My_ICub::lookAtPosition(Vector worldVct) {
    // The robot will look at new fixation point using iGaze interface
    getRobotGazeInteface();
    Vector rootVct(3);
    // Rototransfomation from world reference frame to root frame.
    MatrixOperations::rotoTransfWorldRoot(worldVct, rootVct);

    iGaze->lookAtFixationPointSync(rootVct);
    iGaze->waitMotionDone();

}

void My_ICub::getWorldRpcClient() {
    if (world_client==NULL) {
        world_client = new RpcClient();
        world_client->open(getFullPortName(world_port, true));
        connectToPort(world_port, true);
    }
}

void My_ICub::putObjectToPosition(Vector worldVct) {
    getWorldRpcClient();
    // Delete all old objects generated by previous steps of collecting data
    deleteAllObject();
    Bottle rqs, res;

    rqs = WorldYaprRpc::createBOX(worldVct);
    world_client->write(rqs, res);
}

void My_ICub::deleteAllObject() {
    getWorldRpcClient();
    Bottle rqs = WorldYaprRpc::deleteAllObjects();
    Bottle req;
    world_client->write(rqs, req);
}

Vector My_ICub::getRightPalmWorldPosition() {
    getWorldRpcClient();
    Bottle rqs, res;

    rqs = WorldYaprRpc::getRightHandWorldPosition();
    world_client->write(rqs, res);

    Vector rHandWPos(4);
    rHandWPos[0] = res.get(0).asFloat32();
    rHandWPos[1] = res.get(1).asFloat32();
    rHandWPos[2] = res.get(2).asFloat32();
    rHandWPos[3] = 1;
    return rHandWPos;
}

void My_ICub::setArmToDefaultPosition(Hand hand) {
    // set right or left arm's angles so the hand could not be visible by iCub.
    right_arm_vector[0] = right_arm_vector[1] = right_arm_vector[2] = 10; right_arm_vector[3] = 20;

    if (hand == RIGHT) {
        getArmController(RIGHT);
        setArmPosition(RIGHT, true);
    } else if (hand == LEFT) {
        getArmController(LEFT);
        setArmPosition(LEFT, true);
    };
}

void My_ICub::test() {
    getHeadController();
    getArmController(RIGHT);

}

void My_ICub::setRightArmVector() {
    getArmController(RIGHT);
    int jnts = getRightArmJoints();
    right_arm_vector.resize(jnts);
    double refs;

    for (int i=0; i < jnts; i++) {
        right_arm_controller->getTargetPosition(i, &refs);
        right_arm_vector[i] = refs;
    }
}

void My_ICub::saveHandAngles() {
    getHeadController();
    double refs;
    for (int i = 0; i < 3; ++i) {
        head_controller->getTargetPosition(i, &refs);
        datafile << refs << " ";
    }
}

void My_ICub::saveRightHandAngles() {
    for (int i = 0; i < 4; ++i) {
        datafile << right_arm_vector[i] << " ";
    }
}

void My_ICub::saveObjectPosition(Vector obj) {
    for (int i=0; i<3; i++) {
        datafile << obj[i] << " ";
    }
}
