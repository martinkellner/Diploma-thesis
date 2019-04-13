#include <iostream>
#include <string>
#include <cmath>
#include <stdlib.h>
#include <tuple>

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
        datafile.open(path, std::ios_base::app);
        if (!datafile.is_open()) {
            return 0;
        };
    };
    cout << "Given datafile successufly opened!" << endl;
    return 1;
};

void My_ICub::closeDataFile() {
    datafile.close();
};

double My_ICub::randomDoubleValue(double min, double max) {
    double rang = (double) rand() / RAND_MAX;
    return min + (rang * (max - min));
};

int My_ICub::randomIntValue(int min, int max) {
    int rang = rand() / RAND_MAX;
    return min + (rang * (max - min));
};

void My_ICub::getWorldRpcClient() {
    if (world_client==NULL) {
        world_client = new RpcClient();
        world_client->open(getFullPortName(world_port, true));
        connectToPort(world_port, true);
    }
};

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
};

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
};

void My_ICub::setHeadAnglesAndMove(Vector pose) {
    getHeadController();
    head_controller->positionMove(pose.data());
    bool is_done = false;
    while (!is_done) {
        head_controller->checkMotionDone(&is_done);
        usleep(0.1);
    }
};

void My_ICub::collectData1To1(string path) {
    //Initialization
    getHeadController();
    getRobotGazeInteface();
    getArmController(RIGHT);
    if (getDataFile(path + "dataset.txt") != 1) {
        printf("Cannot open dataset file %s!\n", (path + "dataset.txt").c_str());
        return;
    };

    setRandomVergenceAngle();
    srand(NULL);

    const double maxError = 0.03;
    double maxAngle = 15;
    double minAngle = 3;
    int direction, steps;
    steps = 6;
    tuple<int, int> doneCorrect = make_tuple(-1, 0);
    //Iterate through all values of vergence
    for (int i=0; i <= 20; i++ ) {
        int cntSamples = 0;
        //Set vergence
        setVergenceAngle(24+i);
        //Collect 100 samples
        while (cntSamples < 100) {
            direction = (get<0>(doneCorrect) == -1 || get<0>(doneCorrect) == 10 || get<0>(doneCorrect) == 9 || get<0>(doneCorrect) == 8) ? (rand() % 9) : get<0>(doneCorrect);
            doneCorrect = randomHeadMotions(direction, steps, minAngle, maxAngle, maxError);
            cntSamples += get<1>(doneCorrect);
            printf("%d. %d samples colleted!\n", i+11, cntSamples);
        }
    }
};

void My_ICub::getCurrentFixPoint(Vector &vector) {
    getRobotGazeInteface();
    iGaze->getFixationPoint(vector);
};

// Use for testing
void My_ICub::test() {

    getHeadController();
    getRobotGazeInteface();

    setVergenceAngle(17);
    Vector vector1(4);
    getCurrentFixPoint(vector1);
    printVector(vector1);
    Vector a, b, c;
    getInvKinHandAngles(vector1, a, b, c);
    printVector(vector1);
    printVector(a);
};

tuple<int, int> My_ICub::randomHeadMotions(int direction, int steps, double minAng, double maxAngle, double maxError) {
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
        xDiff = randomDoubleValue(minAng, maxAngle)*xDir;
        yDiff = randomDoubleValue(minAng, maxAngle)*yDir;
        headAngles[0] += xDiff; headAngles[2] += yDiff;
        if (!checkHeadAngles(headAngles)) {
            fprintf(stderr, "Return false because angles check!\n");
            return make_tuple(10, i); // Break the cycle if a joint angle is out of range!
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
            return make_tuple(errChck, i);
        } else {
            datafile << vectorDataToString(headAngles) << vectorDataToString(jointConf) << vectorDataToString(fixPoint) << vectorDataToString(xd) << vectorDataToString(error) << '\n';
            datafile.flush();
        }
        //getRightPalmWorldPosition(wHandY); MatrixOperations::rotoTransfWorldRoot(wHandY, rHandY);
        // print data to the file
        //cout << "Head joints: "; printVector(headAngles); cout << "\nArm joints: "; printVector(jointConf); cout << "\nFix point: "; printVector(fixPoint); cout << "\nXD: "; printVector(xd); /*cout << "\nYArm: "; printVector(rHandY);*/ cout << "\nErr: "; printVector(err); cout << endl;
    }
    return make_tuple(-1, steps);
};

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

void My_ICub::setVergenceAngle(int value) {
    head_controller->positionMove(5, value);
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
        right_arm_controller->positionMove(joints.data());
        bool is_done = false;
        while (!is_done) {
            right_arm_controller->checkMotionDone(&is_done);
            usleep(5);
        }
    }
}

/*
 *
 * titl    - cca 10:-25
 * version - cca 30:-30
 */
void My_ICub::setEyesPosition(double titl, double version, bool adding) {
    getHeadController();
    int jnts; head_controller->getAxes(&jnts);
    Vector headAngles(jnts); head_controller->getTargetPositions(headAngles.data());
    headAngles.data()[3] = titl;
    headAngles.data()[4] = version;
    head_controller->positionMove(headAngles.data());
    bool done = false;
    while (!done) {
        head_controller->checkMotionDone(&done);
        usleep(4);
    }
}

ImageOf<PixelRgb> *My_ICub::getRobotLeftEyeImage() {
    return getRobotLeftEyeDriver()->read();
}

ImageOf<PixelRgb> *My_ICub::getRobotRightEyeImage() {
    return getRobotRightEyeDriver()->read();
}

void My_ICub::takeAndSaveImages(string path) {
    string filename;
    ImageOf<PixelRgb> *leftImg = getRobotLeftEyeImage();
    if (leftImg != NULL) {
        filename = path + "l_img.ppm";
        yarp::sig::file::write(*leftImg, filename);
        cout << "\tReceived image from the left eye was saved as " << filename << endl;
    };
    ImageOf<PixelRgb> *rightImg = getRobotRightEyeImage();
    if (leftImg != NULL) {
        filename = path + "r_img.ppm";
        yarp::sig::file::write(*rightImg, filename);
        cout << "\tReceived image from the right eye was saved as " << filename << endl;
    };
}

/*
 * The main method for collecting data for 2-1 model
 */
void My_ICub::collectData2To1(string pathname) {
    //Initialization of controllers!
    if (getDataFile(pathname + "dataset.txt") != 1) {
        printf("Cannot open dataset file %s!\n", (pathname + "dataset.txt").c_str());
        return;
    };
    getHeadController();
    getWorldRpcClient();
    Vector gazeFixation, worldGCoors, handAngles, xd, od;
    srand(time(NULL));

    int vergence = 17;
    int totalCount = 0;
    int totalCountVergence = 0;
    double xDiff, yDiff;
    int numberForVergerce = 15;
    setVergenceAngle(vergence);

    // For each value from interval <17, 41> collect "numberForVergence" of points in the space
    while (vergence < 41) {
        if (totalCountVergence > numberForVergerce) {
            totalCountVergence = 0;
            vergence ++;
            setVergenceAngle(vergence);
        }
        // Generating new random orientation of the robot's eyes
        xDiff = randomDoubleValue(-20, 10);
        yDiff = randomDoubleValue(-30, 30);
        // Setting new orientation of the eyes
        setEyesPosition(xDiff, yDiff, false);
        // Receiving the point where the gaze is focused on
        getCurrentFixPoint(gazeFixation);
        // Receiving angles of hand's joints corresponding to the fixation point
        getInvKinHandAngles(gazeFixation, xd, od, handAngles);
        Vector error(3);
        for (int k=0; k<3; ++k) {
            error[k] = abs(gazeFixation[k] - xd[k]);
        }
        // Check if the difference between the hand point and fixation point is not over the limit!
        bool success = checkErrorGazeHand(gazeFixation, xd, 3.0);

        if (success == true) {
            totalCountVergence ++;
            // Creating an object representing the position of the palm in the space
            MatrixOperations::rotoTransfRootWorld(gazeFixation, worldGCoors);
            runYarpCommand(WorldYaprRpc::deleteAllObjects());
            runYarpCommand(WorldYaprRpc::createBOX(worldGCoors));
            Vector angles, results, headAngles;
            getCurrentAyesAngles(angles);
            // Designing of four head motions
            designChanges(angles, results);
            // Take image for each head conf in results and save data
            for (int r=0; 4 > r; r++) {
                double xEDiff = results[r*2]; double yEDiff = results[((r*2)+1)];
                setEyesPosition(xEDiff, yEDiff, false);
                takeAndSaveImages(pathname + to_string(totalCount) + "_");
                Vector changedAngles(2);
                getCurrentAyesAngles(changedAngles);

                datafile << totalCount << " " << vectorDataToString(changedAngles) << vectorDataToString(handAngles) << vectorDataToString(gazeFixation) << vectorDataToString(error) << endl;
                datafile.flush();
                totalCount++;
            }
        }
    }
}

bool My_ICub::checkErrorGazeHand(Vector gaze, Vector hand, double limit) {
    int dir = -2;
    bool success = true;
    Vector error(3);
    double repairAngle = 5.0;

    for (int i = 0; i < 3; ++i) {
        error[i] = gaze[i] - hand[i];
        if (abs(error[i]) > limit) {
            success = false;
        }
    }

    if (!success) {
        double xDir, xDiff, yDir, yDiff;
        double directions[] = {0, 1, 0, -1, 1, 0, -1, 0, 1, 1, -1, -1, -1, 1, 1, -1};

        if (error[1] > limit && error[2] < limit * -1) {
            dir = 4;
        } else if (error[1] < limit * -1 && error[2] > limit) {
            dir = 5;
        } else if (error[1] > limit && error[2] > limit) {
            dir = 6;
        } else if (error[1] < limit * -1 && error[2] < limit * -1) {
            dir = 7;
        } else if (error[1] < limit * -1) {
            dir = 1;
        } else if (error[1] > limit) {
            dir = 0;
        } else if (error[2] > limit) {
            dir = 3;
        } else if (error[2] < limit * -1) {
            dir = 2;
        } else {
            dir = -1;
        }
        if (dir != -1) {
            xDiff = repairAngle * directions[(dir * 2)];
            yDiff = repairAngle * directions[(dir * 2) + 1];
            setEyesPosition(xDiff, yDiff, true);
        }
    }
    return success;
}

Bottle My_ICub::runYarpCommand(Bottle bottle) {
    getWorldRpcClient();
    Bottle response;
    world_client->write(bottle, response);
    return response;
}

void My_ICub::getCurrentAyesAngles(Vector &pOf) {
    double titl; double version; double vergence;
    getHeadController();
    pOf.resize(3);

    head_controller->getTargetPosition(3, &titl);
    head_controller->getTargetPosition(4, &version);
    head_controller->getTargetPosition(5, &vergence);

    pOf[0] = titl; pOf[1] = version; pOf[2] = vergence;
}

void My_ICub::designChanges(Vector of, Vector &pOf) {
    /*
    * titl    - cca 10:-25
    * version - cca 30:-30
    */

    double upXGap, downXGap, leftYGap, rightYGap;
    upXGap = 10 - of[0];
    downXGap = of[0] + 25;
    rightYGap = 30 - of[1];
    leftYGap = of[1] + 30;

    pOf.resize(8);
    pOf[0] = randomDoubleValue(of[0], 10);
    pOf[1] = randomDoubleValue(of[1], 30);
    pOf[2] = randomDoubleValue(of[0], 10);
    pOf[3] = randomDoubleValue(-30, of[1]);
    pOf[4] = randomDoubleValue(-20, of[0]);
    pOf[5] = randomDoubleValue(of[1], 30);
    pOf[6] = randomDoubleValue(-20, of[0]);
    pOf[7] = randomDoubleValue(-30, of[1]);
}

void My_ICub::getInvKinHandAngles(Vector of, Vector &vectorOf, Vector &od, Vector &angles) {
    Property option;
    option.put("device", "cartesiancontrollerclient");
    option.put("remote", "/icubSim/cartesianController/right_arm");
    option.put("local", "/client/right_arm");

    PolyDriver clientCartCtrl(option);
    ICartesianControl *icart = NULL;

    if (clientCartCtrl.isValid()) {
        clientCartCtrl.view(icart);
    }

    icart->askForPosition(of, vectorOf, od, angles);
    //clientCartCtrl.close();
}

Vector My_ICub::getFixPointFromHeadConf(Vector headGAngles, bool takeImages, string savepath) {
    getHeadController();
    Vector headAngles; getHeadCurrentVector(headAngles);
    if (takeImages == true) {
        headAngles[3] = headGAngles[0];
        headAngles[4] = headGAngles[1];
        headAngles[5] = headGAngles[2];

    } else {
        headAngles[0] = headGAngles[0];
        headAngles[2] = headGAngles[1];
        headAngles[5] = headGAngles[2];
    }
    setHeadAnglesAndMove(headAngles);
    Vector fixPoint(3);
    getCurrentFixPoint(fixPoint);
    if (takeImages == true) {
        takeAndSaveImages(savepath);
    }

    return fixPoint;
}

Vector My_ICub::getBPointFromHandConf(Vector handGAngles, bool createBox) {
    getArmController(RIGHT);
    Vector crrhand = getCrrHandAngles();

    for (int i=0; i<handGAngles.size(); i++) {
        crrhand[i] = handGAngles[i];
    }

    if (createBox == true) {
        runYarpCommand(WorldYaprRpc::deleteAllObjects());
    }

    setArmJoints(RIGHT, crrhand);

    Bottle bottle = WorldYaprRpc::getRightHandWorldPosition();
    Bottle resp = runYarpCommand(bottle);

    Vector hand(4), bhand(3);
    hand[0] = resp.get(0).asFloat64(); hand[1] = resp.get(1).asFloat64(); hand[2] = resp.get(2).asFloat64(); hand[3] = 1.0;
    if (createBox == true) {

        for (int i=0; i<7; i++) {
            crrhand[i] = 0;
        }

        setArmJoints(RIGHT, crrhand);
        runYarpCommand(WorldYaprRpc::deleteAllObjects());
        runYarpCommand(WorldYaprRpc::createBOX(hand));
    }

    MatrixOperations::rotoTransfWorldRoot(hand, bhand);
    return bhand;
}

Vector My_ICub::getCrrHandAngles() {
    int jnts = getRightArmJoints();
    getArmController(RIGHT);
    double *angs = new double[jnts];
    right_arm_controller->getTargetPositions(angs);
    Vector handVector(jnts);
     for (int i=0; i<jnts; i++) {
        handVector[i] = angs[i];
    }
    return handVector;
}
