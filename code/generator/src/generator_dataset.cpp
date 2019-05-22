#include <string>
#include "iostream"
#include <fstream>
#include <math.h>
using namespace std;

#include <yarp/os/Network.h>
using namespace yarp::os;

#include <yarp/dev/PolyDriver.h>
#include <yarp/sig/Vector.h>

using namespace yarp::dev;

#include "My_ICub.h"
#include "matrix_operations.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <bits/stdc++.h>
#include <zconf.h>
#include "yarp_world_rpc.h"

float euclideanDistance(Vector pose1, Vector pose2) {
    return sqrt(pow(pose1[0] - pose2[0], 2) + pow(pose1[1] - pose2[1], 2) + pow(pose1[2] - pose2[2], 2));
}

void EyePalmCoordination() {
    Network yarp;
    My_ICub *icub = new My_ICub();
    string choise = "";
    std::cout << "\n###################################\n[1] Eye->Arm Coordination. \n[2] Arm->Eye Coordination. \n###################################" << std::endl;
    std::cout << "Select your choise: ";
    getline(cin, choise);
    string cmd = "python3 /home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/ModelTest.py 2";
    int vergence = 22;
    Vector fixPoint; Vector worldGCoors; Vector desChanges; Vector eyePose(2);
    string path = "/home/martin/School/Diploma-thesis/code/";
    string resultString = "";
    double xDiffs[3] = {-10.430184, 0.97521, -18.0954};
    double yDiffs[3] = {25.31177, -0.9428, -27.790};
    double vergences[3] = {27, 30, 36};
    Vector handVector = icub->getCrrHandAngles();
    Vector orHandVector = icub->getCrrHandAngles();
    Vector palmWorld; Vector palmRoot;
    icub->setArmJoints(icub->RIGHT, orHandVector);
    float arrayFxp[9] = {-0.206299, -0.00997814, 0.267164, -0.183061, 0.0587184, 0.295898, -0.19319, -0.0707147, 0.306615};
    srand(time(NULL));
    if (choise == "1") {
        for (int i=0;i < 3; i++) {
            ifstream file(path + "channel.txt");
            double xDiff = xDiffs[i];//icub->randomDoubleValue(-20, 10);
            double yDiff = yDiffs[i];//icub->randomDoubleValue(-30, 30);
            cout << xDiff << " " << yDiff << endl;
            icub->setEyesPosition(xDiff, yDiff, false);
            eyePose[0] = xDiff; eyePose[1] = yDiff;
            icub->setVergenceAngle(vergences[i]);

            icub->getCurrentFixPoint(fixPoint);
            MatrixOperations::rotoTransfRootWorld(fixPoint, worldGCoors);
            icub->runYarpCommand(WorldYaprRpc::createBOX(worldGCoors));
            icub->designChanges(eyePose, desChanges);
            int chidx = static_cast<int>(icub->randomDoubleValue(0, 3));
            double chXdiff = desChanges[chidx*2]; double chYdiff = desChanges[(chidx*2)+1];
            int diff = 7;
            if (abs(chXdiff-xDiff) > diff) {
                if (chXdiff > xDiff) {
                    chXdiff = xDiff + diff;
                } else {
                    chXdiff = chXdiff + diff;
                }
            }
            if (abs(chYdiff-yDiff) > diff) {
                if (chYdiff > yDiff) {
                    chYdiff = yDiff + diff;
                } else {
                    chYdiff = chYdiff + diff;
                }
            }

            icub->setEyesPosition(chXdiff, chYdiff, false);
            icub->takeAndSaveImages(path);
            sleep(4);
            icub->runYarpCommand(WorldYaprRpc::deleteAllObjects());
            string cmdE = cmd + " " + to_string(xDiff) + " " + to_string(yDiff) + " " + to_string(vergence);
            int wait = std::system(cmdE.c_str());
            sleep(3);
            string line;
            getline(file, line);
            std::stringstream lineS(line);

            for (int j = 0; j < 7; j++) {
                if (lineS.peek() == ' ') {
                    lineS.ignore();
                }
                float value;
                lineS >> value;
                handVector[j] = value;
            }
            file.close();
            lineS.clear();
            icub->setArmJoints(icub->RIGHT, handVector);

            icub->getRightPalmWorldPosition(palmWorld);
            icub->runYarpCommand(WorldYaprRpc::createBOX(worldGCoors));
            sleep(3);

            MatrixOperations::rotoTransfWorldRoot(palmWorld, palmRoot);
            icub->printVector(palmRoot);

            float dist = euclideanDistance(fixPoint, palmRoot);
            resultString += to_string(i) + ".: " + to_string(dist) + "\n";
            icub->runYarpCommand(WorldYaprRpc::deleteAllObjects());
            icub->setArmJoints(icub->RIGHT, orHandVector);
        }
        cout << "Result: \n####################\n" << resultString << endl;

    } else if (choise == "2") {
        Vector objPoint(3);
        Vector handJoint, xd, od, wordlCoords, eyeVector(3);
        for (int i=0; i<3; i++) {
            ifstream file(path + "channel.txt");
            objPoint[0] = arrayFxp[i * 3];
            objPoint[1] = arrayFxp[(i * 3) + 1];
            objPoint[2] = arrayFxp[(i * 3) + 2];
            cout << "Checkpoint1" << endl;
            icub->getInvKinHandAngles(objPoint, xd, od, handJoint);
            icub->getArmJoints(handJoint);
            cout << "Checkpoint2" << endl;
            MatrixOperations::rotoTransfRootWorld(objPoint, wordlCoords);
            icub->runYarpCommand(WorldYaprRpc::createBOX(wordlCoords));
            string cmdE = cmd + " " + to_string(handJoint[0]) + " " + to_string(handJoint[1]) + " " +
                          to_string(handJoint[2]) + " " + to_string(handJoint[3]) + " " + to_string(handJoint[4]) +
                          " " + to_string(handJoint[5]) + " " + to_string(handJoint[6]);
            int wait = std::system(cmdE.c_str());
            sleep(3);
            string line;
            getline(file, line);
            std::stringstream lineS(line);

            for (int j = 0; j < 3; j++) {
                if (lineS.peek() == ' ') {
                    lineS.ignore();
                }
                float value;
                lineS >> value;
                eyeVector[j] = value;
            }
            file.close();
            lineS.clear();
            icub->setEyesPosition(eyeVector[0], eyeVector[1], false);
            icub->setVergenceAngle(static_cast<int>(eyeVector[2]));
            sleep(3);
            icub->runYarpCommand(WorldYaprRpc::deleteAllObjects());
        }
    }
}

void HeadPalmCoordination() {
    Network yarp;
    My_ICub *icub = new My_ICub();
    Vector headJoints; Vector fixPoint(3);
    string cmd = "python3 /home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/ModelTest.py 1";
    Vector handJoints; Vector xd; Vector od;
    Vector handOrVector = icub->getCrrHandAngles();
    handJoints = handOrVector;
    Vector palmWorld; Vector palmRoot;

    string choise = "";
    std::cout << "\n###################################\n[1] Head->Arm Coordination. \n[2] Arm->Head Coordination. \n###################################" << std::endl;
    std::cout << "Select your choise: ";
    getline(cin, choise);
    string resultString = "";
    icub->printVector(handOrVector);

    float arrayFxp[15] = {-0.10, 0.1, 0.35, -0.2, 0.12, 0.23, -0.17, -0.03, 0.35, -0.22, -0.05, 0.23, -0.30, 0.0, 0.35};

    if (choise == "1") {
        for (int i = 0; i < 5; ++i) {
            ifstream file("/home/martin/School/Diploma-thesis/code/channel.txt");
            icub->setRandomVergenceAngle();
            double xDiff = icub->randomDoubleValue(-40, 30);
            double yDiff = icub->randomDoubleValue(-54, 42);
            icub->getHeadCurrentVector(headJoints);
            headJoints[0] = xDiff;
            headJoints[2] = yDiff;
            icub->setHeadAnglesAndMove(headJoints);
            icub->getCurrentFixPoint(fixPoint);
            string cmdE = cmd + " " + to_string(headJoints[0]) + " " + to_string(headJoints[2]) + " " + to_string(headJoints[5]);
            int wait = std::system(cmdE.c_str());
            usleep(3);
            string line;
            getline(file, line);
            std::stringstream lineS(line);

            for (int j = 0; j < 7; j++) {
                if (lineS.peek() == ' ') {
                    lineS.ignore();
                }
                float value;
                lineS >> value;
                handJoints[j] = value;
            }
            file.close();
            lineS.clear();
            icub->setArmJoints(icub->RIGHT, handJoints);

            icub->getRightPalmWorldPosition(palmWorld);
            MatrixOperations::rotoTransfWorldRoot(palmWorld, palmRoot);
            float dist = euclideanDistance(fixPoint, palmRoot);
            resultString += to_string(i) + ".: " + to_string(dist) + "\n";

        }

        cout <<  "Results: \n#####################\n" << resultString << endl;
        icub->setArmJoints(icub->RIGHT, handOrVector);
    } else if (choise == "2") {
        icub->setArmJoints(icub->RIGHT, handOrVector);
        for (int i = 0; i < 5; i++) {
            ifstream file("/home/martin/School/Diploma-thesis/code/channel.txt");
            fixPoint[0] = arrayFxp[(3 * i)];
            fixPoint[1] = arrayFxp[(3 * i) + 1];
            fixPoint[2] = arrayFxp[(3 * i) + 2];

            icub->getInvKinHandAngles(fixPoint, xd, od, handJoints);
            icub->getArmJoints(handJoints);
            icub->setArmJoints(icub->RIGHT, handJoints);
            string cmdE = cmd + " " + to_string(handJoints[0]) + " " + to_string(handJoints[1]) + " " +
                          to_string(handJoints[2]) + " " + to_string(handJoints[3]) + " " + to_string(handJoints[4]) +
                          " " + to_string(handJoints[5]) + " " + to_string(handJoints[6]);
            int wait = std::system(cmdE.c_str());
            usleep(5);
            icub->getHeadCurrentVector(headJoints);
            headJoints[1] = headJoints[3] = headJoints[4] = 0;
            string line;
            getline(file, line);
            std::stringstream lineS(line);
            Vector storeRes(3);
            for (int j = 0; j < 3; j++) {
                if (lineS.peek() == ' ') {
                    lineS.ignore();
                }
                float value;
                lineS >> value;
                if (j == 0) {
                    headJoints[0] = value;
                } else if (j == 1) {
                    headJoints[2] = value;
                } else {
                    headJoints[5] = value;
                }
            }

            //icub->printVector(headJoints);

            file.close();
            lineS.clear();
            icub->setHeadAnglesAndMove(headJoints);
            usleep(3);
            icub->getCurrentFixPoint(fixPoint);
            icub->getRightPalmWorldPosition(palmWorld);
            MatrixOperations::rotoTransfWorldRoot(palmWorld, palmRoot);
            float dist = euclideanDistance(fixPoint, palmRoot);
            resultString += to_string(i) + ".: " + to_string(dist) + "\n";
            icub->setArmJoints(icub->RIGHT, handOrVector);
        }


        cout <<  "Results: \n#####################\n" << resultString << endl;

    } else {
        std::cout << "Wrong input. Bey!" << endl;
    }
}


void headAnglesToPoint() {
    Network yarp;
   My_ICub *icub = new My_ICub();

    ofstream fileOut("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/1-1v2/points.txt");

    string line;
    ifstream file("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/1-1v2/_predicted_u.csv");

    if (file.is_open()) {
        getline(file, line);
        while (getline(file, line)) {
            std::stringstream lineS(line);

            Vector head(3); Vector hand(7);
            for (int i=-1; i < 10; i++) {
                if (lineS.peek() == ',') {
                    lineS.ignore();
                }
                float value; lineS >> value;
                if (i==-1) continue;
                if (i < 3) {
                    head[i] = value;
                } else {
                    hand[i-3] = value;
                }
            }

            Vector fxP = icub->getFixPointFromHeadConf(head, false, "");
            Vector plP = icub->getBPointFromHandConf(hand, false);

            fileOut << icub->vectorDataToString(fxP) << icub->vectorDataToString(plP) << '\n';
            fileOut.flush();
        }
    }
}

void retinalModelValidData() {

    Network yarp;
    My_ICub *icub = new My_ICub();

    //ofstream fileOut("/home/martin/data/testingV2/ret_point.txt");
    //ofstream fileOut("/home/martin/data/newTesting/2-1v1/21v_ret_point.csv");
    ofstream fileOut("/home/martin/data/newTesting/2-1v2np/21vnp_ret_point.csv");

    string line;
    //ifstream file("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/prediction/new_res_to_v.csv");
    //ifstream file("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v1/pred_datasetv2.csv");
    ifstream file("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v2np/preDataset.csv");

    int idx = 0;
    if (file.is_open()) {
        getline(file, line);
        while (getline(file, line)) {

            std::stringstream lineS(line);
            Vector head(3);
            Vector hand(7);
            for (int i = -1; i < 10; i++) {
                if (lineS.peek() == ',') {
                    lineS.ignore();
                }
                float value;
                lineS >> value;
                if (i == -1) continue;
                if (i < 3) {
                    head[i] = value;
                } else if (i >= 3) {
                    hand[i - 3] = value;
                }
            }
            icub->printVector(hand);
            icub->printVector(head);

            hand[0] = hand[0] < -95.0 ? -95 : hand[0];
            hand[3] = hand[3] > 106.0 ? 106 : hand[3];
            hand[5] = hand[5] > 0.0 ? 0 : hand[5];


            Vector plP = icub->getBPointFromHandConf(hand, true);
            //Vector fxP = icub->getFixPointFromHeadConf(head, true, "/home/martin/data/testingV2/" + to_string(i));
            Vector fxP = icub->getFixPointFromHeadConf(head, true, "/home/martin/data/newTesting/2-1v2np/" + to_string(idx));
            fileOut << icub->vectorDataToString(fxP) << icub->vectorDataToString(plP) << '\n';
            fileOut.flush();

            idx ++;
        }
    }
}

void retinalModelValidData2() {

    Network yarp;
    My_ICub *icub = new My_ICub();

    string line;
    ifstream file("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v1/preffereddir.csv");
    string pathtosave = "/home/martin/data/preff/";
    Vector eyes(3);
    Vector fixp(3);

    int idx = 0;
    if (file.is_open()) {
        getline(file, line);
        while (getline(file, line)) {

            std::stringstream lineS(line);
            for (int i = -1; i < 6; i++) {
                if (lineS.peek() == ',') {
                    lineS.ignore();
                }
                float value;
                lineS >> value;
                if (i == -1) continue;
                if (i < 3) {
                    eyes[i] = value;
                } else if (i >= 3) {
                    fixp[i - 3] = value;
                }
            }
            icub->printVector(eyes);
            icub->printVector(fixp);
            icub->explorePreffDir(eyes, fixp, pathtosave + "_" + to_string(idx));
            idx ++;
        }
    }

    delete icub;
}

void retinalModelValidData3() {

    Network yarp;
    My_ICub *icub = new My_ICub();


    string line;
    ofstream fileOut("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v2np/21vnp_ret_point2.csv");
    ifstream file("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v2np/preDataset2.csv");

    int idx = 0;
    if (file.is_open()) {
        getline(file, line);
        while (getline(file, line)) {
            Vector hand(7);
            std::stringstream lineS(line);
            for (int i = -1; i < 10; i++) {
                if (lineS.peek() == ',') {
                    lineS.ignore();
                }
                float value;
                lineS >> value;
                if (i < 3) continue;
                else {
                    hand[i-3] = value;
                }
            }

            hand[0] = hand[0] < -95.0 ? -95 : hand[0];
            hand[1] = hand[1] > 26 ? 26 : hand[1];
            hand[3] = hand[3] > 106.0 ? 106 : hand[3];
            hand[5] = hand[5] > 0.0 ? 0 : hand[5];


            icub->printVector(hand);
            Vector bpoint = icub->getBPointFromHandConf(hand, false);
            fileOut << icub->vectorDataToString(bpoint) << endl;
            idx ++;
        }
    }

    delete icub;
}

int main(int argc, char* argv[]) {

    Network yarp;
    My_ICub *icub = new My_ICub();

    std::cout << "\n###################################\n[1] Data collecting script for 1-1.\n[2] Data collecting script for 2-1.\n[3] Demonstrating the functionality of the first model.\n[4] Demonstrating the functionality of the second model.\n###################################" << std::endl;
    std::cout << "(Warning: If want to run some script, make sure yourself that Yarpserver, iCub_SIM, and other required tools are running!)" << std::endl;
    std::cout << "Select your choise: ";
    string choise = "";
    getline(cin, choise);
    string pathname = "";
    if (choise == "1") {
        std::cout << "\nWhere to save data? " << std::endl;
        getline(cin, pathname);
    } else if (choise == "2") {
        std::cout << "\nWhere to save data? " << std::endl;
        getline(cin, pathname);
    } else if (choise == "3") {
        //TODO: Run what you want instead of data collecting scripts
        //delete icub;
        //retinalModelValidData();
        //headAnglesToPoint();
        //retinalModelValidData2();
        //retinalModelValidData3();
        HeadPalmCoordination();
        return EXIT_SUCCESS;
    } else if (choise == "4") {
        EyePalmCoordination();
        return EXIT_SUCCESS;
    } else {
        delete icub;
        return EXIT_FAILURE;
    }

    string deafult = "/home/martin/datav2/";

    if (pathname == "" || pathname == "\n") {
        printf("No folder for saving data given, default folder will be used!\n");
        struct stat info;
        if (stat(deafult.c_str(), &info) != 0) {
            printf("Cannot access %s!", deafult.c_str());
        } else if (!(info.st_mode & S_IFDIR)) {
            if (mkdir(deafult.c_str(), 0777) != -1 ) {
                printf("Default dir created! %s\n", deafult.c_str());
            } else {
                printf("Cannot create default dir %s!\n", deafult.c_str());
                return EXIT_FAILURE;
            }
        }
        pathname = deafult;
    } else {
        string filename = pathname;
        struct stat info2;
        if (stat(filename.c_str(), &info2) != 0) {
            printf("Cannot access %s!\n", filename.c_str());
        } else if ((info2.st_mode == S_IFDIR)) {
            if (mkdir(filename.c_str(), 0777) != -1 ) {
                printf("New folder %s created!\n", filename.c_str());
            } else {
                printf("Cannot create folder %s!\n", filename.c_str());
                return EXIT_FAILURE;
            }
        }
        pathname = filename;
    }
    pathname = pathname[pathname.length()-1] != '/' ? pathname + "/" : pathname;
    if (choise == "1") {
        icub->collectData1To1(pathname);
    } else if (choise == "2") {
        icub->collectData2To1(pathname);
    }

    //icub->collectData2To1(pathname);
    //system("../kill.sh"); // run the shell script that kills all processes that needed!
    return EXIT_SUCCESS;
};