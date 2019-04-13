#include <string>
#include "iostream"
#include <fstream>
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

int main(int argc, char* argv[]) {

    Network yarp;
    My_ICub *icub = new My_ICub();

    std::cout << "\n###################################\n[1] Data collecting script for 1-1.\n[2] Data collecting script for 2-1.\n[3] None of previous choises.\n###################################" << std::endl;
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
        delete icub;
        retinalModelValidData();
        //headAnglesToPoint();
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