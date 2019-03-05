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

    ofstream fileOut("/home/martin/School/Diploma-thesis/code/generator/data/points.txt");

    string line;
    ifstream file("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/prediction/2.NET_predicted_u.csv");

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

    ofstream fileOut("/home/martin/data/testing/ret_pointsv2.txt");

    string line;
    ifstream file("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/prediction/pred_datasetv2.csv");

    int i = 0;
    if (file.is_open()) {
        getline(file, line);
        while (getline(file, line)) {
            std::stringstream lineS(line);
            Vector head(3);
            Vector hand(7);
            for (int i = -1; i < 16; i++) {
                if (lineS.peek() == ',') {
                    lineS.ignore();
                }
                float value;
                lineS >> value;
                if (i == -1) continue;
                if (i < 3) {
                    head[i] = value;
                } else if (i > 8) {
                    hand[i - 9] = value;
                }
            }

            Vector plP = icub->getBPointFromHandConf(hand, true);
            Vector fxP = icub->getFixPointFromHeadConf(head, true, "/home/martin/data/testing/" + to_string(i));
            fileOut << icub->vectorDataToString(fxP) << icub->vectorDataToString(plP) << '\n';
            fileOut.flush();

            i ++;
        }
    }
}

int main(int argc, char* argv[]) {

    //Network yarp;
    //My_ICub *icub = new My_ICub();

    //system("../start.sh");
    string deafult = "~/data/";
    //Network yarp;
    //My_ICub *icub = new My_ICub();
    string pathname;

    //headAnglesToPoint();
    retinalModelValidData();

    //icub->test();
    return 1;

    if (argv[1] == NULL) {
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
        string filename = argv[1];
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


    //icub->collectData(pathname);

    //icub->closeDat
    //system("../kill.sh"); // run the shell script that kills all processes that needed!
    return EXIT_SUCCESS;
};


