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


int main(int argc, char* argv[]) {
    //system("../start.sh");
    string deafult = "~/data/";
    Network yarp;
    My_ICub *icub = new My_ICub();
    string pathname;

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


    icub->collectData(pathname);


    //icub->closeDat
    //system("../kill.sh"); // run the shell script that kills all processes that needed!
    return EXIT_SUCCESS;
};


