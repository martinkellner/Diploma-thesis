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

int main(int argc, char* argv[]) {
    string path;
    if (argc > 0) {
        path = argv[1];
        cout << "Directory for saving data: " << path << endl;
    } else {
        path = "/home/martin/School/Diploma-thesis/code/generator/data/";
        cout << "Directory for saving data: " << path << " (Default)" << endl;
    }

    fstream datafile(path + "dataset.txt", fstream::out);
    if (!datafile.is_open()) {
        cout << "Unable to open data file: " << path + "dataset.txt" << endl;
        cout << "Program exits!" << endl;
        return 0;
    };

    Network yarp;
    My_ICub *icub = new My_ICub();

    //icub->collectingData(path, 100, icub->RANDOM);
    icub->test();
    icub->closeDataFile();
};


