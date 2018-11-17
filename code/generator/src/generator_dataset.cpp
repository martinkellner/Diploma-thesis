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

    icub->randomLookWayCollecting(path, 0, 50);
    //Vector w(4); w[0] = -0.121811519717529; w[1] = 0.727852684714889; w[2] = 0.307035784201394; w[3] =1;
    //Vector r;
    //MatrixOperations::rotoTransfWorldRoot(w,r); cout << r[0] << " " << r[1] << " " << r[2] << endl;
    //icub->test();
    icub->closeDataFile();
};


