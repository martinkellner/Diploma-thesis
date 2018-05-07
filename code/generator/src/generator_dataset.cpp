#include <stdio.h>
using namespace std;

#include <iostream>
#include <fstream>
#include <string>
#include <yarp/os/Network.h>

using namespace yarp::os;

#include <yarp/dev/PolyDriver.h>
using namespace yarp::dev;

#include "My_ICub.h"


int main() {
    Network yarp;
    My_ICub *icub = new My_ICub();

    PolyDriver *driver = icub->getRobotHeadDriver();
};
