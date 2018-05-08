#include <string>
using namespace std;

#include <yarp/os/Network.h>
using namespace yarp::os;

#include <yarp/dev/PolyDriver.h>
using namespace yarp::dev;

#include "My_ICub.h"

int main() {
    Network yarp;
    My_ICub *icub = new My_ICub();
    
    icub->headMovement();
};
