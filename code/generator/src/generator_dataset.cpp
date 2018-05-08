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

    int axs = 0;
    for (int i=0; i<5; i++) {
        double angle = 30.0;
        icub->headMovement(angle, axs, true);
        angle = angle * (-1);
        icub->headMovement(angle, axs, true);
    };
};
