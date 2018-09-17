#include <string>
using namespace std;

#include <yarp/os/Network.h>
using namespace yarp::os;

#include <yarp/dev/PolyDriver.h>
#include <yarp/sig/Vector.h>

using namespace yarp::dev;

#include "My_ICub.h"

int main() {
    Network yarp;
    My_ICub *icub = new My_ICub();
    int axs, joint;
    const double angle = 15.0;

    // Get number of joints of the right arm
    joint = icub->getRightArmJoints();
    Vector position;
    position.resize(joint);

    // Set start position of the right arm, for now is a null vector //TODO: detect correct initial position for each of joints
    for (int i = 0; i < joint; i++) {
        position[i] = 0;
    };

    // a cycle over the entire space of the hand, each of joints is sequentially incresed by a constant angle (variable 'angle')
    for (int i=0; i<90; i++) {
        axs = i%joint;
        if (axs > 6) continue;
        position[axs] = position[axs] + angle;
        icub->rightArmMovement(position, true);
    };
};
