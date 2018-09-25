#include <string>
#include "iostream"
using namespace std;

#include <yarp/os/Network.h>
using namespace yarp::os;

#include <yarp/dev/PolyDriver.h>
#include <yarp/sig/Vector.h>

using namespace yarp::dev;

#include "My_ICub.h"

int main(int argc, char* argv[]) {
    string path;
    if (argc > 0) {
        path = argv[1];
        cout << "Directory for saving data: " << path << endl;
    } else {
        path = "/home/martin/School/Diploma-thesis/code/generator/data/";
        cout << "Directory for saving data: " << path << " (Default)" << endl;
    }

    Network yarp;
    My_ICub *icub = new My_ICub();
    int axs, joint;
    const double angle = 15.0;

    // Get number of joints of the right arm
    joint = icub->getRightArmJoints();
    Vector position;
    position.resize(joint);

    // Set start position of the right arm, for now is a null vector
    for (int i = 0; i < joint; i++) {
        position[i] = 0;
    };
    // minimal values for significant joints
    position[0] = -95.5; // shoulder pitch, max: 8
    position[1] = 0;     // shoulder roll, max: 160
    position[2] = -32;   // shoulder yaw, max: 80
    position[3] = 15;    // elbow, max: 106

    // a cycle over the entire space of the hand, each of joints is sequentially incresed by a constant angle (variable 'angle')
    int itr = 0;
    while (true) {
        cout << itr + 1 << ".movement, joints values -> " << "pos[0]: " << position[0] << " pos[1]: " << position[1] << " pos[2]: " << position[2]<< " pos[3]: " << position[3];
        axs = itr%4;

        if (((position[0] + angle) < 8) || ((position[1] + angle) < 160) || ((position[2] + angle) < 80) || ((position[3] + angle) < 106)) {
            if (axs == 0 && position[0] + angle < 8) {
                position[0] = position[0] + angle;
            } else if (axs == 1 && position[1] + angle < 160) {
                position[1] = position[1] + angle;
            } else if (axs == 2 && position[2] + angle < 80) {
                position[2] = position[2] + angle;
            } else if (axs == 3 && position[3] + angle < 106) {
                position[3] = position[3] + angle;
            }
            icub->rightArmMovement(position, true);
            icub->headMovement(angle, path);
        } else {
            position[0] = 8;
            position[1] = 160;
            position[2] = 80;
            position[3] = 15;
            icub->rightArmMovement(position, true);
            icub->headMovement(angle, path);
            break;
        };
        cout << " -- DONE!\n";
        itr = itr + 1;
    };
};
