#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <yarp/os/Network.h>
#include "My_ICub.h"

using namespace yarp::os;
using namespace std;


int main() {
    Network yarp;
    My_ICub *icub = new My_ICub();    
};
