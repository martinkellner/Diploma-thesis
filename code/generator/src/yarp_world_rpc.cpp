//
// Created by martin on 10/31/18.
//

#include <iostream>
using namespace std;

#include "yarp_world_rpc.h"
#include <yarp/os/Bottle.h>
#include <stdio.h>
#include <stdarg.h>
#include <yarp_world_rpc.h>


using namespace yarp::os;

Bottle WorldYaprRpc::createBOX(Vector worldVct) {
    //TODO: set color to red (or another)
    return createCmd("sssfffffffff", "world", "mk", "sbox", .03, .03, .03, worldVct[0], worldVct[1], worldVct[2], 1, 1, 1);

}

Bottle WorldYaprRpc::getRightHandWorldPosition() {
    return createCmd("sss", "world", "get", "rhand");
}

Bottle WorldYaprRpc::createCmd(char *types, ...) {
    va_list list;
    Bottle bottle;
    va_start(list, types);

    for (int i = 0; types[i] != '\0'; i++) {
        if (types[i] == 's') {
            bottle.addString(va_arg(list, char *));
        } else if (types[i] == 'i') {
            bottle.addInt(va_arg(list, int));
        } else if (types[i] == 'f') {
            bottle.addDouble(va_arg(list, double));
        }
    }

    return bottle;
}

Bottle WorldYaprRpc::deleteAllObjects() {
    return createCmd("sss", "world", "del", "all");
}

