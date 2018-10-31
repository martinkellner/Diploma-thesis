//
// Created by martin on 10/31/18.
//

#ifndef DATASET_GENERATOR_YARP_WORLD_RPC_H
#define DATASET_GENERATOR_YARP_WORLD_RPC_H

#include <yarp/os/Bottle.h>
using namespace yarp::os;

#include <yarp/sig/Vector.h>
using namespace yarp::sig;

namespace WorldYaprRpc {
    Bottle createBOX(Vector worldVct);
    Bottle getRightHandWorldPosition();
    Bottle createCmd(char *types, ...);
    Bottle deleteAllObjects();
}

#endif //DATASET_GENERATOR_YARP_WORLD_RPC_H
