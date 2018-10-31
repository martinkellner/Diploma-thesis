//
// Created by martin on 10/26/18.
//

#ifndef DATASET_GENERATOR_MATRIX_OPERATIONS_H
#define DATASET_GENERATOR_MATRIX_OPERATIONS_H

#include <yarp/sig/Vector.h>
using namespace yarp::sig;

namespace MatrixOperations  {
    void rotoTransfWorldRoot(Vector worldVct, Vector &rootVct);
    void rotoTransfRootWorld(Vector rootVct, Vector &rWorldVct);
}

#endif //DATASET_GENERATOR_MATRIX_OPERATIONS_H


