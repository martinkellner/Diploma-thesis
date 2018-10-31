//
// Created by martin on 10/26/18.cm
//

#include <eigen3/Eigen/Dense>
#include <matrix_operations.h>

using namespace Eigen;

#include "yarp/sig/Vector.h"
using namespace yarp::sig;

#include <string>
#include <stdlib.h>
#include <iostream>
using namespace std;

/*
 * Transformation a point from the world reference frame to the root frame of reference
 */
void MatrixOperations::rotoTransfWorldRoot(Vector worldVct, Vector &rootVct) {
    Matrix4f rototransM; // Rototransformation matrix from world to root frame
    /*
     *      |  0  0 -1   0.026   |
     *      | -1  0  0   0       |
     * M =  |  0  1  0   -0.5976 |
     *      |  0  0  0   1       |
     */
    rototransM <<  0, 0, -1, -0.026, -1, 0, 0, 0, 0, 1, 0, -0.5976, 0, 0, 0, 1;

    // Yapr vector -> Eigen lib vector;
    Vector4f worldVctE;
    int size;
    if ((size = worldVct.size()) < 4) {
        printf("Given world vector must have at least 4 dimensions, current size: %d\n", size);
        return;
    }
    worldVctE << worldVct[0], worldVct[1], worldVct[2], worldVct[3];
    Vector4f rootVctE;
    // Rototransfomation to the reference frame.
    rootVctE = rototransM * worldVctE;
    // Setting Eigen vector's data to the result root vector (yarp vector)
    float *resData = rootVctE.data();
    for (int i=0; i<3; i++) {
        rootVct[i] = resData[i];
    }
}

void MatrixOperations::rotoTransfRootWorld(Vector rootVct, Vector &rWorldVct) {
    Matrix4f rototransM; // Rototransformation matrix from root to world frame
    /*
     *      |  0 -1 0      0 |
     *      |  0  0 1 0.5976 |
     * M =  | -1  0 0 -0.026 |
     *      |  0  0 0      1 |
     */

    rototransM <<  0, -1, 0, 0, 0, 0, 1, 0.5976, -1, 0, 0, -0.026, 0, 0, 0, 1;
    // Yapr vector -> Eigen lib vector;
    Vector4f rootVctE;
    int size;
    if ((size = rootVct.size()) < 4) {
        printf("Given world vector must have at least 4 dimensions, current size: %d\n", size);
        return;
    }
    rootVctE << rootVct[0], rootVct[1], rootVct[2], rootVct[3];
    Vector4f worldVctE;
    // Rototransfomation to the reference frame.
    worldVctE = rototransM * rootVctE;
    // Setting Eigen vector's data to the result world vector (yarp vector)
    float *resData = worldVctE.data();
    for (int i=0; i<3; i++) {
        rWorldVct[i] = resData[i];
    }
}
