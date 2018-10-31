CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(DATASET_GENERATOR)

FIND_PACKAGE(YARP REQUIRED)

INCLUDE_DIRECTORIES(${YARP_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(include)

SET(SOURCES src/generator_dataset.cpp
            src/My_ICub.cpp
            src/matrix_operations.cpp
            src/yarp_world_rpc.cpp
            include/My_ICub.h
            include/matrix_operations.h
            include/yarp_world_rpc.h
)

ADD_EXECUTABLE(dataset ${SOURCES})

TARGET_LINK_LIBRARIES(dataset ${YARP_LIBRARIES})
