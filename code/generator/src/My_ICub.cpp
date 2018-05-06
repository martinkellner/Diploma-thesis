#include <iostream>
#include <string>
using namespace std;

#include "My_ICub.h"


//_____________________________________________________________________________
//______ CONSTRUCTOR, DESTRUCTOR, STATIC DECLARATIONS _________________________

My_ICub::My_ICub(string robot_name, string own_port_name) {
    this->robot_name = robot_name;
    this->own_port_name = own_port_name;
};

My_ICub::~My_ICub() {
    // TODO
};

