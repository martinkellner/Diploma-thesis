#pragma once
#ifndef MY_ICUB_H
#define MY_ICUB_H

#include <string.h>
using namespace std;

class My_ICub {
    
public:
    My_ICub(string robot_name="/icubSim", string own_port_name="/mysim");
    ~My_ICub();
   
protected:
    string
        robot_name,
        own_port_name;            
};

#endif
