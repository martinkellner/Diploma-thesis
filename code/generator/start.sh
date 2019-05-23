#!/bin/bash

yarpserver &
sleep 2
iCub_SIM & 
sleep 2 
yarprobotinterface --context simCartesianControl &
sleep 2
iKinCartesianSolver --part right_arm &
sleep 2
iKinGazeCtrl --robot icubSim --imu::mode off &
sleep 2
yarpview --name /cameraleft &
sleep 2
yarp connect /icubSim/cam/left /cameraleft &
sleep 2
yarpview --name /cameraright &
sleep 2
yarp connect /icubSim/cam/right /cameraright &

