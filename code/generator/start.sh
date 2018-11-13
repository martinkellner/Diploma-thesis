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
