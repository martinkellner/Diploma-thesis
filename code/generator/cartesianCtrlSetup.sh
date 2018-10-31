#!/bin/bash

yarpserver &

sleep 2

iCub_SIM &

sleep 2

yarprobotinterface --context simCartesianControl &

sleep 1

iKinCartesianSolver --part right_arm &
