#!/bin/bash

yarpserver &

sleep 2

iCub_SIM & 

sleep 2

iKinGazeCtrl --robot icubSim --imu::mode off
