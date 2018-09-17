#!/bin/bash
rm -p ~/.config/yarp/yarp.conf
# make build directrory if not exist
mkdir -p build/ && cd build/
# create required cmake's files and create a executable file 
cmake .. && make

# open a new terminal and run yarpserver
gnome-terminal -e yarpserver
# open a new terminal and run a instance of iCub 
gnome-terminal -e iCub_SIM
# it may take a few seconds to start yarpserver and iCub, so wait for 2 seconds
sleep 4
# run generator and collect data
./dataset

