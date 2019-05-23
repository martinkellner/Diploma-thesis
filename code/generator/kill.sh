killall yarpserver yarpview yarprobotinterface iKinGazeCtrl iKinCartesianSolver
ps au | grep -E "^*+iCub_SIM$" | grep -oE '[0-9]*' | head -n 1 | xargs kill -9

