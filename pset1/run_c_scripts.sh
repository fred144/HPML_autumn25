#!/bin/bash

# run dp1 with different arguments and save outputs
./dp1 1000000 1000 | tee STDOUT/dp1_N1e6_1e3reps_out.txt
./dp1 300000000 20 | tee STDOUT/dp1_N3e8_20reps_out.txt

./dp2 1000000 1000 | tee STDOUT/dp2_N1e6_1e3reps_out.txt
./dp2 300000000 20 | tee STDOUT/dp2_N3e8_20reps_out.txt

./dp3 1000000 1000 | tee STDOUT/dp3_N1e6_1e3reps_out.txt
./dp3 300000000 20 | tee STDOUT/dp3_N3e8_20reps_out.txt