#!/bin/bash

# run dp1 with different arguments and save outputs
python3 dp4.py 1000000 1000 | tee STDOUT/dp4_N1e6_1e3reps_out.txt
python3 dp4.py 300000000 20 | tee STDOUT/dp4_N3e8_20reps_out.txt

python3 dp5.py 1000000 1000 | tee STDOUT/dp5_N1e6_1e3reps_out.txt
python3 dp5.py 300000000 20 | tee STDOUT/dp5_N3e8_20reps_out.txt