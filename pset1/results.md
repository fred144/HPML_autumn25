# Fred Angelo Garcia (fbg2107)
```
Note, all the print statements and calculation results of  
- /dp1.c
- /dp2.c
- /dp3.c
- /dp4.py
- /dp5.py
are located in ./STDOUT
```

## Summary of results

Here are the benchmark performance for the 10 runs: 
Recall, for $N=1000000$, we repeat for 1000, while for $N=300000000$ we repeat $20$ times 
### C1 simple `dp`
```
N: 1000000 <T>: 0.001510 sec  <B>: 4.934005 GB/sec  <F>: 1.324462 GFLOP/sec
N: 300000000 <T>: 0.454184 sec  <B>: 4.921297 GB/sec  <F>: 1.321051 GFLOP/sec
```
### C2 `dpunroll`
```
N: 1000000 <T>: 0.000565 sec  <B>: 13.189902 GB/sec  <F>: 3.540637 GFLOP/sec
N: 300000000 <T>: 0.262146 sec  <B>: 8.526440 GB/sec  <F>: 2.288799 GFLOP/sec
```
### C3 `bdp` from MKL
```
N: 1000000 <T>: 0.000085 sec  <B>: 87.629475 GB/sec  <F>: 23.522858 GFLOP/sec
N: 300000000 <T>: 0.054169 sec  <B>: 41.262987 GB/sec  <F>: 11.076449 GFLOP/sec
```
### C4 simple python loop
```

```
### C5 numpy.dot
```

```
## Q1





