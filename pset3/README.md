# fbg2107 - HW3


## Part A 

the code for Part A is in the `pset3/PartA/` directory.
The Makefile therein is just for this part.
in additon to the kernel codes required we also have
```
vecaddKernel01.cu
matmultKernel01.cu
```

for each problem, there are also batch scripts to run multiple trials and python scripts to analyze the results by taking averages.

The files to do multiple run for each configuration (either values per threa   d or matrix size) and save the results to text files are in
```
run_trial_save_analyze_Problem1.sh
run_trial_save_analyze_Problem2.sh
```
the raw data can then be found in `results`, which is then prcess by 
```
analyze_trials_P1_q1q2.py
analyze_trials_P1_q3q4.py
```
the output of which are piped to text files in `results` as well.

We will discuss them, in turn, below.



### Problem  1 - Vector add and coalescing memory access


#### Q1: The response is in the Makefile, but here we highlight the relevant lines

```Makefile
# original non-coalesced vector addition (Q1)
vecaddKernel00.o : vecaddKernel00.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

# dependencies for timing, note we are using vecadd.cu as main program
# but vecaddKernel00.o as the kernel object file, this can change as 
# as we see below, where we write a different kernel 
vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o 
	${NVCC} $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)
```

> and also we chose to run each configuration for 5 trials, as shown below in `run_trial_save_analyze_Problem1.sh` to have robust measurements by averaging the time it takes for each trial.


#### Q2: The output of this experiment is  STDOUT_analysis_q1q2

```
Q1 - vecadd00 (Non-coalesced Memory Access)
--------------------------------------------------
500 values/thread:
  Trials: 5
  Avg Time: 0.000815 sec
  Min Time: 0.000796 sec
  Max Time: 0.000829 sec
  Std Dev:  0.000013 sec
  Avg GFlops/S: 4.712858
  Avg GBytes/S: 56.554295
1000 values/thread:
  Trials: 5
  Avg Time: 0.004187 sec
  Min Time: 0.001795 sec
  Max Time: 0.012657 sec
  Std Dev:  0.004742 sec
  Avg GFlops/S: 3.141563
  Avg GBytes/S: 37.698762
2000 values/thread:
  Trials: 5
  Avg Time: 0.004246 sec
  Min Time: 0.003024 sec
  Max Time: 0.008118 sec
  Std Dev:  0.002170 sec
  Avg GFlops/S: 4.135380
  Avg GBytes/S: 49.624565

Q2 - vecadd01 (Coalesced Memory Access)
--------------------------------------------------
500 values/thread:
  Trials: 5
  Avg Time: 0.000356 sec
  Min Time: 0.000312 sec
  Max Time: 0.000410 sec
  Std Dev:  0.000036 sec
  Avg GFlops/S: 10.881106
  Avg GBytes/S: 130.573276
1000 values/thread:
  Trials: 5
  Avg Time: 0.000699 sec
  Min Time: 0.000651 sec
  Max Time: 0.000748 sec
  Std Dev:  0.000044 sec
  Avg GFlops/S: 11.022135
  Avg GBytes/S: 132.265613
2000 values/thread:
  Trials: 5
  Avg Time: 0.001448 sec
  Min Time: 0.001283 sec
  Max Time: 0.001505 sec
  Std Dev:  0.000094 sec
  Avg GFlops/S: 10.645869
  Avg GBytes/S: 127.750435

PERFORMANCE COMPARISON - Coalesced vs Non-coalesced
------------------------------------------------------------
Speedup = Non-coalesced Time / Coalesced Time
(Higher speedup = better performance)
------------------------------------------------------------
 500 values/thread: 2.29x speedup (++129.2%)
1000 values/thread: 5.99x speedup (++499.1%)
2000 values/thread: 2.93x speedup (++193.2%)
```

> Based on our observations, changing from non-coalesced to coalesced memory access broadly increases the performance of the vector addition operation. This is as expected because coalesced memory access allows for more efficient use of the GPU's memory bandwidth by accessing things that are close together in memory, reducing the number of memory transactions required.


### Problem 2 - Matrix multiplication with shared memory and loop unrolling
The BLOCK_SIZE is the number of threads in a thread block (16x16 = 256 threads). This is always 16 for both matmult00 and matmult01.

The FOOTPRINT_SIZE (changing from 16 to 32), changes the size of the output submatrix that each thread block computes.

So for matmult00 
- Threads per block: 16 × 16 = 256 threads
- Elements per block: 16 × 16 = 256 elements  
- Elements per thread: 256 / 256 = 1 element
and for matmult01
- Threads per block: 16 × 16 = 256 threads
- Elements per block: 32 × 32 = 1024 elements  
- Elements per thread: 1024 / 256 = 4 elements


#### Q3: here, we chose to 8 trials for each matrix size to have robust measurements by averaging the time it takes for each trial, as shown in `run_trial_save_analyze_Problem2.sh`.

and the output of this experiment is  STDOUT_analysis_q3q4

```
matmul averaging
================================================================================
Matrix       Version    Trials Avg Time (s) Avg GFlops/S Min Time   Max Time  
--------------------------------------------------------------------------------
256x256      matmult00  8      0.000096    367.05    0.000069  0.000114
512x512      matmult00  8      0.000621    454.12    0.000425  0.000752
1024x1024    matmult00  8      0.003590    605.20    0.002881  0.004056
256x256      matmult01  8      0.000054    648.98    0.000040  0.000075
512x512      matmult01  8      0.000323    884.55    0.000218  0.000422
1024x1024    matmult01  8      0.002443    956.86    0.001545  0.003400
--------------------------------------------------------------------------------

speedup (matmult01 vs matmult00)
==================================================
Matrix Size  Speedup    GFlops Improvement  
--------------------------------------------------
256x256      1.76x     +76.8%
512x512      1.92x     +94.8%
1024x1024    1.47x     +58.1%

================================================================================
SUMMARY STATISTICS
================================================================================

matmult00:
  256x256: 8 trials, std dev: 0.000020s
  512x512: 8 trials, std dev: 0.000142s
  1024x1024: 8 trials, std dev: 0.000398s

matmult01:
  256x256: 8 trials, std dev: 0.000014s
  512x512: 8 trials, std dev: 0.000086s
  1024x1024: 8 trials, std dev: 0.000731s

```

#### Q4: Results of the experimemt

show that matmult01, which uses shared memory and loop unrolling, consistently outperforms matmult00 across all tested matrix sizes. 

The speedup ranges from approximately 1.47x to 1.92x, with the most significant improvement observed at the 512x512 matrix size. This shows that there is an optimum matrix size (in this case 512x512) where the benefits of shared memory and loop unrolling are most pronounced. When it gets too large (1024x1024), the speedup decreases, possibly due to increased memory access overhead or other factors.

Nonetheless, the use of shared memory and loop unrolling in matmult01 leads to substantial performance improvements in matrix multiplication on the GPU, almost reaching double in some cases.


#### Q5: some performance suggestion
- memomry coalescing    ensures adjacent threads are accesed
- unrolling loops reduces the overhead of loop control and increases instruction-level parallelism
- computing more per thread by increasing footprint size reduces the number of thread blocks and overhead associated with managing them
- there might be diminishing returns when increasing footprint size too much due to increased register pressure and memory access overhead

## Part-B: CUDA Unified Memory 

Compare vector operations executed on host vs on GPU to quantify the speed-up.

### Q1 
