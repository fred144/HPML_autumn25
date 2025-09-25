# copy the structure of the .c codes, using np.dot
import numpy as np
import time
import sys
from scipy.stats import hmean

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python dp5.py N reps")
        sys.exit(1)
        
    N = int(sys.argv[1]); reps = int(sys.argv[2])
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    times = []
    result = 0.0
    for r in range(reps):
        t0 = time.perf_counter()
        result = np.dot(A,B)
        t1 = time.perf_counter()
        times.append(t1-t0)
        print(f"run {r}: {times[-1]:.6f} sec result={result}")
    start = reps // 2 # burn in
    times_used = times[start:]
    
    mean = sum(times_used) / len(times_used) #arithmetic mean
    hmean_time = hmean(times_used) # harmonic mean
    
    bytes_ = N * 2 * 4
    gb = bytes_ / (1024**3)
    gbps = gb / hmean_time
    
    gflops = (2.0 * N) / hmean_time / 1e9
    print(f"\nN: {N} <T>: {mean:.6f} sec B: {gbps:.6f} GB/sec F: {gflops:.6f} FLOP/sec ")

