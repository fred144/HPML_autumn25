# dp4.py - Python dot product benchmark (simple loop)
import numpy as np
import time
import sys
from scipy.stats import hmean

def dp(N, A, B):
    R = 0.0
    for j in range(N):
        R += A[j] * B[j]
    return R

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dp4.py N reps")
        sys.exit(1)
        
    N = int(sys.argv[1]) 
    reps = int(sys.argv[2])

    # initialize arrays to 1.0
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    times = []
    result = 0.0

    for r in range(reps):
        t0 = time.perf_counter()
        result = dp(N, A, B)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"run {r}: {times[-1]:.6f} sec result={result}")

    # discard first half as burn-in
    start = reps // 2
    times_used = times[start:]

    # arithmetic mean of times
    mean_time = sum(times_used) / len(times_used)

    # harmonic mean of bandwidth and FLOP rate
    bytes_ = N * 2 * 4  # 2 arrays, 4 bytes each (float32)
    gb = bytes_ / (1024.0**3)  # GB per run

    bw_values = [gb / t for t in times_used]
    flops_values = [(2.0 * N) / t / 1e9 for t in times_used]

    harm_bw = hmean(bw_values)
    harm_flops = hmean(flops_values)

    print(f"\nN: {N} <T>: {mean_time:.6f} sec "
          f"B: {harm_bw:.6f} GB/sec "
          f"F: {harm_flops:.6f} GFLOP/sec")
