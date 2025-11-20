#!/usr/bin/env python3

import re
import statistics

def extract_metrics(content):
    """Extract time, GFlops/S, and GBytes/S """
    # time, gflops, gbytes on the same line with spaces
    # 0.000818       4.694295        56.331544       500
    lines = content.split('\n')
    for line in lines:
        # look for lines with 4 numbers separated by spaces
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                time = float(parts[0])
                gflops = float(parts[1])
                gbytes = float(parts[2])
                # parts[3] is the ValuesPerThread, which we don't need here and is in the text filename
                return {
                    'time': time,
                    'gflops': gflops,
                    'gbytes': gbytes
                }
            except (ValueError, IndexError):
                continue
    return None

def analyze_trials(filename, desc):
    """analyze multiple trials from a file"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # split into individual trial outputs using "Total vector size:" as delimiter
        trials = content.split('Total vector size:')
        trials = [t for t in trials if t.strip()]  # Remove empty
        
        times = []
        gflops_list = []
        gbytes_list = []
        
        for trial in trials:
            metrics = extract_metrics(trial)
            if metrics:
                times.append(metrics['time'])
                gflops_list.append(metrics['gflops'])
                gbytes_list.append(metrics['gbytes'])
        
        if times:
            avg_time = statistics.mean(times)
            avg_gflops = statistics.mean(gflops_list)
            avg_gbytes = statistics.mean(gbytes_list)
            min_time = min(times)
            max_time = max(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            
            print(f"{desc}:")
            print(f"  Trials: {len(times)}")
            print(f"  Avg Time: {avg_time:.6f} sec")
            print(f"  Min Time: {min_time:.6f} sec")
            print(f"  Max Time: {max_time:.6f} sec")
            print(f"  Std Dev:  {std_time:.6f} sec")
            print(f"  Avg GFlops/S: {avg_gflops:.6f}")
            print(f"  Avg GBytes/S: {avg_gbytes:.6f}")
            return avg_time
        else:
            print(f"{desc}: no valid data found")
            return None
            
    except FileNotFoundError:
        print(f"{desc}: file not found, make sure its in the results/ directory")
        return None


print("=" * 60)
print("VECTOR ADDITION - MULTIPLE TRIALS ANALYSIS")
print("=" * 60)

# Q1 analysis - non-coalesced
print("\nQ1 - vecadd00 (Non-coalesced Memory Access)")
print("-" * 50)
q1_500 = analyze_trials("results/q1_500_trials.txt", "500 values/thread")
q1_1000 = analyze_trials("results/q1_1000_trials.txt", "1000 values/thread")
q1_2000 = analyze_trials("results/q1_2000_trials.txt", "2000 values/thread")

# Q2 analysis - coalesced
print("\nQ2 - vecadd01 (Coalesced Memory Access)")
print("-" * 50)
q2_500 = analyze_trials("results/q2_500_trials.txt", "500 values/thread")
q2_1000 = analyze_trials("results/q2_1000_trials.txt", "1000 values/thread")
q2_2000 = analyze_trials("results/q2_2000_trials.txt", "2000 values/thread")

# Performance Comparison
print("\nPERFORMANCE COMPARISON - Coalesced vs Non-coalesced")
print("-" * 60)
print("Speedup = Non-coalesced Time / Coalesced Time")
print("(Higher speedup = better performance)")
print("-" * 60)

comparisons = [
    (500, q1_500, q2_500),
    (1000, q1_1000, q2_1000),
    (2000, q1_2000, q2_2000)
]

for vpt, q1_time, q2_time in comparisons:
    if q1_time and q2_time:
        speedup = q1_time / q2_time
        improvement = (speedup - 1) * 100
        if speedup > 1:
            result = f"{speedup:.2f}x speedup (+{improvement:+.1f}%)"
        else:
            result = f"{speedup:.2f}x slowdown ({improvement:+.1f}%)"
        print(f"{vpt:4d} values/thread: {result}")
    else:
        print(f"{vpt:4d} values/thread: comparison not available -- make sure both trials completed successfully")



