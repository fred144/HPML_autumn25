#!/usr/bin/env python3

import re
import statistics
import os


def extract_metrics(content):
    """
    extract time and GFlops from matmult output
    didn't want to mess with the outputing of the matmult programs, so using regex here
    """
    time_match = re.search(r"Time:\s+([0-9.]+)", content)
    gflops_match = re.search(r"GFlopsS:\s+([0-9.]+)", content)

    if time_match and gflops_match:
        return float(time_match.group(1)), float(gflops_match.group(1))
    return None, None


def analyze_file(filename):
    """
    analyze a file and return averages for time and GFlops
    """
    try:
        with open(filename, "r") as f:
            content = f.read()

        # split into individual trials
        trials = content.split("Data dimensions:")
        trials = [t for t in trials if t.strip()]

        times = []
        gflops_list = []

        for trial in trials:
            time, gflops = extract_metrics(trial)
            if time and gflops:
                times.append(time)
                gflops_list.append(gflops)

        if times:
            avg_time = statistics.mean(times)
            avg_gflops = statistics.mean(gflops_list)
            min_time = min(times)
            max_time = max(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0

            return {
                "avg_time": avg_time,
                "avg_gflops": avg_gflops,
                "min_time": min_time,
                "max_time": max_time,
                "std_time": std_time,
                "num_trials": len(times),
            }
        else:
            return None

    except FileNotFoundError:
        return None


print("matmul averaging")
print("=" * 80)

# Define the files to analyze
files = [
    # matmult00 files
    ("matmult00_256x256_mat_16_in.txt", "matmult00", "256x256"),
    ("matmult00_512x512_mat_32_in.txt", "matmult00", "512x512"),
    ("matmult00_1024x1024_mat_64_in.txt", "matmult00", "1024x1024"),
    # matmult01 files
    ("matmult01_256x256_mat_8_in.txt", "matmult01", "256x256"),
    ("matmult01_512x512_mat_16_in.txt", "matmult01", "512x512"),
    ("matmult01_1024x1024_mat_32_in.txt", "matmult01", "1024x1024"),
]

results = {}

print(
    f"{'Matrix':<12} {'Version':<10} {'Trials':<6} {'Avg Time (s)':<12} {'Avg GFlops/S':<12} {'Min Time':<10} {'Max Time':<10}"
)
print("-" * 80)

for filename, version, matrix_size in files:
    filepath = f"results/{filename}"
    data = analyze_file(filepath)

    if data:
        results[(version, matrix_size)] = data
        print(
            f"{matrix_size:<12} {version:<10} {data['num_trials']:<6} {data['avg_time']:.6f}    {data['avg_gflops']:.2f}    {data['min_time']:.6f}  {data['max_time']:.6f}"
        )
    else:
        print(
            f"{matrix_size:<12} {version:<10} {'N/A':<6} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<10}"
        )

print("-" * 80)

# Calculate speedup comparison
print("\nspeedup (matmult01 vs matmult00)")
print("=" * 50)
print(f"{'Matrix Size':<12} {'Speedup':<10} {'GFlops Improvement':<20}")
print("-" * 50)

matrix_sizes = ["256x256", "512x512", "1024x1024"]
for size in matrix_sizes:
    matmult00_data = results.get(("matmult00", size))
    matmult01_data = results.get(("matmult01", size))

    if matmult00_data and matmult01_data:
        # just simple speedup calculation
        time_speedup = matmult00_data["avg_time"] / matmult01_data["avg_time"]
        # turn to percentage
        gflops_improvement = (
            (matmult01_data["avg_gflops"] - matmult00_data["avg_gflops"])
            / matmult00_data["avg_gflops"]
        ) * 100

        print(f"{size:<12} {time_speedup:.2f}x     {gflops_improvement:+.1f}%")
    else:
        print(f"{size:<12} {'N/A':<10} {'N/A':<20}")

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# stdev
for version in ["matmult00", "matmult01"]:
    print(f"\n{version}:")
    for size in matrix_sizes:
        data = results.get((version, size))
        if data and data["std_time"] > 0:
            print(
                f"  {size}: {data['num_trials']} trials, std dev: {data['std_time']:.6f}s"
            )
