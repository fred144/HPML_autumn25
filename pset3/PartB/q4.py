#%%
import glob
import re
import numpy as np
print("Q1 Results Analysis")
print("=" * 60)
K = [1, 5, 10, 50, 100]
for k in K:
    filename = f"results/q1_runs_K{k}.txt"
    try:
        files_exist = True
        with open(filename, 'r') as f:
            content = f.read()
        
        # Find all lines with timing data
        times = []
        bandwidths = []
        gflops_list = []
        
        lines = content.split('\n')
        for line in lines:
            # Look for lines that contain the actual data (have large numbers)
            if any(x in line for x in ['1000000', '5000000', '10000000', '50000000', '100000000']):
                numbers = re.findall(r'[\d.]+', line)
                if len(numbers) >= 4:
                    times.append(float(numbers[1]))
                    bandwidths.append(float(numbers[2]))
                    gflops_list.append(float(numbers[3]))
        
        if times:
            avg_time = sum(times) / len(times)
            avg_bw = sum(bandwidths) / len(bandwidths)
            avg_gflops = sum(gflops_list) / len(gflops_list)
            
            print(f"K={k}: {len(times)} runs")
            print(f"  avg. time:    {avg_time:.12f} s (min: {min(times):.6f}, max: {max(times):.6f})")
            print(f"  avg. BW:      {avg_bw:.12f} GB/s")
            print(f"  avg. GFLOP/s: {avg_gflops:.12f}")
            print()
        
       
            
    except FileNotFoundError:
        files_exist = False
        print(f"file {filename} not found, make sure to run q1_trials_script.sh first.")
        print()

 #  now save as a txt file, with K avge time, avg bw, avg gflops fr each row
if files_exist:
    # concatenate all results into a single file
    out  = np.concatenate((
        np.array(K).reshape(-1,1), np.array(times).reshape(-1,1),
        np.array(bandwidths).reshape(-1,1), np.array(gflops_list).reshape(-1,1)
    ), axis=1)
    np.savetxt("results/q1_summary.txt", out, fmt=["%d", "%.12f", "%.12f", "%.12f"],
               header="K avg_time(s) avg_BW(GB/s) avg_GFLOP/s")
    print("summary saved to results/q1_summary.txt")
    

