#%%
import glob
import re
import numpy as np
#%% strictly CPU
print("Q1 Results Analysis")
print("=" * 60)

K_values = [1, 5, 10, 50, 100]
all_averages = []  # store averages for each K

for k in K_values:
    filename = f"results/q1_runs_K{k}.txt"
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # find all lines with timing data
        times = []
        bandwidths = []
        gflops_list = []
        
        lines = content.split('\n')
        for line in lines:
            # look for lines that contain the actual data
            if "N:" in line:
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
            
            # store the averages for this K value
            all_averages.append([k, avg_time, avg_bw, avg_gflops])
       
    except FileNotFoundError:
        print(f"file {filename} not found, make sure to run q1_trials_script.sh first.")
        print()

# save the averages to file
if all_averages:
    # convert to numpy array - each row is [K, avg_time, avg_bw, avg_gflops]
    results_array = np.array(all_averages)
    np.savetxt("results/q1_summary.txt", results_array, 
               fmt=["%d", "%.12f", "%.12f", "%.12f"],
               header="K avg_time(s) avg_BW(GB/s) avg_GFLOP/s")
    print("summary saved to results/q1_summary.txt")
    
    # Print what was saved
    print("\nsaved averages:")
    print("K    avg_time(s)    avg_BW(GB/s)    avg_GFLOP/s")
    print("-" * 55)
    for row in all_averages:
        print(f"{row[0]:2d}   {row[1]:12.6f}   {row[2]:12.6f}   {row[3]:12.6f}")
else:
    print("No data found to save.")
 #%%  without unified memory
print("Q2 Results Analysis")
print("=" * 60)
K_values = [1, 5, 10, 50, 100]
scenarios = [1, 2, 3]
all_results = []

for k in K_values:
    for scenario in scenarios:
        filename = f"results/q2_scenario{scenario}_K{k}.txt"
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            # find all lines with timing data
            times = []
            bandwidths = []
            gflops_list = []
            
            lines = content.split('\n')
            for line in lines:
                # look for lines that contain the actual data
                if "N:" in line:
                    numbers = re.findall(r'[\d.]+', line)
                    # print("numbers:", numbers)
                    if len(numbers) >= 4:
                        times.append(float(numbers[1]))
                        bandwidths.append(float(numbers[2]))
                        gflops_list.append(float(numbers[3]))
 
            if times:
                avg_time = sum(times) / len(times)
                avg_bw = sum(bandwidths) / len(bandwidths)
                avg_gflops = sum(gflops_list) / len(gflops_list)
                
                # Scenario descriptions
                scenario_desc = {
                    1: "1 block, 1 thread",
                    2: "1 block, 256 threads", 
                    3: "multiple blocks, 256 threads/block"
                }
                if len(times) == 10:
                    print(times)
                    print()
                
                print(f"K={k}, Scenario {scenario} ({scenario_desc[scenario]}): {len(times)} runs")
                print(f"  avg. time:    {avg_time:.12f} s (min: {min(times):.6f}, max: {max(times):.6f})")
                print(f"  avg. BW:      {avg_bw:.12f} GB/s")
                print(f"  avg. GFLOP/s: {avg_gflops:.12f}")
                print()
                
                # Store results for saving
                all_results.append([k, scenario, avg_time, avg_bw, avg_gflops])
            
        except FileNotFoundError:
            print(f"File {filename} not found")
            print()

# save all results to a single file
if all_results:
    results_array = np.array(all_results)
    np.savetxt("results/q2_summary.txt", results_array, 
               fmt=["%d", "%d", "%.12f", "%.12f", "%.12f"],
               header="K Scenario avg_time(s) avg_BW(GB/s) avg_GFLOP/s")
    print("summary saved to results/q2_summary.txt")
    
    # Also create separate files for each scenario for easier plotting
    for scenario in scenarios:
        scenario_results = [r for r in all_results if r[1] == scenario]
        if scenario_results:
            scenario_array = np.array(scenario_results)
            # Keep only K, avg_time, avg_BW, avg_GFLOP/s (remove scenario column)
            scenario_array = scenario_array[:, [0, 2, 3, 4]]
            np.savetxt(f"results/q2_scenario{scenario}_summary.txt", scenario_array,
                      fmt=["%d", "%.12f", "%.12f", "%.12f"],
                      header="K avg_time(s) avg_BW(GB/s) avg_GFLOP/s")
            print(f"scenario {scenario} summary saved to results/q2_scenario{scenario}_summary.txt")
else:
    print("no Q2 result files found. Make sure to run q2_trials_script.sh first.")
#%% with unified memory
print("Q3 Results Analysis")
print("=" * 60)


K_values = [1, 5, 10, 50, 100]
scenarios = [1, 2, 3]
all_results = []

for k in K_values:
    for scenario in scenarios:
        filename = f"results/q3_scenario{scenario}_K{k}.txt"
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            # find all lines with timing data
            times = []
            bandwidths = []
            gflops_list = []
            
            lines = content.split('\n')
            for line in lines:
                # look for lines that contain the actual data
                if "N:" in line:
                    numbers = re.findall(r'[\d.]+', line)
                   
                    if len(numbers) >= 4:
                        times.append(float(numbers[1]))
                        bandwidths.append(float(numbers[2]))
                        gflops_list.append(float(numbers[3]))
                        
            
            if times:
                avg_time = sum(times) / len(times)
                avg_bw = sum(bandwidths) / len(bandwidths)
                avg_gflops = sum(gflops_list) / len(gflops_list)
                
                # Scenario descriptions
                scenario_desc = {
                    1: "1 block, 1 thread",
                    2: "1 block, 256 threads", 
                    3: "multiple blocks, 256 threads/block"
                }
                if len(times) == 10:
                    print(times)
                    print()
                
                print(f"K={k}, Scenario {scenario} ({scenario_desc[scenario]}): {len(times)} runs")
                print(f"  avg. time:    {avg_time:.12f} s (min: {min(times):.6f}, max: {max(times):.6f})")
                print(f"  avg. BW:      {avg_bw:.12f} GB/s")
                print(f"  avg. GFLOP/s: {avg_gflops:.12f}")
                print()
                
                # Store results for saving
                all_results.append([k, scenario, avg_time, avg_bw, avg_gflops])
            
        except FileNotFoundError:
            print(f"File {filename} not found")
            print()

# Save all results to a single file
if all_results:
    results_array = np.array(all_results)
    np.savetxt("results/q3_summary.txt", results_array, 
               fmt=["%d", "%d", "%.12f", "%.12f", "%.12f"],
               header="K Scenario avg_time(s) avg_BW(GB/s) avg_GFLOP/s")
    print("summary saved to results/q3_summary.txt")
    
    # Also create separate files for each scenario for easier plotting
    for scenario in scenarios:
        scenario_results = [r for r in all_results if r[1] == scenario]
        if scenario_results:
            scenario_array = np.array(scenario_results)
            # Keep only K, avg_time, avg_BW, avg_GFLOP/s (remove scenario column)
            scenario_array = scenario_array[:, [0, 2, 3, 4]]
            np.savetxt(f"results/q3_scenario{scenario}_summary.txt", scenario_array,
                      fmt=["%d", "%.12f", "%.12f", "%.12f"],
                      header="K avg_time(s) avg_BW(GB/s) avg_GFLOP/s")
            print(f"Scenario {scenario} summary saved to results/q3_scenario{scenario}_summary.txt")
else:
    print("no Q3 result files found. Make sure to run q3_trials_script.sh first.")
# %% now, we generate the plots using saved summary files K as a function of time, for each secnario
import matplotlib.pyplot as plt 
q1_summary = np.loadtxt("results/q1_summary.txt", skiprows=1)
q2_summary = np.loadtxt("results/q2_summary.txt", skiprows=1)
q3_summary = np.loadtxt("results/q3_summary.txt", skiprows=1)

s1 = "one block w/ 1 thread"
s2 = "one block w/ 256 threads"
s3 = "multiple blocks w/ 256 threads/block"

print(q1_summary.shape, q2_summary.shape, q3_summary.shape)
cpu_K = q1_summary[:,0]
cpu_time = q1_summary[:,1] 

# non-coalesced memory access
gpu1_K = q2_summary[q2_summary[:,1]==1][:,0]
gpu1_time_scenario1_time = q2_summary[q2_summary[:,1]==1][:,2]
gpu1_time_scenario2_time  = q2_summary[q2_summary[:,1]==2][:,2]
gpu1_time_scenario3_time  = q2_summary[q2_summary[:,1]==3][:,2]


# coalesced memory access
gpu2_K = q3_summary[q3_summary[:,1]==2][:,0]
gpu2_time_scenario1_time = q3_summary[q3_summary[:,1]==1][:,2]
gpu2_time_scenario2_time = q3_summary[q3_summary[:,1]==2][:,2]
gpu2_time_scenario3_time = q3_summary[q3_summary[:,1]==3][:,2]

# figure 1: time vs K for non-coalesced memory access for 3 scenarios, and CPU
fig, ax = plt.subplots(1,1 , figsize=(6,5), dpi=200)
ax.plot(cpu_K, cpu_time, marker='o', label='CPU')
ax.plot(gpu1_K, gpu1_time_scenario1_time, marker='o', label=s1)
ax.plot(gpu1_K, gpu1_time_scenario2_time, marker='o', label=s2)
ax.plot(gpu1_K, gpu1_time_scenario3_time, marker='o', label=s3)
ax.set(xlabel=r'K $\times 10^6$ (number of vector additions) ', ylabel='avg. time (seconds)', yscale='log', ylim = (1e-4, 20)) 
ax.legend(title="without unified memory", fontsize=8)
fig.savefig("./q4_without_unified.jpg", dpi=200)

# figure 2: time vs K for coalesced memory access for 3 scenarios, and CPU
fig, ax = plt.subplots(1,1 , figsize=(6,5), dpi=200)
ax.plot(cpu_K, cpu_time, marker='o', label='CPU')
ax.plot(gpu2_K, gpu2_time_scenario1_time, marker='o', label=s1)
ax.plot(gpu2_K, gpu2_time_scenario2_time, marker='o', label=s2)
ax.plot(gpu2_K, gpu2_time_scenario3_time, marker='o', label=s3)
ax.set(xlabel=r'K $\times 10^6$ (number of vector additions) ', ylabel='avg. time (seconds)', yscale='log', ylim = (1e-4, 20)) 
ax.legend(title="with unified memory", fontsize=8)
fig.savefig("./q4_with_unified.jpg", dpi=200)
# %%
