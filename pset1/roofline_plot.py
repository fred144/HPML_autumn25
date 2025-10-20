"""
answerin Q2
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

# 
peak_gflops = 200.0   # GFLOPS
bw_gb = 30.0          # GB/s

# arithmetic intensities to plot roofline curve on log-log
ai = np.logspace(-2, 2, 400)  # 0.01 to 100 FLOP/byte
mem_perf = bw_gb * ai / 1.0   #
mem_gflops = bw_gb * ai   #
roof = np.minimum(peak_gflops, mem_gflops)

# plot a vertical line where peak_gflops and mem_gflops intersect
# peak_gflops = bw_gb * ai_intersect
ai_intersect = peak_gflops / bw_gb

# calculate arithmetic intensity for dot product
# 2N FLOP, 2N float32 (4 bytes each) = 8N bytes
# AI = 2N / 8N = 0.25  # FLOP / byte
float_size = 4  # bytes
N = 1000000
dotprod_ai = 2 * N / (2 * N * float_size)  # FLOP / byte


fig, ax = plt.subplots(1,1,figsize=(5,4), dpi=200)
ax.plot(ai, roof, linewidth=3, color="k")
ax.axvline(ai_intersect, linestyle='--', color='gray')


# measured quanties in GFLOPS or GFLOP/sec
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
measured_N1e6 = {
    'C simple': 1.324462,   
    'C unroll': 3.540637,
    'MKL': 23.522858,
    'python loop': 0.006028,
    'numpy.dot': 5.714158,
} 
measured_N3e8 = {
    'C simple': 1.321051,  
    'C unroll': 2.288799,
    'MKL': 11.076449,
    'python loop': 0.006441,
    'numpy.dot': 2.994503,
}

for i, (name, gflops) in enumerate(measured_N1e6.items()):
    ax.scatter(dotprod_ai, gflops, s=60, marker='o', label=name, color=colors[i], alpha=0.6)

for i, (name, gflops) in enumerate(measured_N3e8.items()):
    ax.scatter(dotprod_ai, gflops, s=60, marker='x', label=name, color=colors[i])


ax.set(xscale='log', yscale='log', xlabel="Arithmetic Intensity (FLOP / byte)",
       ylabel="Performance (GFLOPS)", title="Roofline Model (peak 200 GFLOPS, BW 30 GB/s)")
# ax.grid(True, which='both', ls='--', alpha=0.5)
ax.legend(fontsize=8, ncols=2, title="N=1e6, N=3e8")
ax.set_xlim(0.01, 100)
ax.set_ylim(0.001, 300)

ax.text(0.4, 0.5,  'memory-bound', color='gray', alpha=0.5, fontsize=8, transform=ax.transAxes)
ax.text(0.75, 0.5, 'compute-bound', color='gray', alpha=0.5, fontsize=8, transform=ax.transAxes)
# add DRAM and CPU FLOPS
ax.text(0.02, 1, 'DRAM 30 GB/s', rotation=30, color='gray',fontsize=8)
ax.text(15, 120, '200 GFLOPS', color='gray',  fontsize=8)

plt.savefig("roofline.png", dpi=200)
plt.show()

# %%
