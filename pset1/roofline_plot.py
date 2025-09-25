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

fig, ax = plt.subplots(1,1,figsize=(5,4), dpi=200)
ax.loglog(ai, roof, linewidth=3, label='roofline')

ax.axvline(0.25, linestyle=':', color='gray', label='dotprod AI = 0.25 FLOP/byte')

measured_N1e6 = {
    'C simple': 50.0,   
    'C unroll': 75.0,
    'MKL': 180.0,
    'python loop': 0.001,
    'numpy.dot': 150.0,
}
measured_N3e8 = {
    'C simple': 50.0,  
    'C unroll': 75.0,
    'MKL': 180.0,
    'python loop': 0.001,
    'numpy.dot': 150.0,
}

for name, gflops in measured_N1e6.items():
    ax.scatter(0.25, gflops, s=60)
    ax.text(0.25*1.05, gflops*1.05, name)

ax.set_xlabel("Arithmetic Intensity (FLOP / byte)")
ax.set_ylabel("Performance (GFLOPS)")
ax.set_title("Roofline Model (peak 200 GFLOPS, BW 30 GB/s)")
# ax.grid(True, which='both', ls='--', alpha=0.5)
ax.legend()
plt.savefig("roofline.png", dpi=200)
plt.show()
