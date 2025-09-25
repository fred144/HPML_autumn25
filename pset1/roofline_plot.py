"""
answerin Q2
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

peak_gflops = 200.0   # GFLOPS
bw_gb = 30.0          # GB/s

# arithmetic intensities to plot roofline curve on log-log
ai = np.logspace(-2, 2, 400)  # 0.01 to 100 FLOP/byte
mem_perf = bw_gb * ai / 1.0   # GB/s * FLOP/byte = GFLOP/s ? careful units:
# convert: bw_gb (GB/s) * ai (FLOP/byte) * (1 byte = 1/1e9 GB) -> easier: we maintain AI in FLOP/byte:
# memory-limited GFLOPS = bw_gb * 1e9 bytes/s * AI FLOP/byte / 1e9 = bw_gb * AI
mem_gflops = bw_gb * ai   # since AI in FLOP/byte and bw in GB/s, mem_gflops in GFLOPS

roof = np.minimum(peak_gflops, mem_gflops)

fig, ax = plt.subplots(1,1,figsize=(5,4), dpi=200)
ax.loglog(ai, roof, linewidth=3, label='roofline')
# vertical line for our AI (0.25)
ax.axvline(0.25, linestyle=':', color='gray', label='dotprod AI = 0.25 FLOP/byte')

# now load measured points (if you saved dp1_..., dp2_..., dp3_..., dp4_..., dp5_... files)
# Example: Manually specify measured gflops for each microbenchmark:
# Replace these numbers with your measured GFLOPS (from dp*.txt)
measured = {
    'C simple': 50.0,   # example GFLOPS - **replace with your measured values**
    'C unroll': 75.0,
    'MKL': 180.0,
    'python loop': 0.001,
    'numpy.dot': 150.0,
}
# compute AI for dot product (same for all): 0.25
for name, gflops in measured.items():
    ax.scatter(0.25, gflops, s=60)
    ax.text(0.25*1.05, gflops*1.05, name)

ax.set_xlabel("Arithmetic Intensity (FLOP / byte)")
ax.set_ylabel("Performance (GFLOPS)")
ax.set_title("Roofline Model (peak 200 GFLOPS, BW 30 GB/s)")
# ax.grid(True, which='both', ls='--', alpha=0.5)
ax.legend()
plt.savefig("roofline.png", dpi=200)
plt.show()
