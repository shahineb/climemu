import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Fill in measured values here 
# (time based on 12 months, 2 samples per month; 
# (fid based on June, reference within 0.5 of 2 degrees global warming, 3 samples (diffusion) or 20 samples (consistency))
results = [
    {"label": "Diffusion",             "fid": 279166.7319, "time_per_sample": 219.278},
    {"label": "Consistency (1 step)",  "fid": 399405.4669, "time_per_sample": 0.738},
    {"label": "Consistency (2 steps)", "fid": 199130.1030, "time_per_sample": 1.428},
    {"label": "Consistency (3 steps)", "fid": 194351.8878, "time_per_sample": 2.107},
    # {"label": "Consistency (4 steps)", "fid": 191507.1499, "time_per_sample": 0.825 on GPU},
]

labels = [r["label"] for r in results]
fids   = [r["fid"]            for r in results]
times  = [r["time_per_sample"] for r in results]

x = np.arange(len(labels))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

bars1 = ax1.bar(x, fids, width, color="steelblue")
ax1.set_title("FID (June, T=2)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=15, ha="right")
ax1.set_ylabel("FID")
for bar, val in zip(bars1, fids):
    if val is not None:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)

bars2 = ax2.bar(x, times, width, color="coral")
ax2.set_title("Inference time per sample (s, lower is better)")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15, ha="right")
ax2.set_ylabel("Seconds per sample")
for bar, val in zip(bars2, times):
    if val is not None:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.3f}s", ha="center", va="bottom", fontsize=9)

fig.suptitle("Diffusion vs Consistency Model: Quality and Speed", fontsize=13)
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/comparison.png", dpi=150)
print("Saved outputs/comparison.png")
