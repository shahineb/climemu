"""Benchmark truncated diffusion (warm-start) with DPM-Solver."""
import time
import numpy as np
import climemu
from diffusion import DPMSolverVE

ref = np.load("scripts/reference_sample.npy")

configs = [
    ("DPM-1 full",  2, None,  None),
    ("DPM-1 t=0.7", 2, None,  0.7),
    ("DPM-1 t=0.5", 2, None,  0.5),
    ("DPM-1 t=0.3", 2, None,  0.3),
    ("DPM-1 full",  3, None,  None),
    ("DPM-1 t=0.5", 3, None,  0.5),
    ("DPM-2 full",  2, "dpm2", None),
    ("DPM-2 t=0.5", 2, "dpm2", 0.5),
]

print(f"{'Config':<16} {'Steps':>5} {'Compile':>9} {'Generate':>9} {'RMSE':>10}")
print("-" * 58)

for name, steps, solver, t_start in configs:
    emulator = climemu.build_emulator("MPI-ESM1-2-LR", variables=["tas"])
    emulator.load(which="default")

    t0 = time.perf_counter()
    emulator.compile(n_samples=1, n_steps=steps, solver=solver, sampler_cls=DPMSolverVE, t_start=t_start)
    compile_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    samples = emulator(gmst=2.0, month=6, seed=42)
    gen_t = time.perf_counter() - t0

    rmse = float(np.sqrt(np.mean((np.array(samples[0, 0], dtype=np.float32) - ref[0, 0]) ** 2)))
    print(f"{name:<16} {steps:>5} {compile_t:>8.1f}s {gen_t:>8.3f}s {rmse:>10.2f}")
