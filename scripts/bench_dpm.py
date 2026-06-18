"""Benchmark DPM-Solver vs Euler."""
import time
import numpy as np
import diffrax as dfx
import climemu
from diffusion import DPMSolverVE

ref = np.load("scripts/reference_sample.npy")

configs = [
    ("Euler",  2, None,        dfx.Euler()),
    ("DPM-1",  2, DPMSolverVE, None),
    ("DPM-1",  3, DPMSolverVE, None),
    ("DPM-2",  2, DPMSolverVE, "dpm2"),
    ("DPM-2",  3, DPMSolverVE, "dpm2"),
    ("Euler",  5, None,        dfx.Euler()),
    ("DPM-1",  5, DPMSolverVE, None),
    ("DPM-2",  5, DPMSolverVE, "dpm2"),
]

print(f"{'Solver':<8} {'Steps':>5} {'Compile':>9} {'Generate':>9} {'RMSE':>10}")
print("-" * 50)

for name, steps, sampler_cls, solver in configs:
    emulator = climemu.build_emulator("MPI-ESM1-2-LR", variables=["tas"])
    emulator.load(which="default")

    t0 = time.perf_counter()
    emulator.compile(n_samples=1, n_steps=steps, solver=solver, sampler_cls=sampler_cls)
    compile_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    samples = emulator(gmst=2.0, month=6, seed=42)
    gen_t = time.perf_counter() - t0

    rmse = float(np.sqrt(np.mean((np.array(samples[0, 0], dtype=np.float32) - ref[0, 0]) ** 2)))
    print(f"{name:<8} {steps:>5} {compile_t:>8.1f}s {gen_t:>8.3f}s {rmse:>10.2f}")
