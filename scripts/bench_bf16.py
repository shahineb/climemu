"""Benchmark bfloat16 vs float32 inference."""
import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true --xla_cpu_enable_fast_math=true"

import time
import numpy as np
import jax.numpy as jnp
import diffrax as dfx
import climemu

ref = np.load("scripts/reference_sample.npy")

configs = [
    ("f32",    None,          dfx.Euler(), 2),
    ("bf16",   jnp.bfloat16, dfx.Euler(), 2),
    ("f32",    None,          dfx.Euler(), 3),
    ("bf16",   jnp.bfloat16, dfx.Euler(), 3),
]

print(f"{'dtype':<6} {'Steps':>5} {'Compile':>9} {'Generate':>9} {'RMSE':>10}")
print("-" * 48)

for dtype_name, dtype, solver, steps in configs:
    emulator = climemu.build_emulator("MPI-ESM1-2-LR", variables=["tas"])
    emulator.load(which="default", dtype=dtype)

    t0 = time.perf_counter()
    emulator.compile(n_samples=1, n_steps=steps, solver=solver)
    compile_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    samples = emulator(gmst=2.0, month=6, seed=42)
    gen_t = time.perf_counter() - t0

    rmse = float(np.sqrt(np.mean((np.array(samples[0, 0], dtype=np.float32) - ref[0, 0]) ** 2)))
    print(f"{dtype_name:<6} {steps:>5} {compile_t:>8.1f}s {gen_t:>8.3f}s {rmse:>10.2f}")
