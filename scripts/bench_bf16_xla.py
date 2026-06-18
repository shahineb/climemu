"""Benchmark bfloat16 inference and XLA CPU flags."""
import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true --xla_cpu_enable_fast_math=true"

import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx
import climemu

ref = np.load("scripts/reference_sample.npy")

# --- float32 + Euler 2 steps (baseline for this test) ---
emulator = climemu.build_emulator("MPI-ESM1-2-LR", variables=["tas"])
emulator.load(which="default")
emulator.compile(n_samples=1, n_steps=2, solver=dfx.Euler())

t0 = time.perf_counter()
s_f32 = emulator(gmst=2.0, month=6, seed=42)
t_f32 = time.perf_counter() - t0
rmse_f32 = float(np.sqrt(np.mean((np.array(s_f32[0, 0]) - ref[0, 0]) ** 2)))
print(f"float32 + XLA flags, Euler 2: {t_f32:.3f}s, RMSE={rmse_f32:.2f}")

# --- bfloat16 + Euler 2 steps ---
emulator2 = climemu.build_emulator("MPI-ESM1-2-LR", variables=["tas"])
emulator2.load(which="default")

# Cast NN weights to bfloat16
precursor_orig = emulator2.precursor
# The precursor is a partial wrapping draw_samples_single with nn, schedule, etc.
# We need to cast the nn inside. Access via partial keywords.
nn_bf16 = jax.tree.map(
    lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x,
    precursor_orig.keywords['nn']
)
from functools import partial
from climemu.emulators.bouabid2025 import draw_samples_single
emulator2.precursor = partial(draw_samples_single,
                               nn=nn_bf16,
                               schedule=precursor_orig.keywords['schedule'],
                               output_size=precursor_orig.keywords['output_size'],
                               μ=precursor_orig.keywords['μ'],
                               σ=precursor_orig.keywords['σ'])

emulator2.compile(n_samples=1, n_steps=2, solver=dfx.Euler())

t0 = time.perf_counter()
s_bf16 = emulator2(gmst=2.0, month=6, seed=42)
t_bf16 = time.perf_counter() - t0
rmse_bf16 = float(np.sqrt(np.mean((np.array(s_bf16[0, 0]) - ref[0, 0]) ** 2)))
print(f"bfloat16 + XLA flags, Euler 2: {t_bf16:.3f}s, RMSE={rmse_bf16:.2f}")
print(f"Speedup: {t_f32/t_bf16:.2f}x")
