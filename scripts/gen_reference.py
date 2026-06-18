"""Generate a 30-step Heun reference sample and save to disk."""
import time
import numpy as np
import climemu

emulator = climemu.build_emulator("MPI-ESM1-2-LR")
emulator.load(which="default")

print("Compiling (n_samples=1, n_steps=30, Heun)...")
t0 = time.perf_counter()
emulator.compile(n_samples=1, n_steps=30)
print(f"Compilation: {time.perf_counter() - t0:.1f}s")

print("Generating reference sample...")
t0 = time.perf_counter()
samples = emulator(gmst=2.0, month=6, seed=42)
elapsed = time.perf_counter() - t0
print(f"Generation: {elapsed:.1f}s")

np.save("scripts/reference_sample.npy", np.array(samples))
print(f"Saved reference_sample.npy, shape={samples.shape}")
