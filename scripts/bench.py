"""Quick benchmark: time sample generation (excludes compilation)."""
import time
import climemu

emulator = climemu.build_emulator("MPI-ESM1-2-LR", variables=["tas"])
emulator.load(which="default")

print("Compiling (n_samples=1, n_steps=2)...")
t0 = time.perf_counter()
emulator.compile(n_samples=1, n_steps=2)
print(f"Compilation: {time.perf_counter() - t0:.1f}s")

print("Generating samples...")
t0 = time.perf_counter()
samples = emulator(gmst=2.0, month=6, seed=42, xarray=True)
elapsed = time.perf_counter() - t0
print(f"Generation:  {elapsed:.3f}s")
print(f"Output shape: {dict(samples.dims)}")
