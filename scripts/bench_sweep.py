"""Benchmark sweep: Euler vs Heun at various step counts."""
import time
import numpy as np
import diffrax as dfx
import climemu

ref = np.load("scripts/reference_sample.npy")

emulator = climemu.build_emulator("MPI-ESM1-2-LR", variables=["tas"])
emulator.load(which="default")

configs = [
    ("Heun", dfx.Heun(), 2),
    ("Euler", dfx.Euler(), 2),
    ("Euler", dfx.Euler(), 3),
    ("Euler", dfx.Euler(), 5),
    ("Heun", dfx.Heun(), 3),
    ("Heun", dfx.Heun(), 5),
]

print(f"{'Solver':<8} {'Steps':>5} {'Evals':>5} {'Compile':>9} {'Generate':>9} {'RMSE':>10}")
print("-" * 55)

for solver_name, solver, steps in configs:
    evals = steps * (2 if solver_name == "Heun" else 1)

    t0 = time.perf_counter()
    emulator.compile(n_samples=1, n_steps=steps, solver=solver)
    compile_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    samples = emulator(gmst=2.0, month=6, seed=42)
    gen_t = time.perf_counter() - t0

    # RMSE against reference (tas is index 0 in full 4-var reference)
    rmse = float(np.sqrt(np.mean((np.array(samples[0, 0]) - ref[0, 0]) ** 2)))

    print(f"{solver_name:<8} {steps:>5} {evals:>5} {compile_t:>8.1f}s {gen_t:>8.3f}s {rmse:>10.4f}")
