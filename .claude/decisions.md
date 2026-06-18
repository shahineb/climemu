# Decisions Log

> Document architecture, model, statistical, and design decisions with rationale.

---

## 2026-06-18 - DPM-Solver for VE Schedules

**Type:** Architecture / Model Choice
**Status:** Accepted

### Context
CPU inference with the default Heun ODE sampler was 4.66s for 2 steps — too slow for interactive use. Goal was <1s.

### Decision
Implement a DPM-Solver adapted for variance-exploding (VE) noise schedules as a new sampler class `DPMSolverVE`.

### Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| Euler solver | Simple, 2× speedup | Still 2.4s — not enough |
| bfloat16 inference | Standard optimization | No speedup on Apple Silicon CPU |
| XLA CPU flags | Zero code change | No measurable effect |
| DPM-Solver (chosen) | 16× speedup, proven method | New code to maintain |

### Rationale
DPM-Solver is a well-established fast sampler for diffusion models. Adapting it for VE schedules gave 16× speedup (0.29s). Combined with truncated diffusion (warm-start at t_start=0.5), achieves 32× speedup (0.147s).

### Consequences
- New file `src/diffusion/samplers/dpm_solver.py` to maintain
- Quality validation needed across all ESMs
- Compile-time API expanded with new kwargs

### Related Files
- `src/diffusion/samplers/dpm_solver.py` — Implementation
- `src/climemu/emulators/bouabid2025.py` — API integration

---

## 2026-06-18 - Truncated Diffusion (Warm-Start)

**Type:** Model Choice
**Status:** Accepted (pending quality validation)

### Context
DPM-Solver alone gave 0.29s. Could we start the diffusion process partway through (skip the noisiest portion) for additional speedup?

### Decision
Add `t_start` parameter to `compile()` that truncates the diffusion schedule. Default is 1.0 (full schedule). Setting t_start=0.5 starts from medium noise instead of maximum noise.

### Rationale
The early (high-noise) portion of the reverse diffusion process contributes less to fine structure. Skipping it halves generation time and, empirically, RMSE=3.96 vs 30-step reference is acceptable.

### Consequences
- Trade-off between speed and quality that needs domain expert judgment
- RMSE threshold for acceptability not yet established
- Only validated on MPI-ESM1-2-LR so far

### Related Files
- `src/climemu/emulators/bouabid2025.py` — t_start parameter in compile()
- `src/diffusion/samplers/dpm_solver.py` — Warm-start support

---

## 2026-06-18 - Extended compile() API

**Type:** Architecture
**Status:** Accepted

### Context
New optimization strategies required new parameters (solver choice, sampler class, dtype, t_start).

### Decision
Add optional kwargs to `compile()`: `solver`, `sampler_cls`, `dtype`, `t_start`. All default to original behavior.

### Rationale
Keeps the simple path simple (`compile(n_samples=5)` works as before) while enabling power users to tune performance. Compile-time is the right place because these affect JIT compilation.

### Consequences
- Backward compatible — no existing code breaks
- API surface grows, needs documentation

### Related Files
- `src/climemu/emulators/bouabid2025.py` — compile() method

---

## 2026-06-15 - Initialize Context Management
**Type:** Architecture
**Status:** Accepted

### Context
Setting up a new project with comprehensive tracking.

### Decision
Use the standard context management system with scratchpad, plan, handoff, errors, decisions, codebase-map, and references files.

### Rationale
Enables session continuity, traceable decisions, and verifiable sources.

---
