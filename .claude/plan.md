# Current Plan

**Goal:** Cut CPU inference time from ~5s to <1s without retraining
**Started:** 2026-06-15
**Status:** Complete (core goal achieved)

## Steps
- [x] Baseline benchmark: 2-step Heun = 4.66s, 30-step = 136.2s
- [x] Euler solver parameterization — 2× speedup (2.4s) (commit: 128351d)
- [x] Test bfloat16 inference — dead end on Apple Silicon (commit: 128351d)
- [x] Test XLA CPU flags — dead end, no measurable speedup (commit: 128351d)
- [x] Implement DPM-Solver for VE schedules — 16× speedup (0.29s) (commit: 128351d)
- [x] Implement truncated diffusion (warm-start) — 32× speedup (0.147s) (commit: 128351d)
- [x] Wire new options through compile() API (commit: 128351d)
- [x] All 31 tests pass
- [ ] Evaluate quality across all ESMs (MPI, MIROC, ACCESS, CanESM, IPSL)
- [ ] Decide if DPMSolverVE should be default sampler
- [ ] Add unit tests for DPMSolverVE
- [ ] Add documentation for new compile() kwargs

## Blockers
- None

## Notes
- Best config: `DPMSolverVE` + `t_start=0.5` + `n_steps=2` = 0.147s, RMSE=3.96
- All work is on `speedy` branch, single commit 128351d
- Production quality validation still needed across ESMs
