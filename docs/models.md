# Model availability

The library provides pretrained emulator weights for several CMIP6 Earth system models (ESMs), with monthly—and, where available, daily—temporal resolution. Each emulator generates joint samples of impact-relevant climate variables on the ESM’s native atmospheric latitude–longitude grid. The ESM, frequency, and variables are selected at build time e.g.
Variables are selected at build time
```python
>>> emulator = earthsampler.build_emulator(esm_name="MPI-ESM1-2-LR",
                                           frequency="daily",
                                           variables=["hurs", "pr"])
```


## Supported ESMs

| ESM | Institution | Monthly | Daily | Grid size (lat × lon) | Resolution |
|-----|-------------|:-------:|:-----:|:---------------------:|:----------------------:|
| MPI-ESM1-2-LR | Max Planck Institute for Meteorology | ✓ | ✓ | 96 × 192 | 1.9° × 1.9° |
| MIROC6 | JAMSTEC / AORI, University of Tokyo / NIES / RIKEN R-CCS | ✓ | — | 128 × 256 | 1.4° × 1.4° |
| ACCESS-ESM1-5 | CSIRO | ✓ | — | 145 × 192 | 1.25° × 1.875° |
| CanESM5 | Canadian Centre for Climate Modelling and Analysis, ECCC | ✓ | — | 64 × 128 | 2.8° × 2.8° |
| IPSL-CM6A-LR | Institut Pierre-Simon Laplace | ✓ | — | 143 × 144 | 1.3° × 2.5° |



## Supported Variables

All emulators generate **anomalies** relative to the corresponding ESM’s pre-industrial climatology. This climatology is available through `emulator.climatology`.


| Short name | Long name | Units | Description |
|------------|-----------|-------|-------------|
| `tas` | Near-surface air temperature | °C | 2 m temperature anomaly |
| `pr` | Precipitation | mm day⁻¹ | Total precipitation rate anomaly |
| `hurs` | Near-surface relative humidity | % | 2 m relative humidity anomaly |
| `sfcWind` | Near-surface wind speed | m s⁻¹ | 10 m wind speed anomaly |


### Availability

| ESM | `tas` | `pr` | `hurs` | `sfcWind` |
|-----|:-----:|:----:|:------:|:---------:|
| MPI-ESM1-2-LR | ✓ | ✓ | ✓ | ✓ |
| MIROC6 | ✓ | ✓ | ✓ | ✓ |
| ACCESS-ESM1-5 | ✓ | ✓ | ✓ | ✓ |
| CanESM5 | ✓ | ✓ | — | — |
| IPSL-CM6A-LR | ✓ | ✓ | — | — |
