# Getting started

MIT EarthSampler provides access to pretrained generative climate model emulators for sampling spatially coherent projections of impact-relevant climate variables under global warming. It was developed as part of the [MIT Bringing Computation to the Climate Challenge (BC3) project](https://bc3.mit.edu/).

## Installation
Requires Python ≥3.11. GPU/TPU support is recommended for practical usage.

| Platform   | Command                              |
|------------|--------------------------------------|
| CPU        | `pip install earthsampler`           |
| NVIDIA GPU | `pip install earthsampler[cuda12]`   |
| Google TPU | `pip install earthsampler[tpu]`      |



## Quick example

```python
import earthsampler

# Instantiate a monthly emulator for the MPI model
emulator = earthsampler.build_emulator("MPI-ESM1-2-LR", frequency="monthly")

# Download pretrained weights and compile (~1min)
emulator.load()
emulator.compile(n_samples=1)   # Nb of samples generated at each call

# Generate 5 samples for a given gmst and month
samples = emulator(gmst=2,       # GMST anomaly wrt pre-industrial (°C)
                   month=3,      # Month index (1-12)
                   seed=0,       # Random seed
                   xarray=True)  # Return samples wrapped as a xr.Dataset
```


This produces a `(5, 96, 192) xarray.Dataset` on the MPI-ESM1-2-LR latitude–longitude grid, containing 5 joint samples of 2m temperature, precipitation, 2m relative humidity, and 10m wind speed anomalies that MPI-ESM1-2-LR could plausibly have simulated in March at 2°C of global warming above pre-industrial levels.
```
>>> print(samples)
<xarray.Dataset> Size: 592kB
Dimensions:  (member: 5, lat: 96, lon: 192)
Coordinates:
  * member   (member) int32 8B 1 2
  * lat      (lat) float64 768B -88.57 -86.72 -84.86 -83.0 ... 84.86 86.72 88.57
  * lon      (lon) float64 2kB 0.0 1.875 3.75 5.625 ... 352.5 354.4 356.2 358.1
Data variables:
    tas      (member, lat, lon) float32 147kB ...
    pr       (member, lat, lon) float32 147kB ...
    hurs     (member, lat, lon) float32 147kB ...
    sfcWind  (member, lat, lon) float32 147kB ...


>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(2, 2, figsize=(10, 6))
>>> samples['tas'].isel(member=0).plot(ax=ax[0, 0])
>>> samples['pr'].isel(member=0).plot(ax=ax[0, 1])
>>> samples['hurs'].isel(member=0).plot(ax=ax[1, 0])
>>> samples['sfcWind'].isel(member=0).plot(ax=ax[1, 1])
>>> plt.tight_layout(); plt.show()
```

```{image} img/gettingstarted.png
:width: 100%
:align: center
```



## Citation

If you found this library useful in your work, please cite the accompanying [paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025MS005558)

```bibtex
@article{bouabid2026score,
  title={Score-based generative emulation of impact-relevant Earth system model outputs},
  author={Bouabid, Shahine and Souza, Andre Nogueira and Ferrari, Raffaele},
  journal={Journal of Advances in Modeling Earth Systems},
  volume={18},
  number={3},
  pages={e2025MS005558},
  year={2026},
  publisher={Wiley Online Library}
}
```

(Also consider starring the project on [GitHub](https://github.com/shahineb/mit-earthsampler).)
