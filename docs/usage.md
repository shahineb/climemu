# Usage guide

## Building an emulator

Use `build_emulator` to instantiate an emulator for a given ESM:

```python
import earthsampler

emulator = earthsampler.build_emulator("MPI-ESM1-2-LR")
```

You can select the temporal frequency and restrict the output variables at build time:

```python
# Daily emulator (currently available for MPI-ESM1-2-LR only)
emulator = earthsampler.build_emulator("MPI-ESM1-2-LR", frequency="daily")

# Only generate temperature and precipitation
emulator = earthsampler.build_emulator("MPI-ESM1-2-LR", variables=["tas", "pr"])
```

See {doc}`models` for the full list of supported ESMs, frequencies and variables.


## Loading weights

`load()` downloads the pretrained weights and piControl climatology from HuggingFace. The download runs once; subsequent calls use the local cache.

```python
emulator.load()
```

After loading, the pre-industrial climatology is available as an `xr.Dataset` that can be used to convert anomalies to absolute values.

```python
>>> emulator.climatology
<xarray.Dataset> Size: 7MB
Dimensions:  (month: 12, lat: 96, lon: 192)
Coordinates:
  * lon      (lon) float64 2kB 0.0 1.875 3.75 5.625 ... 352.5 354.4 356.2 358.1
  * lat      (lat) float64 768B -88.57 -86.72 -84.86 -83.0 ... 84.86 86.72 88.57
  * month    (month) int64 96B 1 2 3 4 5 6 7 8 9 10 11 12
Data variables:
    tas      (month, lat, lon) float64 2MB ...
    pr       (month, lat, lon) float64 2MB ...
    hurs     (month, lat, lon) float64 2MB ...
    sfcWind  (month, lat, lon) float64 2MB ...
```



## Compilation

`compile()` JIT-compiles the JAX diffusion sampler and can take ~1min.

```python
emulator.compile(n_samples=5)
```

**Parameters:**

- **`n_samples`** — number of ensemble members generated at each call. Determines the `member` dimension in the output.
- **`n_steps`** *(default 30)* — number of reverse-diffusion steps. More steps improve sample quality but increase runtime. Fewer than ~10 steps may produce numerical instabilities (NaN/Inf).
- **`batch_size`** *(default 1)* — number of months or days to generate in a single call. See [Batched generation](#batched-generation) below.


## Sampling

### Monthly

```python
emulator = earthsampler.build_emulator("MPI-ESM1-2-LR")
emulator.load()
emulator.compile(n_samples=5)

samples = emulator(gmst=2.0, month=3, seed=42, xarray=True)
```

**Parameters:**

- **`gmst`** — global mean surface temperature anomaly relative to pre-industrial (°C).
- **`month`** — calendar month (1–12).
- **`seed`** — random seed for reproducibility. If `None`, a random seed is drawn.
- **`xarray`** — if `True`, returns an `xr.Dataset`; otherwise a raw JAX array of shape `(n_samples, n_vars, nlat, nlon)`.

```
>>> samples
<xarray.Dataset> Size: 737kB
Dimensions:  (member: 5, lat: 96, lon: 192)
Coordinates:
  * member   (member) int32 20B 1 2 3 4 5
  * lat      (lat) float64 768B -88.57 -86.72 -84.86 ... 84.86 86.72 88.57
  * lon      (lon) float64 2kB 0.0 1.875 3.75 5.625 ... 352.5 354.4 356.2 358.1
Data variables:
    tas      (member, lat, lon) float32 369kB ...
    pr       (member, lat, lon) float32 369kB ...
    hurs     (member, lat, lon) float32 369kB ...
    sfcWind  (member, lat, lon) float32 369kB ...
```

### Daily

Daily emulators take a day-of-year instead of a month. The day can be an integer (1–365) or a date string in `"dd/mm"` format:

```python
emulator = earthsampler.build_emulator("MPI-ESM1-2-LR", frequency="daily")
emulator.load()
emulator.compile(n_samples=5)

# These are equivalent
samples = emulator(gmst=2.0, doy=172, seed=42, xarray=True)
samples = emulator(gmst=2.0, doy="21/06", seed=42, xarray=True)
```

### Converting anomalies to absolute values

All emulators produce **anomalies** relative to the ESM's pre-industrial climatology. To recover absolute fields, select the matching time step from the climatology and add it:

```python
# Monthly: select the month used for sampling
absolute = samples + emulator.climatology.sel(month=3)

# Daily: select the day of year used for sampling
absolute = samples + emulator.climatology.sel(dayofyear=172)
```


## Batched generation

By default the emulator generates samples for a single (GMST, month) or (GMST, day) pair per call. Batched generation produces samples for multiple pairs in one call.

### Setup

Set `batch_size` at compile time to the number of pairs you want per call:

```python
emulator = earthsampler.build_emulator("MPI-ESM1-2-LR")
emulator.load()
emulator.compile(n_samples=3, batch_size=4)
```

### Calling

Pass lists of `gmst` and `month` (or `doy`) whose lengths match `batch_size`. Each element is paired: the first GMST with the first month, the second with the second, and so on.

```python
samples = emulator(
    gmst=[1.0, 1.5, 2.0, 2.5],
    month=[3, 6, 9, 12],
    xarray=True,
)
# batch entry 0 → (gmst=1.0, month=3)
# batch entry 1 → (gmst=1.5, month=6)
# batch entry 2 → (gmst=2.0, month=9)
# batch entry 3 → (gmst=2.5, month=12)
```

A scalar `gmst` is broadcast to all batch elements:

```python
# Same warming level for all four months
samples = emulator(gmst=1.5, month=[3, 6, 9, 12], xarray=True)
# batch entry 0 → (gmst=1.5, month=3)
# batch entry 1 → (gmst=1.5, month=6)
# batch entry 2 → (gmst=1.5, month=9)
# batch entry 3 → (gmst=1.5, month=12)
```


### Output

The output gains an extra `batch` dimension. The input GMST and month (or day) values are stored as non-dimension coordinates on the `batch` axis, so you can always trace each batch element back to the pair that produced it:

```
>>> samples
<xarray.Dataset>
Dimensions:  (batch: 4, member: 3, lat: 96, lon: 192)
Coordinates:
  * batch         (batch) int32 ...
  * member        (member) int32 1 2 3
  * lat           (lat) float64 ...
  * lon           (lon) float64 ...
    month         (batch) int32 3 6 9 12
    gmst_anomaly  (batch) float32 1.0 1.5 2.0 2.5
Data variables:
    tas      (batch, member, lat, lon) float32 ...
    pr       (batch, member, lat, lon) float32 ...
    hurs     (batch, member, lat, lon) float32 ...
    sfcWind  (batch, member, lat, lon) float32 ...
```

You can use these coordinates to select specific pairs:

```python
# Select the batch element for month=6
samples.sel(batch=samples.month == 6)

# Select by GMST
samples.sel(batch=samples.gmst_anomaly == 2.0)
```
