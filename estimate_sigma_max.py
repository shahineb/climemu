# %%
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt

# %%
x_shape = (5, 20, 30)
μ = jnp.zeros(x_shape)
σ = jnp.ones(x_shape)
ctx_size = 1
search_interval = (0, 200)
seed = 42
α = 0.05


σmax_low, σmax_high = search_interval
σmax = 0.5 * (σmax_low + σmax_high)
max_montecarlo = 10000
batch_size = 16
tgt_pow = 0.1
tol = 0.001 + 1.96 * np.sqrt(tgt_pow * (1 - tgt_pow)  / max_montecarlo)
key = jr.PRNGKey(seed)

# %%
ds = xr.open_dataset("/Users/shahine/Desktop/tmp/from_pattern_scaling.nc")

# %%
x = ds.isel(time=slice(0, 100), member=0).to_array().values
x = x.transpose(1, 0, 2, 3).reshape(100, -1)

# %%
U, S, Vt = jnp.linalg.svd(x, full_matrices=False)
eof1 = Vt[0] *  (S[0] / np.sqrt(x.shape[0] - 1))
eof1 = eof1.reshape(4, 145, 192)


# %%
n_montecarlo = 100
rejections = 0

with tqdm(total=n_montecarlo) as pbar:
    pbar.set_description(f"Estimating power for σmax = {σmax:.1f}")
    for _ in range(n_montecarlo):
        # Draw batch and flatten
        key, subkey = jr.split(key)
        x = jr.normal(subkey, shape=(batch_size, *x_shape))
        x0 = np.array(x.reshape(batch_size, -1))

        # Compute leading EOF
        u, Σ, vT = jnp.linalg.svd(x0, full_matrices=False)
        eof0 =  (Σ[0] / np.sqrt(batch_size - 1)) * vT[0]

        # Add noise and perform test
        eofn = eof0 + σmax * np.random.randn(*eof0.shape)
        _, pvalue = kstest(eofn, "norm", args=(0, σmax))
        rejections += (pvalue < α)
        _ = pbar.update(1)
power= rejections / n_montecarlo
