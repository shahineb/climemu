# %%
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import xarray as xr
from scipy.stats import kstest
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
batch_size = 1 << 13
tgt_pow = 0.1
tol = 0.001 + 1.96 * np.sqrt(tgt_pow * (1 - tgt_pow)  / max_montecarlo)
key = jr.PRNGKey(seed)

# %%
ds = xr.open_dataset("/Users/shahine/Desktop/tmp/from_pattern_scaling.nc")
x = ds.isel(time=slice(0, batch_size), member=0).to_array().values
# x = x.transpose(1, 0, 2, 3).reshape(batch_size, -1)
# x = (x - x.mean(axis=0)) / x.std(axis=0)
# x = jr.normal(key, shape=(batch_size, 4 * 145 * 192))



# %%
xc = x - x.mean(axis=0)
G = xc @ xc.T
eigvals, eigvecs = jnp.linalg.eigh(G)
lambda_max = eigvals[-1] / (batch_size - 1)
σmax = jnp.sqrt(lambda_max)
print("Estimated σmax : ", σmax)


# %%

# Leading eigenvalue: 7316.19
# EOF :  [-0.0010555  -0.00107857 -0.0010427  -0.00103347


# %%
n_montecarlo = 100
rejections = 0
σmax = 50
with tqdm(total=n_montecarlo) as pbar:
    pbar.set_description(f"Estimating power for σmax = {σmax:.1f}")
    for _ in range(n_montecarlo):
        # Draw batch and flatten
        key, subkey = jr.split(key)
        x = np.array(x.reshape(batch_size, -1))

        # Compute leading EOF
        x = x - x.mean(axis=0)
        G = x @ x.T 
        eigvals, eigvecs = jnp.linalg.eigh(G)
        eigvecs1 = eigvecs[:, -1]
        x0 = (x.T @ eigvecs1) / jnp.sqrt(batch_size - 1)

        # Add noise and perform test
        xn = x0 + σmax * np.random.randn(*x0.shape)
        _, pvalue = kstest(xn, "norm", args=(0, σmax))
        rejections += (pvalue < α)
        _ = pbar.update(1)
power = rejections / n_montecarlo
print("Power = ", power)

# %%
