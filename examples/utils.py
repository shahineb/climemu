import os
from typing import Tuple, Any
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from src.diffusion import ContinuousHeunSampler


################################################################################
#                               FILES LOADING                                  #
################################################################################

def load_edges(path: str) -> Tuple[jax.Array]:
    """
    Load precomputed edges
    """
    if os.path.exists(path):
        print(f"Loading precomputed edges from {path}")
        edges_data = jnp.load(path)
        to_healpix = jnp.array(edges_data['to_healpix']).astype(jnp.int32)
        to_latlon = jnp.array(edges_data['to_latlon']).astype(jnp.int32)
        return to_healpix, to_latlon
    else:
        raise FileNotFoundError(f"Edges file not found at {path}.")


def load_normalization(path: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Loads mean and standard deviation for data standardization.
    """
    if os.path.exists(path):
        print(f"Loading normalization statistics from {path}")
        stats = jnp.load(path)
        return stats['μ'], stats['σ']
    else:
        raise FileNotFoundError(f"Normalization stats file not found at {path}.")
    

def load_σmax(path: str) -> float:
    """
    Loads maximum noise level for noise schedule.
    """
    if os.path.exists(path):
        print(f"Loading σmax from {path}")
        σmax = np.load(path)
        return float(σmax)
    else:
        raise FileNotFoundError(f"σmax file not found at {path}.")
    

def load_β(path: str) -> jnp.ndarray:
    """
    Loads pattern scaling coefficients.
    """
    if os.path.exists(path):
        print(f"Loading pattern scaling coefficients from {path}")
        β = jnp.load(path)
        return β
    else:
        raise FileNotFoundError(f"Pattern scaling coefficients file not found at {path}.")



################################################################################
#                               BATCHES PROCESSING                             #
################################################################################


@eqx.filter_jit
def normalize(x: jnp.ndarray, μ: jnp.ndarray, σ: jnp.ndarray) -> jnp.ndarray:
    """Normalize data using mean and standard deviation."""
    return (x - μ) / σ

@eqx.filter_jit
def denormalize(x: jnp.ndarray, μ: jnp.ndarray, σ: jnp.ndarray) -> jnp.ndarray:
    """Denormalize data using mean and standard deviation."""
    return σ * x + μ


################################################################################
#                               SAMPLING FUNCTIONS                             #
################################################################################

def create_sampler(model: eqx.Module, schedule: Any, pattern: jnp.ndarray,
                   μ: jnp.ndarray, σ: jnp.ndarray, output_size: Tuple) -> ContinuousHeunSampler:
    """Create a sampler for a given pattern."""
    context = normalize(pattern, μ[-1], σ[-1])[None, ...]
    def model_with_context(x, t):
        x = jnp.concatenate((x, context), axis=0)
        return model(x, t)
    return ContinuousHeunSampler(schedule, model_with_context, output_size)

@eqx.filter_jit
def draw_samples_single(model: eqx.Module, schedule: Any, pattern: jnp.ndarray,
                        n_samples: int, n_steps: int, μ: jnp.ndarray, σ: jnp.ndarray,
                        output_size: Tuple, key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:
    """Draw samples for a given pattern."""
    sampler = create_sampler(model, schedule, pattern, μ, σ, output_size)
    samples = sampler.sample(n_samples, steps=n_steps, key=key)
    return denormalize(samples, μ[:-1], σ[:-1])


################################################################################
#                               LOGGING FUNCTIONS                              #
################################################################################

def print_parameter_count(model):
    leaves = eqx.filter(model, eqx.is_array)
    n_parameters = sum(jnp.size(x) for x in jax.tree.leaves(leaves))
    print(f"Number of parameters = {n_parameters} \n")