# %%
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from functools import partial
import matplotlib.pyplot as plt

def mapping_func(iters, σ):
    q = 2 # hyperparameter >1
    d = 10000/8

    def n(σ):
        b = 1
        k = 8
        # return 1
        # return 1+k
        return 1 + k/(1+jnp.exp(b*σ))

    return jnp.maximum(0, (1 - q**(-jnp.ceil(iters/d)) * n(σ)) * σ)

# Mapping Function visualization:
# σt = 0.8
# plt.plot(range(3000), [mapping_func(iters, σt)/ σt for iters in range(3000)], color='red')
# plt.xlabel('nb iterations')
# plt.ylabel(f'σr/σt')
# plt.savefig("mapping_func.png", dpi=150)
# plt.show()


def weighting_function(σt, σr):
    return 1/jnp.maximum(1e-5, σt)
    # return 1/jnp.maximum(1e-5, σt - σr)

# %%
def difference_minimizing_single_loss(model, ctx_size, x, σ, iters, key):
    x0, ctx = x[:-ctx_size, ...], x[-ctx_size:, ...]
    # Get target noise level σr
    σr = mapping_func(iters, σ)

    # Add Gaussian noise to target and rescale
    ε = jr.normal(key, x0.shape)
    x̃ = x0 + σ * ε
    x̃r = x0 + σr * ε
    x̃_rescaled = x̃ / (1 + σ**2)**0.5
    x̃r_rescaled = x̃r / (1 + σr**2)**0.5

    # Concatenate context
    x̃_rescaled = jnp.concatenate([x̃_rescaled, ctx], axis=0)
    x̃r_rescaled = jnp.concatenate([x̃r_rescaled, ctx], axis=0)

    # Forward pass through denoiser
    denoised_x̃ = model(x̃_rescaled, σ)
    denoised_x̃r = jax.lax.stop_gradient(model(x̃r_rescaled, σr))

    # Compute loss with custom weighting
    c = 1e-3
    mse = jnp.mean((denoised_x̃ - denoised_x̃r) ** 2)
    loss = jnp.sqrt(mse + c**2) - c
    loss = weighting_function(σ, σr) * loss
    return loss, mse


# %%
@eqx.filter_jit
def difference_minimizing_batch_loss(model, ctx_size, schedule, x, iters, key):
    batch_size = x.shape[0]
    χ1, χ2 = jr.split(key)

    # Vectorize single-sample loss over batch
    # in_axes: (None, 0, 0, None, 0) means vectorize over x, σ, and keys, but NOT ctx_size, iters
    L = jax.vmap(partial(difference_minimizing_single_loss, model), in_axes=(None, 0, 0, None, 0))

    # Sample noise scales for each batch element
    keys = jr.split(χ1, batch_size)
    σ = jax.vmap(schedule.sample_σ)(keys)

    # Compute mean batch loss
    keys = jr.split(χ2, batch_size)
    batch_loss, batch_mse = L(ctx_size, x, σ, iters, keys)
    return batch_loss.mean(), batch_mse.mean()


@eqx.filter_jit
def difference_minimizing_make_step(model, ctx_size, schedule, x, iters, key, opt_state, opt_update):
    """
    Performs a single optimization step for difference minimizing
    """
    # Compute loss and gradients with respect to model parameters
    loss_function = eqx.filter_value_and_grad(difference_minimizing_batch_loss, has_aux=True)
    (loss, mse), grads = loss_function(model, ctx_size, schedule, x, iters, key)

    # Compute gradient norm (for logging)
    grad_norm = compute_grad_norm(grads)

    # Update optimizer state and model parameters
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    # Split PRNGKey for next step
    key, _ = jr.split(key)
    return loss, mse, model, key, opt_state, grad_norm


@eqx.filter_jit
def compute_grad_norm(grads):
    """
    Computes the L2 norm of gradients in a pytree.
    """
    # Flatten the gradient pytree and sum the squared norms
    squared_norms = [jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads) if g is not None]
    return jnp.sqrt(jnp.sum(jnp.array(squared_norms)))