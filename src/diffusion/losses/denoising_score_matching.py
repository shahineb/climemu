"""
Implements single-sample and batch denoising score matching losses to train the emulator.

Code adapted from: https://github.com/sandreza/JaxDiffusion/blob/main/jaxdiffusion/losses/residual_loss.py
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from functools import partial


@eqx.filter_jit
def denoising_single_loss(model, ctx_size, x, σ, key):
    """
    Computes the denoising score matching loss for a single sample.

    Args:
        model: The denoising model to train.
        ctx_size: Number of context timesteps at the end of x.
        x: Input array.
        σ: Noise scale (float).
        key: JAX PRNGKey for noise generation.

    Returns:
        loss: Scalar loss value.
    """
    # Split input into target and context
    x0, ctx = x[:-ctx_size, ...], x[-ctx_size:, ...]

    # Add Gaussian noise to target and rescale
    ε = jr.normal(key, x0.shape)
    x̃ = x0 + σ * ε
    x̃_rescaled = x̃ / (1 + σ)

    # Concatenate context
    x̃_with_ctx = jnp.concatenate([x̃_rescaled, ctx], axis=0)

    # Forward pass through denoiser
    denoised_x̃ = model(x̃_with_ctx, σ)

    # Compute loss with custom weighting
    λ = 1 / (1 + jnp.log1p(σ))
    loss = λ * jnp.mean((denoised_x̃ - x0) ** 2)
    return loss


@eqx.filter_jit
def denoising_batch_loss(model, ctx_size, schedule, x, key):
    """
    Computes the average denoising score matching loss over a batch.

    Args:
        model: The denoising model to train.
        ctx_size: Number of context timesteps.
        schedule: Object with sample_σ method for noise sampling.
        x: Batch input array, shape (batch_size, ...).
        key: JAX PRNGKey for randomness.

    Returns:
        batch_loss: Scalar mean loss over the batch.
    """
    batch_size = x.shape[0]
    χ1, χ2 = jr.split(key)

    # Vectorize single-sample loss over batc
    L = jax.vmap(partial(denoising_single_loss, model, ctx_size))

    # Sample noise scales for each batch element
    keys = jr.split(χ1, batch_size)
    σ = jax.vmap(schedule.sample_σ)(keys)

    # Compute mean batch loss
    keys = jr.split(χ2, batch_size)
    batch_loss = L(x, σ, keys).mean()
    return batch_loss


@eqx.filter_jit
def denoising_make_step(model, ctx_size, schedule, x, key, opt_state, opt_update):
    """
    Performs a single optimization step for denoising score matching.

    Args:
        model: The denoising model.
        ctx_size: Number of context timesteps.
        schedule: Noise schedule object.
        x: Batch input array.
        key: JAX PRNGKey.
        opt_state: Optimizer state.
        opt_update: Optimizer update function.

    Returns:
        loss: Scalar loss value.
        model: Updated model.
        key: Updated PRNGKey.
        opt_state: Updated optimizer state.
        grad_norm: Norm of the gradients.
    """
    # Compute loss and gradients with respect to model parameters
    loss_function = eqx.filter_value_and_grad(denoising_batch_loss)
    loss, grads = loss_function(model, ctx_size, schedule, x, key)

    # Compute gradient norm (for loggin)
    grad_norm = compute_grad_norm(grads)

    # Update optimizer state and model parameters
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    # Split PRNGKey for next step
    key, _ = jr.split(key)
    return loss, model, key, opt_state, grad_norm


@eqx.filter_jit
def compute_grad_norm(grads):
    """
    Computes the L2 norm of gradients in a pytree.
    """
    # Flatten the gradient pytree and sum the squared norms
    squared_norms = [jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads) if g is not None]
    return jnp.sqrt(jnp.sum(jnp.array(squared_norms)))