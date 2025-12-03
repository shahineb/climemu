"""HEALPix-specific attention modules.
"""
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class HealPIXAttention(eqx.Module):
    norm: eqx.nn.GroupNorm
    conv: eqx.nn.Conv1d
    proj: eqx.nn.Conv1d
    n_heads: int
    channels_per_head: int
    def __init__(self, channels, n_heads, key=jr.PRNGKey(0)):
        groups = min(max(1, channels // 4), 32) if channels % 4 == 0 else 1
        self.norm = eqx.nn.GroupNorm(groups=groups,
                                     channels=channels,
                                     channelwise_affine=True)
        key, χ = jr.split(key)
        self.conv = eqx.nn.Conv1d(in_channels=channels,
                                  out_channels=channels * 3,
                                  kernel_size=1,
                                  key=χ)
        
        key, χ = jr.split(key)
        self.proj = eqx.nn.Conv1d(in_channels=channels,
                                  out_channels=channels,
                                  kernel_size=1,
                                  key=χ)
        self.n_heads = n_heads
        self.channels_per_head = channels // n_heads

    def __call__(self, x):
        Fx = self.conv(self.norm(x))
        Fx = Fx.reshape(self.n_heads, self.channels_per_head, 3, -1)
        q, k, v = Fx[:, :, 0], Fx[:, :, 1], Fx[:, :, 2]
        α = jnp.einsum("ncq,nck->nqk", q, k / jnp.sqrt(k.shape[1]))
        expα = jnp.exp(α - α.max(axis=2, keepdims=True))
        w = expα / jnp.sum(expα, axis=2, keepdims=True)
        a = jnp.matmul(v, w.mT)
        x = x + self.proj(a.reshape(*x.shape))
        return x