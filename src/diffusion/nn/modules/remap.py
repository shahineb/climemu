"""Remapping modules for transforming between different grid types.

This module provides neural network layers for remapping data between different
grid types (e.g., HEALPix to lat-lon) using attention-based graph operations.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class BipartiteRemap(eqx.Module):
    """Remapping between two sets of nodes.
    
    This module implements a bipartite graph neural network for remapping
    features between two different sets of nodes (e.g., different grid types).

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        edges: Edge connectivity matrix of shape (n_edges, 2) where each row [i,j]
               represents an edge from source node j to target node i
        n_out_nodes: Number of target nodes
        linear: Linear layer for feature transformation
        prelu: PReLU activation for attention computation
        weights: Learnable attention weights for feature transformation

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        edges: Edge connectivity matrix of shape (n_edges, 2)
        key: PRNG key for initialization
    """
    in_channels: int
    out_channels: int
    edges: jax.Array
    n_out_nodes: int
    linear: eqx.nn.Linear
    prelu: eqx.nn.PReLU
    weights: jax.Array

    def __init__(self, in_channels, out_channels, edges, key=jr.PRNGKey(0)):
        χ1, χ2 = jr.split(key, 2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = eqx.nn.Linear(in_channels, out_channels, key=χ1)
        self.weights = jr.normal(χ2, (out_channels,)) * jnp.sqrt(2.0 / (out_channels + 1))
        self.prelu = eqx.nn.PReLU()
        self.edges = edges
        self.n_out_nodes = int(jax.device_get(edges[:, 0].max()).item()) + 1

    def __call__(self, x):
        # Pass every vertex/grid-cell through the linear layer
        x = x.T  # (in_nodes, in_channels)
        Wx = jax.vmap(self.linear)(x)  # (in_nodes, out_channels)

        # Extract features for each edge i <- j
        tgt_nodes = self.edges[:, 0]  # (edges,)
        src_nodes = self.edges[:, 1]  # (edges,)
        Wxj = Wx[src_nodes]  # (edges, out_channels)

        # Compute edges weights
        a = jnp.einsum("e f, f -> e", Wxj, self.weights)  # (edges,)
        expa = jnp.exp(self.prelu(a))[..., None]  # (edges, 1)

        # Average target node features over edges attention
        ΣexpaWx = jax.ops.segment_sum(expa * Wxj, tgt_nodes, self.n_out_nodes)   # (out_nodes, out_channels)
        Σexpa = jax.ops.segment_sum(expa, tgt_nodes, self.n_out_nodes)  # (out_nodes, 1)
        EWx = ΣexpaWx / jnp.where(Σexpa == 0.0, 1.0, Σexpa)  # (out_nodes, out_channels)

        # Reshape
        y = EWx.T  # (out_channels, out_nodes)
        return y