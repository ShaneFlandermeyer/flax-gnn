import time
from typing import Optional, Tuple
import flax.linen as nn
from flax_gnn.util import add_self_edges
import jax
import jax.numpy as jnp
from einops import rearrange
from flax_gnn.layers.activations import mish
from typing import Optional, Tuple


class GIN(nn.Module):
  mlp: nn.Module
  epsilon: Optional[float] = None

  @nn.compact
  def __call__(self,
               nodes: jnp.ndarray,
               edges: jnp.ndarray,
               globals_: jnp.ndarray,
               senders: jnp.ndarray,
               receivers: jnp.ndarray
               ) -> jnp.ndarray:
    num_nodes = nodes.shape[-2]
    num_edges = senders.shape[-1]

    sent_attributes = jnp.take_along_axis(nodes, senders[..., None], axis=-2)
    received_attributes = jnp.take_along_axis(
        nodes, receivers[..., None], axis=-2
    )

    # Edge update
    if edges is not None:
      embed_dim = sent_attributes.shape[-1]
      W_e = nn.Dense(embed_dim, name='W_e')
      edges = mish(sent_attributes + W_e(edges))
    else:
      edges = sent_attributes

    # Vmap aggregation function over batch dims
    leading_dims = nodes.shape[:-2]
    edge_aggr = jax.ops.segment_sum
    for _ in range(len(leading_dims)):
      edge_aggr = jax.vmap(edge_aggr, in_axes=(0, 0, None))
    received_attributes = edge_aggr(edges, receivers, num_nodes)

    # Node update
    if self.epsilon is None:
      epsilon = self.param('epsilon', nn.initializers.zeros, (1, 1))
    else:
      epsilon = self.epsilon
    epsilon = jnp.tile(epsilon, (*nodes.shape[:-2], 1, 1))
    nodes = (1 + epsilon) * nodes + received_attributes
    
    
    if globals_ is not None:
      global_node_attributes = globals_.repeat(num_nodes, axis=-2)
      nodes = jnp.concatenate([nodes, global_node_attributes], axis=-1)
    
    nodes = self.mlp(nodes)

    return nodes
