import time
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax_gnn.layers.activations import mish
from typing import Callable


class GCN(nn.Module):
  """
  Implementation of a Graph Convolution layer using jraph.GraphNetwork

  The implementation is based on the appendix in Kipf et al. (2017) "Semi-Supervised Classification with Graph Convolutional Networks".

  Incorporates global and edge features in the node update step
  """
  embed_dim: int
  normalize: bool = True
  self_edges: bool = False
  kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()

  @nn.compact
  def __call__(self,
               nodes: jax.Array,
               edge_features: jax.Array,
               global_features: jax.Array,
               senders: jax.Array,
               receivers: jax.Array
               ) -> jax.Array:
    num_nodes = nodes.shape[-2]
    num_edges = senders.shape[-1]

    ####################################
    # Node update
    ####################################
    W = nn.Dense(self.embed_dim, kernel_init=self.kernel_init, name='W')
    nodes = W(nodes)

    ####################################
    # Edge update
    ####################################
    W_e = nn.Dense(self.embed_dim, kernel_init=self.kernel_init, name='W_e')
    sent_attributes = jnp.take_along_axis(nodes, senders[..., None], axis=-2)
    if edge_features is None and global_features is None:
      edges = sent_attributes
    elif edge_features is not None and global_features is None:
      edges = mish(sent_attributes + W_e(edge_features))
    elif edge_features is None and global_features is not None:
      edge_features = global_features.repeat(num_edges, axis=-2)
      edges = mish(sent_attributes + W_e(edge_features))
    else:
      edge_features = jnp.concatenate(
          [edge_features, global_features.repeat(num_edges, axis=-2)], axis=-1
      )
      edges = mish(sent_attributes + W_e(edge_features))

    #####################################
    # Aggregate edges
    #####################################
    leading_dims = nodes.shape[:-2]
    edge_aggr = jax.ops.segment_sum
    for _ in range(len(leading_dims)):
      edge_aggr = jax.vmap(edge_aggr, in_axes=(0, 0, None))

    if self.normalize:
      in_degree = edge_aggr(
          jnp.ones_like(receivers), receivers, num_nodes
      ).astype(float)
      send_degree = jnp.take_along_axis(in_degree, senders, axis=-1)
      recv_degree = jnp.take_along_axis(in_degree, receivers, axis=-1)
      if self.self_edges:
        send_degree += 1
        recv_degree += 1
      edges *= jax.lax.rsqrt(
          send_degree.clip(1, None) * recv_degree.clip(1, None)
      )[..., None]

    if self.self_edges:
      nodes = edge_aggr(edges, receivers, num_nodes) + nodes
    else:
      nodes = edge_aggr(edges, receivers, num_nodes)

    return nodes
