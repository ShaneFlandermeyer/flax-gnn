import time
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax_gnn.layers.activations import mish
from typing import Callable, Optional


class GCN(nn.Module):
  """
  Implementation of a Graph Convolution layer using jraph.GraphNetwork

  The implementation is based on the appendix in Kipf et al. (2017) "Semi-Supervised Classification with Graph Convolutional Networks".

  Incorporates global and edge features in the node update step
  """
  embed_dim: int
  normalize: bool = True
  self_edges: bool = False
  node_update_fn: Optional[Callable] = None
  edge_update_fn: Optional[Callable] = None
  
  @nn.compact
  def __call__(self,
               node_features: jax.Array,
               edge_features: jax.Array,
               global_features: jax.Array,
               senders: jax.Array,
               receivers: jax.Array
               ) -> jax.Array:
    num_nodes = node_features.shape[-2]
    num_edges = senders.shape[-1]

    ####################################
    # Node update
    ####################################
    if self.node_update_fn is None:
      W = nn.Dense(self.embed_dim, name='W')
    else:
      W = self.node_update_fn
    node_features = W(node_features)

    ####################################
    # Edge update
    ####################################
    if self.edge_update_fn is None:
      W_e = nn.Dense(self.embed_dim, name='W_e')
    else:
      W_e = self.edge_update_fn
    send_nodes = jnp.take_along_axis(
        node_features, senders[..., None], axis=-2
    )
    if edge_features is None and global_features is None:
      edges = send_nodes
    elif edge_features is not None and global_features is None:
      edges = mish(send_nodes + W_e(edge_features))
    elif edge_features is None and global_features is not None:
      edge_features = global_features.repeat(num_edges, axis=-2)
      edges = mish(send_nodes + W_e(edge_features))
    else:
      edge_features = jnp.concatenate(
          [edge_features, global_features.repeat(num_edges, axis=-2)], axis=-1
      )
      edges = mish(send_nodes + W_e(edge_features))

    #####################################
    # Aggregate edges
    #####################################
    leading_dims = node_features.shape[:-2]
    aggregate_edges = jax.ops.segment_sum
    for _ in range(len(leading_dims)):
      aggregate_edges = jax.vmap(aggregate_edges, in_axes=(0, 0, None))

    if self.normalize:
      in_degree = aggregate_edges(
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
      node_features = node_features + aggregate_edges(
          edges, receivers, num_nodes
      )
    else:
      node_features = aggregate_edges(edges, receivers, num_nodes)

    return dict(
        node_features=node_features,
        edge_features=edge_features,
        global_features=global_features,
        senders=senders,
        receivers=receivers
    )
