import time
from typing import Optional, Tuple
import flax.linen as nn
import flax_gnn.util
import jax
import jax.numpy as jnp
from einops import rearrange
from flax_gnn.layers.activations import mish


class GATv2(nn.Module):
  """
  Implementation of GATv2 using jraph.GraphNetwork.

  The implementation is based on the appendix in Battaglia et al. (2018) "Relational inductive biases, deep learning, and graph networks".

  Incorporates global and edge features as in Wang2021
  """
  embed_dim: int
  num_heads: int
  share_weights: bool = True
  add_self_edges: bool = False
  kernel_init: nn.initializers.Initializer = nn.initializers.xavier_normal()

  @nn.compact
  def __call__(self,
               node_features: jax.Array,
               edge_features: jax.Array,
               global_features: jax.Array,
               senders: jax.Array,
               receivers: jax.Array,
               ) -> jax.Array:
    ############################
    # Pre-processing
    ############################
    input_graph = dict(
        node_features=node_features,
        edge_features=edge_features,
        global_features=global_features,
        senders=senders,
        receivers=receivers,
    )
    num_nodes = node_features.shape[-2]
    num_edges = senders.shape[-1]
    leading_dims = node_features.shape[:-2]

    segment_softmax = flax_gnn.util.segment_softmax
    segment_sum = flax_gnn.util.segment_sum
    for _ in range(len(leading_dims)):
      segment_softmax = jax.vmap(segment_softmax, in_axes=(0, 0, None))
      segment_sum = jax.vmap(segment_sum, in_axes=(0, 0, None))

    ############################
    # Edge update
    ############################
    if self.share_weights:
      W = nn.Dense(self.embed_dim, name='W', kernel_init=self.kernel_init)
      send_nodes = recv_nodes = W(node_features)
    else:
      W_s = nn.Dense(self.embed_dim, name='W_s', kernel_init=self.kernel_init)
      W_r = nn.Dense(self.embed_dim, name='W_r', kernel_init=self.kernel_init)
      send_nodes = W_s(node_features)
      recv_nodes = W_r(node_features)
    send_edges = jnp.take_along_axis(send_nodes, senders[..., None], axis=-2)
    recv_edges = jnp.take_along_axis(recv_nodes, receivers[..., None], axis=-2)
    x = send_edges + recv_edges

    if edge_features is not None or global_features is not None:
      if edge_features is None:
        edge_features = global_features.repeat(num_edges, axis=-2)
      elif edge_features is not None and global_features is not None:
        edge_features = jnp.concatenate(
            [edge_features, global_features.repeat(num_edges, axis=-2)], axis=-1
        )
      W_e = nn.Dense(self.embed_dim, name='W_e', kernel_init=self.kernel_init)
      x += W_e(edge_features)

    if self.add_self_edges:
      node_inds = jnp.broadcast_to(
          jnp.arange(num_nodes), leading_dims + (num_nodes,)
      )
      receivers = jnp.concatenate([receivers, node_inds], axis=-1)
      send_edges = jnp.concatenate([send_edges, send_nodes], axis=-2)
      x = jnp.concatenate([x, send_nodes + recv_nodes], axis=-2)

    ############################
    # Attention
    ############################
    x = mish(x)
    x = rearrange(x, '... (h d) -> ... h d', h=self.num_heads)
    a = self.param(
        'a',
        self.kernel_init,
        (self.num_heads, self.embed_dim // self.num_heads)
    )
    a = jnp.tile(a, (*x.shape[:-2], 1, 1))
    attn_logits = jnp.sum(x * a, axis=-1, keepdims=True)
    attn_weights = segment_softmax(attn_logits, receivers, num_nodes)

    ############################
    # Node Update
    ############################
    edges = rearrange(send_edges, '... (h d) -> ... h d', h=self.num_heads)
    edges = attn_weights * edges
    edges = rearrange(edges, '... h d -> ... (h d)')
    new_nodes = segment_sum(edges, receivers, num_nodes)

    return dict(
        node_features=new_nodes,
        edge_features=input_graph['edge_features'],
        global_features=input_graph['global_features'],
        senders=input_graph['senders'],
        receivers=input_graph['receivers'],
    )
