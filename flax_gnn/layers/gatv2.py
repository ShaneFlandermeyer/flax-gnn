import time
from typing import Optional, Tuple
import flax.linen as nn
import jraph
from flax_gnn.util import add_self_edges
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
               receivers: jax.Array
               ) -> jax.Array:
    num_nodes = node_features.shape[-2]
    num_edges = senders.shape[-1]
    leading_dims = node_features.shape[:-2]

    if self.add_self_edges:
      senders = jnp.concatenate([senders, jnp.arange(num_nodes)], axis=0)
      receivers = jnp.concatenate([receivers, jnp.arange(num_nodes)], axis=0)
      if edge_features is not None:
        self_edge_features = jnp.zeros(
            (leading_dims, num_nodes, edge_features.shape[-1])
        )
        edge_features = jnp.concatenate(
            [edge_features, self_edge_features], axis=0
        )

    if edge_features is not None or global_features is not None:
      if edge_features is None:
        edge_features = global_features.repeat(num_edges, axis=-2)
      elif edge_features is not None and global_features is not None:
        edge_features = jnp.concatenate(
            [edge_features, global_features.repeat(num_edges, axis=-2)], axis=-1
        )

    if self.share_weights:
      W = nn.Dense(self.embed_dim, name='W', kernel_init=self.kernel_init)
      nodes = W(node_features)
      send_nodes = jnp.take_along_axis(nodes, senders[..., None], axis=-2)
      recv_nodes = jnp.take_along_axis(nodes, receivers[..., None], axis=-2)
    else:
      W_s = nn.Dense(self.embed_dim, name='W_s', kernel_init=self.kernel_init)
      W_r = nn.Dense(self.embed_dim, name='W_r', kernel_init=self.kernel_init)
      send_nodes = W_s(
          jnp.take_along_axis(node_features, senders[..., None], axis=-2)
      )
      recv_nodes = W_r(
          jnp.take_along_axis(node_features, receivers[..., None], axis=-2)
      )
    x = send_nodes + recv_nodes

    if edge_features is not None:
      W_e = nn.Dense(
          self.embed_dim, name='W_e', kernel_init=self.kernel_init
      )
      x += W_e(edge_features)
    x = jax.nn.leaky_relu(x)

    # Multi-head attention weights
    x = rearrange(x, '... (h d) -> ... h d', h=self.num_heads)
    a = self.param(
        'a',
        self.kernel_init,
        (self.num_heads, self.embed_dim // self.num_heads)
    )
    a = jnp.tile(a, (*x.shape[:-2], 1, 1))
    attn_logits = jnp.sum(x * a, axis=-1, keepdims=True)
    segment_softmax = jraph.segment_softmax
    for _ in range(len(leading_dims)):
      segment_softmax = jax.vmap(segment_softmax, in_axes=(0, 0, None))
    attn_weights = segment_softmax(attn_logits, receivers, num_nodes)

    # Node update
    segment_sum = jax.ops.segment_sum
    for _ in range(len(leading_dims)):
      segment_sum = jax.vmap(segment_sum, in_axes=(0, 0, None))
    edges = rearrange(
        send_nodes, '... (h d) -> ... h d', h=self.num_heads
    )
    edges = attn_weights * edges
    edges = rearrange(edges, '... h d -> ... (h d)')
    new_nodes = segment_sum(edges, receivers, num_nodes)

    return dict(
        node_features=new_nodes,
        edge_features=edge_features,
        global_features=global_features,
        senders=senders,
        receivers=receivers
    )


if __name__ == '__main__':
  from flax_gnn.test.util import build_toy_graph

  graph = build_toy_graph()
  graph = dict(
      node_features=graph.nodes,
      edge_features=graph.edges,
      global_features=graph.globals,
      senders=graph.senders,
      receivers=graph.receivers
  )
  model = GATv2(embed_dim=8, num_heads=2, share_weights=True)
  params = model.init(jax.random.PRNGKey(42), **graph)

  apply = jax.jit(model.apply)
  start = time.time()
  decoded_graph = apply(params, **graph)
  print(time.time() - start)
  start = time.time()
  decoded_graph = apply(params, **graph)
  print(time.time() - start)
