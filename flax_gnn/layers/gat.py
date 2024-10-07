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
  add_self_edges: bool = False
  share_weights: bool = False
  dtype: jnp.dtype = jnp.float32
  kernel_init: nn.initializers.Initializer = nn.initializers.xavier_normal()

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    head_dim = self.embed_dim // self.num_heads
    W_s = nn.Dense(self.embed_dim, dtype=self.dtype,
                   kernel_init=self.kernel_init)
    if self.share_weights:
      W_r = W_s
    else:
      W_r = nn.Dense(self.embed_dim, dtype=self.dtype,
                     kernel_init=self.kernel_init)
    W_e = nn.Dense(self.embed_dim, dtype=self.dtype,
                   kernel_init=self.kernel_init)

    def update_edge_fn(edges: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_edge_attributes: jnp.ndarray
                       ) -> Tuple[jnp.ndarray, jnp.ndarray]:
      del received_attributes, global_edge_attributes
      sent_attributes = W_s(sent_attributes)

      return sent_attributes, edges

    def attention_logit_fn(edges: jnp.ndarray,
                           sent_attributes: jnp.ndarray,
                           received_attributes: jnp.ndarray,
                           global_edge_attributes: jnp.ndarray) -> jnp.ndarray:
      sent_attributes, edge_attributes = edges  # Computed from update_edge_fn
      received_attributes = W_r(received_attributes)
      x = sent_attributes + received_attributes

      # Handle edge/global attributes
      if edge_attributes is not None or global_edge_attributes is not None:
        if global_edge_attributes is None:  # Edge only
          edge_attributes = edge_attributes
        elif edge_attributes is None:  # Global only
          edge_attributes = global_edge_attributes
        else:  # Edge and global
          edge_attributes = jnp.concatenate(
              [edge_attributes, global_edge_attributes], axis=-1)
        edge_attributes = W_e(edge_attributes)
        x += edge_attributes

      x = jax.nn.leaky_relu(x)

      # Multi-head attention weights
      x = rearrange(x, '... (h d) -> ... h d', h=self.num_heads)
      a = self.param('a', self.kernel_init, (self.num_heads, head_dim))
      a = jnp.tile(a, (*x.shape[:-2], 1, 1)).astype(self.dtype)
      attn_logits = jnp.sum(x * a, axis=-1, keepdims=True)
      return attn_logits

    def attention_reduce_fn(edges: jnp.ndarray,
                            weights: jnp.ndarray) -> jnp.ndarray:
      sent_attributes, _ = edges
      sent_attributes = rearrange(
          sent_attributes, '... (h d) -> ... h d', h=self.num_heads)
      x = weights * sent_attributes
      x = rearrange(x, '... h d -> ... (h d)')
      return x

    def update_node_fn(nodes: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_attributes: jnp.ndarray) -> jnp.ndarray:
      del nodes, sent_attributes, global_attributes
      # Identity transformation - Node features are updated based on the aggregated messages from other nodes
      # Some implementations apply a nonlinearity here, but we allow the user to do that somewhere else
      return received_attributes

    network = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        aggregate_edges_for_nodes_fn=jraph.segment_sum,
        attention_logit_fn=attention_logit_fn,
        attention_normalize_fn=jraph.segment_softmax,
        attention_reduce_fn=attention_reduce_fn

    )

    if self.add_self_edges:
      graph_ = add_self_edges(graph)
    else:
      graph_ = graph

    graph_ = network(graph_)
    graph = graph._replace(nodes=graph_.nodes,)

    return graph


if __name__ == '__main__':
  from flax_gnn.test.util import build_toy_graph

  graph = build_toy_graph()
  graph = graph._replace(globals=None)
  model = GATv2(embed_dim=8, num_heads=2, add_self_edges=True)
  params = model.init(jax.random.PRNGKey(42), graph)

  apply = jax.jit(model.apply)
  start = time.time()
  decoded_graph = apply(params, graph)
  print(time.time() - start)
  start = time.time()
  decoded_graph = apply(params, graph)
  print(time.time() - start)
