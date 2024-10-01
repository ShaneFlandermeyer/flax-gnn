import time
from typing import Optional
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

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    head_dim = self.embed_dim // self.num_heads

    W_s = nn.DenseGeneral(
        features=(self.num_heads, head_dim), dtype=self.dtype)
    if self.share_weights:
      W_r = W_s
    else:
      W_r = nn.DenseGeneral(
          features=(self.num_heads, head_dim), dtype=self.dtype)

    # Using an MLP for these since they don't get updated across layers
    if graph.globals is None:
      W_g = None
    else:
      W_g = nn.DenseGeneral(
          features=(self.num_heads, head_dim), dtype=self.dtype)

    if graph.edges is None:
      W_e = None
    else:
      W_e = nn.DenseGeneral(
          features=(self.num_heads, head_dim), dtype=self.dtype)

    def update_edge_fn(edges: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_edge_attributes: jnp.ndarray
                       ) -> jnp.ndarray:
      x = W_s(sent_attributes)
      # Handle edge features
      if edges is not None:
        x += W_e(edges)
      return x

    def attention_logit_fn(edges: jnp.ndarray,
                           sent_attributes: jnp.ndarray,
                           received_attributes: jnp.ndarray,
                           global_edge_attributes: jnp.ndarray) -> jnp.ndarray:
      sent_attributes = edges  # Computed in update_edge_fn
      received_attributes = W_r(received_attributes)

      x = sent_attributes + received_attributes
      # Handle global features
      if global_edge_attributes is not None:
        x += W_g(global_edge_attributes)
        
      x = jax.nn.leaky_relu(x)
      x = nn.Dense(1, dtype=self.dtype)(x)
      return x

    def attention_reduce_fn(edges: jnp.ndarray,
                            weights: jnp.ndarray) -> jnp.ndarray:
      x = weights * edges
      x = rearrange(x, '... h d -> ... (h d)')
      return x

    def update_node_fn(nodes: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_attributes: jnp.ndarray) -> jnp.ndarray:
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
    graph = graph._replace(
        nodes=graph_.nodes,
    )

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
