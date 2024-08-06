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

  Specifically, attention messages are computed as edge features (the original edge features are discarded). The aggregated messages are then used to update the node features.
  """
  embed_dim: int
  num_heads: int
  add_self_edges: bool = False

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    head_dim = self.embed_dim // self.num_heads

    W_s = nn.DenseGeneral(features=(self.num_heads, head_dim))
    W_r = nn.DenseGeneral(features=(self.num_heads, head_dim))

    # Using an MLP for these since they don't get updated across layers
    W_g = nn.DenseGeneral(features=(self.num_heads, head_dim))
    W_e = nn.DenseGeneral(features=(self.num_heads, head_dim))

    if self.add_self_edges:
      graph = add_self_edges(graph)

    def update_edge_fn(edges: Optional[jnp.ndarray],
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_edge_attributes: Optional[jnp.ndarray]
                       ) -> jnp.ndarray:
      del received_attributes

      sent_attributes = W_s(sent_attributes)

      if global_edge_attributes is None:
        global_edge_attributes = 0
      else:
        global_edge_attributes = W_g(global_edge_attributes)

      if edges is None:
        edge_attributes = 0
      else:
        edge_attributes = W_e(edges)

      return sent_attributes + edge_attributes + global_edge_attributes

    def attention_logit_fn(edges: jnp.ndarray,
                           sent_attributes: jnp.ndarray,
                           received_attributes: jnp.ndarray,
                           global_edge_attributes: jnp.ndarray) -> jnp.ndarray:
      del global_edge_attributes

      sent_attributes = edges  # Computed above
      received_attributes = W_r(received_attributes)

      x = mish(sent_attributes + received_attributes)
      x = nn.Dense(1)(x)
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
      del nodes, sent_attributes, global_attributes

      # Identity transformation - Node features are updated based on the aggregated messages from other nodes
      # Some implementations apply a nonlinearity here, but we allow the user to do that somewhere else
      nodes = received_attributes

      return nodes

    network = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        # update_global_fn=update_global_fn,
        aggregate_edges_for_nodes_fn=jraph.segment_sum,
        aggregate_nodes_for_globals_fn=jraph.segment_mean,
        aggregate_edges_for_globals_fn=jraph.segment_mean,
        attention_logit_fn=attention_logit_fn,
        attention_normalize_fn=jraph.segment_softmax,
        attention_reduce_fn=attention_reduce_fn

    )
    graph = graph._replace(nodes=network(graph).nodes)

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
