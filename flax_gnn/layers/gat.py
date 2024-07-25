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

  TODO: Handle masking of nodes and edges (e.g., from padding)
  """
  embed_dim: int
  num_heads: int
  add_self_edges: bool = False

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    head_dim = self.embed_dim // self.num_heads
    W_e = nn.DenseGeneral(features=(self.num_heads, head_dim))
    W_s = nn.Dense(self.embed_dim) # Already multi-headed
    W_r = nn.DenseGeneral(features=(self.num_heads, head_dim))

    if self.add_self_edges:
      graph = add_self_edges(graph)

    def update_edge_fn(edges: Optional[jnp.ndarray],
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_edge_attributes: Optional[jnp.ndarray]
                       ) -> jnp.ndarray:
      del received_attributes

      if edges is None: # No edge features provided
        edges = sent_attributes
      else:
        edges = jnp.concatenate([edges, sent_attributes], axis=-1)

      if global_edge_attributes is not None:
        edges = jnp.concatenate([edges, global_edge_attributes], axis=-1)

      edges = W_e(edges)

      return edges

    def attention_logit_fn(edges: jnp.ndarray,
                           sent_attributes: jnp.ndarray,
                           received_attributes: jnp.ndarray,
                           global_edge_attributes: jnp.ndarray) -> jnp.ndarray:
      # Already used to compute edge features
      del sent_attributes
      sent_attributes = edges

      if global_edge_attributes is not None:
        received_attributes = jnp.concatenate(
            [received_attributes, global_edge_attributes], axis=-1)

      # GATv2 update rule
      sent_attributes = W_s(sent_attributes)
      received_attributes = W_r(received_attributes)
      x = mish(sent_attributes + received_attributes)
      x = nn.Dense(1)(x)
      return x

    def attention_reduce_fn(edges: jnp.ndarray,
                            weights: jnp.ndarray) -> jnp.ndarray:
      # Scale by softmax weights.
      # We do not aggregate here since aggregation is done during node feature computation
      x = weights * edges
      x = rearrange(x, '... h d -> ... (h d)')
      return x

    def update_node_fn(nodes: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_attributes: jnp.ndarray) -> jnp.ndarray:
      del nodes, sent_attributes, global_attributes

      # Identity transformation - Node features come from the aggregated edge features of the attention mechanism
      nodes = received_attributes

      return nodes

    def update_global_fn(node_attributes: jnp.ndarray,
                         edge_attributes: jnp.ndarray,
                         global_attributes: Optional[jnp.ndarray]
                         ) -> jnp.ndarray:
      if global_attributes is None:
        return

      attributes = jnp.concatenate(
          [node_attributes, edge_attributes, global_attributes], axis=-1)
      return nn.Dense(self.embed_dim)(attributes)

    network = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=update_global_fn,
        aggregate_edges_for_nodes_fn=jraph.segment_sum,
        aggregate_nodes_for_globals_fn=jraph.segment_mean,
        aggregate_edges_for_globals_fn=jraph.segment_mean,
        attention_logit_fn=attention_logit_fn,
        attention_normalize_fn=jraph.segment_softmax,
        attention_reduce_fn=attention_reduce_fn
        
    )
    graph = network(graph)

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
