import time
import flax.linen as nn
import jraph
from flax_gnn.util import add_self_edges
# from flax_gnn.models.gat import GATv2
import jax
import jax.numpy as jnp
from einops import rearrange
from flax_gnn.util import mish


class GATv2(nn.Module):
  """
  Implementation of GATv2 using jraph.GraphNetwork.

  The implementation is based on the appendix in Battaglia et al. (2018) "Relational inductive biases, deep learning, and graph networks".

  Specifically, attention messages are computed as edge features (the original edge features are discarded). The aggregated messages are then used to update the node features.
  
  NOTE: Global features are currently not used, but there are hooks to include them!
  """
  embed_dim: int
  num_heads: int
  add_self_edges: bool = False

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    if self.add_self_edges:
      graph = add_self_edges(graph)

    head_dim = self.embed_dim // self.num_heads
    W_s = nn.DenseGeneral(features=(self.num_heads, head_dim))
    W_r = nn.DenseGeneral(features=(self.num_heads, head_dim))

    def _update_edge_fn(edges, sent_attributes, received_attributes,
                        global_edge_attributes):
      del edges, received_attributes, global_edge_attributes

      edges = W_s(sent_attributes)
      return edges

    def _attention_logit_fn(edges, sent_attributes, received_attributes,
                            global_edge_attributes) -> jnp.ndarray:
      del sent_attributes, global_edge_attributes
      # GATv2 update rule
      # Sent attribute embeddings live in the edge features, so we don't need to recompute them here
      x = mish(edges + W_r(received_attributes))
      x = nn.Dense(1)(x)
      return x

    def _attention_reduce_fn(edges, weights):
      return edges * weights

    def _node_update_fn(nodes, sent_attributes,
                        received_attributes, global_attributes) -> jnp.ndarray:
      del sent_attributes, global_attributes
      nodes = rearrange(received_attributes, '... h d -> ... (h d)')
      return nodes

    graph = jraph.GraphNetwork(
        update_edge_fn=_update_edge_fn,
        update_node_fn=_node_update_fn,
        attention_logit_fn=_attention_logit_fn,
        attention_reduce_fn=_attention_reduce_fn
    )(graph)

    return graph


if __name__ == '__main__':
  from flax_gnn.test.util import build_toy_graph

  graph = build_toy_graph()
  model = GATv2(embed_dim=8, num_heads=2, add_self_edges=True)
  params = model.init(jax.random.PRNGKey(42), graph)

  apply = jax.jit(model.apply)
  start = time.time()
  decoded_graph = apply(params, graph)
  print(time.time() - start)
  start = time.time()
  decoded_graph = apply(params, graph)
  print(time.time() - start)
