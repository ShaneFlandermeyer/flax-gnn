import time
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jraph
from flax_gnn.util import add_self_edges
import jax
import jax.numpy as jnp
from einops import rearrange
from flax_gnn.layers.activations import mish


# def symmetric_normalization(graph: jraph.GraphsTuple):
#   total_num_nodes = jax.tree.leaves(graph.nodes)[0].shape[0]
#   # Calculate the normalization values.
#   def count_edges(x): return jax.ops.segment_sum(
#       jnp.ones_like(graph.senders), x, total_num_nodes)
#   sender_degree = count_edges(graph.senders)
#   receiver_degree = count_edges(graph.receivers)

#   # Pre normalize by sqrt sender degree.
#   # Avoid dividing by 0 by taking maximum of (degree, 1).
#   nodes = jax.tree.map(
#       lambda x: x * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None],
#       graph.nodes,
#   )
#   # Aggregate the pre-normalized nodes.
#   nodes = jax.tree.map(
#       lambda x: jraph.segment_sum(x[graph.senders], graph.receivers,
#                                   total_num_nodes), graph.nodes)
#   # Post normalize by sqrt receiver degree.
#   # Avoid dividing by 0 by taking maximum of (degree, 1).
#   nodes = jax.tree.map(
#       lambda x:
#       (x * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, None]),
#       nodes,
#   )

#   return graph._replace(nodes=nodes)


class GCN(nn.Module):
  """
  Implementation of a Graph Convolution layer using jraph.GraphNetwork

  The implementation is based on the appendix in Kipf et al. (2017) "Semi-Supervised Classification with Graph Convolutional Networks".

  Incorporates global and edge features in the node update step
  """
  embed_dim: int
  add_self_edges: bool = False
  normalize: bool = True
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    W = nn.Dense(self.embed_dim, dtype=self.dtype)
    W_e = nn.Dense(self.embed_dim, dtype=self.dtype)

    def update_edge_fn(edges: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_edge_attributes: jnp.ndarray
                       ) -> jnp.ndarray:
      x = W(sent_attributes)

      # Handle edge/global attributes
      if edges is not None or global_edge_attributes is not None:
        if global_edge_attributes is None:  # Edge only
          edge_attributes = edges
        elif edges is None:  # Global only
          edge_attributes = global_edge_attributes
        else:  # Edge and global
          edge_attributes = jnp.concatenate(
              [edges, global_edge_attributes], axis=-1)
        edge_attributes = W_e(edge_attributes)
        x += edge_attributes

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
    )
    if self.add_self_edges:
      graph_ = add_self_edges(graph)
    else:
      graph_ = graph
    graph_ = network(graph_)

    # Symmetric normalization
    nodes = graph_.nodes
    if self.normalize:
      total_num_nodes = jax.tree.leaves(nodes)[0].shape[0]
      sender_degree = jax.ops.segment_sum(
          jnp.ones_like(graph_.senders), graph_.senders, total_num_nodes
      ).clip(1, None)
      receiver_degree = jax.ops.segment_sum(
          jnp.ones_like(graph_.receivers), graph_.receivers, total_num_nodes
      ).clip(1, None)
      nodes = nodes / jnp.sqrt(sender_degree * receiver_degree)[..., None]

    graph = graph._replace(nodes=nodes)

    return graph


if __name__ == '__main__':
  from flax_gnn.test.util import build_toy_graph

  graph = build_toy_graph()
  graph = graph._replace(globals=None)
  model = GCN(embed_dim=8, num_heads=2, add_self_edges=True)
  params = model.init(jax.random.PRNGKey(42), graph)

  apply = jax.jit(model.apply)
  start = time.time()
  decoded_graph = apply(params, graph)
  print(time.time() - start)
  start = time.time()
  decoded_graph = apply(params, graph)
  print(time.time() - start)
