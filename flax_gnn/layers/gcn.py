import time
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jraph
from flax_gnn.util import add_self_edges
import jax
import jax.numpy as jnp
from einops import rearrange
from flax_gnn.layers.activations import mish


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
  kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    W = nn.Dense(self.embed_dim, kernel_init=self.kernel_init,
                 dtype=self.dtype)
    W_e = nn.Dense(self.embed_dim, kernel_init=self.kernel_init,
                   dtype=self.dtype)

    def update_edge_fn(edges: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_edge_attributes: jnp.ndarray
                       ) -> jnp.ndarray:
      return (edges, sent_attributes)

    def update_node_fn(nodes: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_attributes: jnp.ndarray) -> jnp.ndarray:
      edges, received_attributes = received_attributes
      nodes = W(received_attributes)

      # Handle edge/global attributes
      if edges is not None or global_attributes is not None:
        if global_attributes is None:  # Edge only
          edge_attributes = edges
        elif edges is None:  # Global only
          edge_attributes = global_attributes
        else:  # Edge and global
          edge_attributes = jnp.concatenate(
              [edges, global_attributes], axis=-1)
        nodes += W_e(edge_attributes)

      # Scale by the square root of the node degrees
      if self.normalize:
        def count_edges(x): return jax.ops.segment_sum(
            jnp.ones_like(graph.senders), x, nodes.shape[0])
        sender_degrees = count_edges(graph.senders)
        receiver_degrees = count_edges(graph.receivers)

        nodes = nodes * jax.lax.rsqrt(jnp.maximum(sender_degrees, 1.0))[
            :, None] * jax.lax.rsqrt(jnp.maximum(receiver_degrees, 1.0))[:, None]

      return nodes

    network = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        aggregate_edges_for_nodes_fn=jraph.segment_sum,
    )
    if self.add_self_edges:
      graph_ = add_self_edges(graph)
    else:
      graph_ = graph

    graph = graph._replace(nodes=network(graph_).nodes)

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
