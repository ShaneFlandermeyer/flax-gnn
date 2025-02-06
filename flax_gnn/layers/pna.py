import jax
import flax.linen as nn
import jraph
import jax.numpy as jnp
from typing import Optional

from flax_gnn.util import add_self_edges
from einops import rearrange


def mish(x):
  return x * jnp.tanh(jax.nn.softplus(x))


def log_scaler(d: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
  return (jnp.log1p(d) / jnp.log1p(d).mean())**alpha


class PNA(nn.Module):
  embed_dim: int

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    def update_edge_fn(edges: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_edge_attributes: jnp.ndarray
                       ) -> jnp.ndarray:
      x = jnp.concatenate([sent_attributes, received_attributes], axis=-1)
      if edges is not None:
        x = jnp.concatenate([x, edges], axis=-1)

      M = nn.Dense(self.embed_dim, name='U')
      x = mish(M(x))

      return x

    def update_node_fn(nodes: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_attributes: jnp.ndarray) -> jnp.ndarray:
      # Segment utilities may produce nans
      received_attributes = jnp.nan_to_num(received_attributes)
      x = jnp.concatenate([nodes, received_attributes], axis=-1)
      
      if global_attributes is not None:
        x = jnp.concatenate([x, global_attributes], axis=-1)

      U = nn.Dense(self.embed_dim, name='M')
      x = mish(U(x))
      return x

    def aggregate_edges_for_nodes_fn(edges: jnp.ndarray,
                                     segment_ids: jnp.ndarray,
                                     num_segments: int) -> jnp.ndarray:
      aggregators = jnp.concatenate([
          jraph.segment_mean(edges, segment_ids, num_segments),
          jnp.sqrt(jraph.segment_variance(
              edges, segment_ids, num_segments) + 1e-10),
          jraph.segment_max(edges, segment_ids, num_segments),
          jraph.segment_min(edges, segment_ids, num_segments),
      ], axis=-1)

      # Compute the degree of each node
      d = jax.ops.segment_sum(jnp.ones_like(
          segment_ids), segment_ids, num_segments)
      scalers = jnp.stack([
          jnp.ones_like(d),
          log_scaler(d, 1),
          log_scaler(d, -1)
      ], axis=-1)
      
      # Kronecker product between scalers and aggregators
      x = scalers[..., :, None] * aggregators[..., None, :]
      x = rearrange(x, '... i j  -> ... (i j)')
      return x

    network = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
    )
    graph = graph._replace(nodes=network(graph).nodes)
    return graph


if __name__ == '__main__':
  from flax_gnn.test.util import build_toy_graph

  graph = build_toy_graph()
  graph = graph._replace(globals=None)
  pna = PNA(8)
  params = pna.init(jax.random.PRNGKey(0), graph)
  graph = pna.apply(params, graph)
  print(graph.nodes)
  print(graph.nodes.shape)
  print(graph.globals)
  print(graph.globals.shape)
  print(graph.edges)
  print(graph.edges.shape)
  print(graph.receivers)
  print(graph.receivers.shape)
  print(graph.senders)
  print(graph.senders.shape)
  print(graph.n_node)
  print(graph.n_edge)
  print(graph.n_node.shape)
  print(graph.n_edge)
