import jax.numpy as jnp
from typing import Optional, Tuple
import jraph
import jax


def add_self_edges(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Add self edges and return a new graph"""

  total_num_nodes = jax.tree.leaves(graph.nodes)[0].shape[0]
  senders = jnp.concatenate(
      (graph.senders, jnp.arange(total_num_nodes)), axis=0)
  receivers = jnp.concatenate(
      (graph.receivers, jnp.arange(total_num_nodes)), axis=0)
  edges = graph.edges
  if edges is not None:
    edges = jnp.concatenate(
        (edges, jnp.zeros((total_num_nodes, *edges.shape[1:]))),
        axis=0)
  return graph._replace(senders=senders,
                        receivers=receivers,
                        edges=edges,
                        n_edge=graph.n_edge + total_num_nodes)


def mish(x: jnp.ndarray) -> jnp.ndarray:
  return x * jnp.tanh(jnp.log1p(jnp.exp(x)))
