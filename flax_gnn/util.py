import jax.numpy as jnp
from typing import Tuple
import jraph
import jax


def add_self_edges_fn(receivers: jnp.ndarray, senders: jnp.ndarray,
                      total_num_nodes: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Adds self edges. Assumes self edges are not in the graph yet."""
  # TODO: This does not add edge features for self edges, which might cause shape problems.
  receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)), axis=0)
  senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)), axis=0)
  return receivers, senders


def add_self_edges(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Add self edges and return a new graph"""
  total_num_nodes = jax.tree.leaves(graph.nodes)[0].shape[0]
  receivers, senders = add_self_edges_fn(
      graph.receivers, graph.senders, total_num_nodes)
  return graph._replace(receivers=receivers, senders=senders)


def mish(x: jnp.ndarray) -> jnp.ndarray:
  return x * jnp.tanh(jnp.log1p(jnp.exp(x)))
