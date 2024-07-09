import jax.numpy as jnp
from typing import Optional, Sequence, Tuple
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


def batch_graphs(graphs: Sequence[jraph.GraphsTuple],
                 axis: int = 0
                 ) -> jraph.GraphsTuple:
  """Batch multiple graphs of the same shape along a given dimension."""
  return jraph.GraphsTuple(
      nodes=jnp.stack([g.nodes for g in graphs], axis=axis),
      edges=jnp.stack([g.edges for g in graphs], axis=axis),
      receivers=jnp.stack([g.receivers for g in graphs], axis=axis),
      senders=jnp.stack([g.senders for g in graphs], axis=axis),
      globals=jnp.stack([g.globals for g in graphs], axis=axis),
      n_node=jnp.stack([g.n_node for g in graphs], axis=axis),
      n_edge=jnp.stack([g.n_edge for g in graphs], axis=axis),
  )