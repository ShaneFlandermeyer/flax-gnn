import jax.numpy as jnp
from typing import Optional, Sequence, Tuple
import jraph
import jax
from functools import partial


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


def pad_with_graphs(graph: jraph.GraphsTuple,
                    n_node: int,
                    n_edge: int,
                    n_graph: int = 2) -> jraph.GraphsTuple:
  """
  Identical to jraph.pad_with_graphs, but can be JIT'd if n_node/n_edge/n_graph are constant (using static_argnums)
  """
  if n_graph < 2:
    raise ValueError(
        f'n_graph is {n_graph}, which is smaller than minimum value of 2.')

  graph = jax.device_get(graph)
  sum_n_node = jax.tree.leaves(graph.nodes)[0].shape[0]
  sum_n_edge = jax.tree.leaves(graph.edges)[0].shape[0]
  pad_n_node = n_node - sum_n_node
  pad_n_edge = n_edge - sum_n_edge
  pad_n_graph = n_graph - graph.n_node.shape[0]

  if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
    raise RuntimeError(
        'Given graph is too large for the given padding. difference: '
        f'n_node {pad_n_node}, n_edge {pad_n_edge}, n_graph {pad_n_graph}')

  pad_n_empty_graph = pad_n_graph - 1

  def zero_pad(x, n): return jnp.zeros((n, ) + x.shape[1:], dtype=x.dtype)

  padding_graph = jraph.GraphsTuple(
      n_node=jnp.pad(jnp.array([pad_n_node]), (0, pad_n_empty_graph)),
      n_edge=jnp.pad(jnp.array([pad_n_edge]), (0, pad_n_empty_graph)),
      nodes=jax.tree.map(partial(zero_pad, n=pad_n_node), graph.nodes),
      edges=jax.tree.map(partial(zero_pad, n=pad_n_edge), graph.edges),
      globals=jax.tree.map(partial(zero_pad, n=pad_n_graph), graph.globals),
      senders=jnp.zeros(pad_n_edge, dtype=jnp.int32),
      receivers=jnp.zeros(pad_n_edge, dtype=jnp.int32),
  )

  return jraph.batch([graph, padding_graph])
