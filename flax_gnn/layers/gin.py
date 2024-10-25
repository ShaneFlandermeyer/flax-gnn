import time
from typing import Optional, Tuple
import flax.linen as nn
import jraph
from flax_gnn.util import add_self_edges
import jax
import jax.numpy as jnp
from einops import rearrange
from flax_gnn.layers.activations import mish
from typing import Optional, Tuple


class GIN(nn.Module):
  """
  Implementation of GATv2 using jraph.GraphNetwork.

  The implementation is based on the appendix in Battaglia et al. (2018) "Relational inductive biases, deep learning, and graph networks".

  Incorporates global and edge features as in Wang2021
  """
  mlp: nn.Module
  epsilon: Optional[float] = None
  aggregate_edges_for_nodes_fn: nn.Module = jraph.segment_sum

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple, **mlp_args) -> jraph.GraphsTuple:
    def update_edge_fn(edges: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_edge_attributes: jnp.ndarray
                       ) -> Tuple[jnp.ndarray, jnp.ndarray]:

      # Handle edge/global attributes
      if edges is not None or global_edge_attributes is not None:
        if global_edge_attributes is None:
          edge_attributes = edges
        elif edges is None:
          edge_attributes = global_edge_attributes
        else:
          edge_attributes = jnp.concatenate(
              [edges, global_edge_attributes], axis=-1)
        embed_dim = sent_attributes.shape[-1]
        W_e = nn.Dense(embed_dim, name='W_e')
        sent_attributes += W_e(edge_attributes)

      return sent_attributes

    def update_node_fn(nodes: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_attributes: jnp.ndarray) -> jnp.ndarray:
      if self.epsilon is None:
        epsilon = self.param('epsilon', nn.initializers.zeros, (1, 1))
      else:
        epsilon = self.epsilon

      epsilon = jnp.tile(epsilon, (*nodes.shape[:-2], 1, 1))
      nodes = self.mlp((1 + epsilon) * nodes + received_attributes, **mlp_args)

      return nodes

    network = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        aggregate_edges_for_nodes_fn=self.aggregate_edges_for_nodes_fn,
    )

    graph = graph._replace(
        nodes=network(graph).nodes,
    )
    return graph


if __name__ == '__main__':
  from flax_gnn.test.util import build_toy_graph

  graph = build_toy_graph()
  graph = graph._replace(globals=None)
  mlp = nn.Sequential([
      nn.Dense(8),
      nn.relu,
      nn.Dense(8)
  ], name='mlp')
  model = GIN(mlp=mlp)
  params = model.init(jax.random.PRNGKey(42), graph)
  print(model.tabulate(jax.random.PRNGKey(42), graph))
  # apply = jax.jit(model.apply)
  # start = time.time()
  # decoded_graph = apply(params, graph)
  # print(time.time() - start)
  # start = time.time()
  # decoded_graph = apply(params, graph)
  # print(time.time() - start)
