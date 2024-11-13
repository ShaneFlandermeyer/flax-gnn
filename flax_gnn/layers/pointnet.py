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


class PointNet(nn.Module):
  """
  Implementation of Pointnet++ using jraph.GraphNetwork.

  Adds edge features before aggregation (followed by a nonlinearity)

  Concatenates global attributes to nodes before MLP

  Sources: 
  - https://arxiv.org/pdf/1905.12265
  - https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html#torch_geometric.nn.conv.GINEConv
  """
  mlp: nn.Module

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    def update_edge_fn(edges: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_edge_attributes: jnp.ndarray
                       ) -> Tuple[jnp.ndarray, jnp.ndarray]:
      if edges is not None:
        edges = jnp.concatenate([sent_attributes, edges], axis=-1)
      else:
        edges = sent_attributes

      if global_edge_attributes is not None:
        edges = jnp.concatenate([edges, global_edge_attributes], axis=-1)

      edges = self.mlp(edges)
      return edges

    def update_node_fn(nodes: jnp.ndarray,
                       sent_attributes: jnp.ndarray,
                       received_attributes: jnp.ndarray,
                       global_attributes: jnp.ndarray) -> jnp.ndarray:
      # Possibly getting nans from segment_max
      received_attributes = jnp.nan_to_num(received_attributes)
      return received_attributes

    network = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        aggregate_edges_for_nodes_fn=jraph.segment_max,
    )

    graph = graph._replace(
        nodes=network(graph).nodes,
    )
    return graph
