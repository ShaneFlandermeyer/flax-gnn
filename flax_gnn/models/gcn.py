import jraph
import flax.linen as nn
import jax


class GCN(nn.Module):
  embed_dim: int
  add_self_edges: bool = False
  normalize: bool = True

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:

    graph = jraph.GraphConvolution(
        update_node_fn=lambda node: nn.Dense(self.embed_dim)(node),
        aggregate_nodes_fn=jax.ops.segment_sum,
        add_self_edges=self.add_self_edges,
        symmetric_normalization=self.normalize
    )(graph)

    return graph
