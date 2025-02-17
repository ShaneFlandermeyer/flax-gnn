import time
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax_gnn.layers.activations import mish


class GCN(nn.Module):
  """
  Implementation of a Graph Convolution layer using jraph.GraphNetwork

  The implementation is based on the appendix in Kipf et al. (2017) "Semi-Supervised Classification with Graph Convolutional Networks".

  Incorporates global and edge features in the node update step
  """
  embed_dim: int
  normalize: bool = True

  @nn.compact
  def __call__(self,
               nodes: jax.Array,
               edge_features: jax.Array,
               global_features: jax.Array,
               senders: jax.Array,
               receivers: jax.Array
               ) -> jax.Array:
    num_nodes = nodes.shape[-2]
    num_edges = senders.shape[-1]

    ####################################
    # Edge update
    ####################################
    W = nn.Dense(self.embed_dim, name='W')
    nodes = W(nodes)

    sent_attributes = jnp.take_along_axis(
        nodes, senders[..., None], axis=-2
    )
    W_e = nn.Dense(self.embed_dim, name='W_e')
    if edge_features is None and global_features is None:
      edges = sent_attributes
    elif edge_features is not None and global_features is None:
      edges = mish(sent_attributes + W_e(edge_features))
    elif edge_features is None and global_features is not None:
      edge_features = global_features.repeat(num_edges, axis=-2)
      edges = mish(sent_attributes + W_e(edge_features))
    else:
      edge_features = jnp.concatenate(
          [edge_features, global_features.repeat(num_edges, axis=-2)], axis=-1
      )
      edges = mish(sent_attributes + W_e(edge_features))

    #####################################
    # Aggregate edges
    #####################################
    leading_dims = nodes.shape[:-2]
    edge_aggr = jax.ops.segment_sum
    for _ in range(len(leading_dims)):
      edge_aggr = jax.vmap(edge_aggr, in_axes=(0, 0, None))

    if self.normalize:
      in_degree = edge_aggr(
          jnp.ones_like(receivers), receivers, num_nodes
      ).astype(float)
      edges *= jax.lax.rsqrt(
          in_degree[senders].clip(1, None) * in_degree[receivers].clip(1, None)
      )[..., None]

    ####################################
    # Node update
    ####################################
    nodes = edge_aggr(edges, receivers, num_nodes)
    return nodes


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
