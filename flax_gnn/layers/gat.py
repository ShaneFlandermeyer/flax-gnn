import time
import flax.linen as nn
import jraph
from flax_gnn.util import add_self_edges
from flax_gnn.models.gat import GATv2
import jax
import jax.numpy as jnp
from einops import rearrange
from flax_gnn.util import mish


class GATv2Conv(nn.Module):
  embed_dim: int
  num_heads: int
  add_self_edges: bool = False

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    if self.add_self_edges:
      graph = add_self_edges(graph)

    def _attention_query_fn(sender_attr: jnp.ndarray,
                            receiver_attr: jnp.ndarray) -> jnp.ndarray:
      head_dim = self.embed_dim // self.num_heads
      W_s = nn.DenseGeneral(features=(self.num_heads, head_dim))
      W_r = nn.DenseGeneral(features=(self.num_heads, head_dim))

      sender_attr = W_s(sender_attr)
      receiver_attr = W_r(receiver_attr)
      return sender_attr, receiver_attr

    def _attention_logit_fn(sender_attr: jnp.ndarray,
                            receiver_attr: jnp.ndarray,
                            edges: jnp.ndarray) -> jnp.ndarray:
      # GATv2 update rule
      del edges
      x = mish(sender_attr + receiver_attr)
      x = nn.Dense(1)(x)
      return x

    def _node_update_fn(node: jnp.ndarray) -> jnp.ndarray:
      x = rearrange(node, '... h d -> ... (h d)')

      return x

    graph = GATv2(
        attention_query_fn=_attention_query_fn,
        attention_logit_fn=_attention_logit_fn,
        node_update_fn=_node_update_fn,
    )(graph)

    return graph


if __name__ == '__main__':
  from flax_gnn.test.util import build_toy_graph

  graph = build_toy_graph()
  model = GATv2Conv(embed_dim=8, num_heads=2, add_self_edges=True)
  params = model.init(jax.random.PRNGKey(42), graph)

  apply = jax.jit(model.apply)
  start = time.time()
  decoded_graph = apply(params, graph)
  print(time.time() - start)
  start = time.time()
  decoded_graph = apply(params, graph)
  print(time.time() - start)
