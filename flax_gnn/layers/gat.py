import time
from typing import Callable, Optional
import flax.linen as nn
import jraph
from flax_gnn.util import add_self_edges
import jax
import jax.numpy as jnp
from einops import rearrange
from flax_gnn.util import mish


def GATv2(attention_query_fn: jraph.GATAttentionQueryFn,
          attention_logit_fn: jraph.GATAttentionLogitFn,
          node_update_fn: Optional[jraph.GATNodeUpdateFn] = None):
  """Returns a method that applies a Graph Attention Network v2 layer.

  This is identical to the GAT function in the jraph library, but the attention query function processes sender and receiver nodes as separate inputs as required for GATv2. This also allows us to explicitly model bipartite graphs.

  Graph Attention 2 message passing as described in
  https://arxiv.org/abs/2105.14491. This model expects node features as a
  jnp.array, may use edge features for computing attention weights, and
  ignore global features. It does not support nests.

  NOTE: this implementation assumes that the input graph has self edges. To
  recover the behavior of the referenced paper, please add self edges.

  Args:
    attention_query_fn: function that generates attention queries
      from sender node features.
    attention_logit_fn: function that converts attention queries into logits for
      softmax attention.
    node_update_fn: function that updates the aggregated messages. If None,
      will apply leaky relu and concatenate (if using multi-head attention).

  Returns:
    A function that applies a Graph Attention layer.
  """
  # pylint: disable=g-long-lambda
  def _ApplyGATv2(graph):
    """Applies a Graph Attention layer."""
    nodes, edges, receivers, senders, _, _, _ = graph
    # Equivalent to the sum of n_node, but statically known.
    try:
      sum_n_node = nodes.shape[0]
    except IndexError:
      raise IndexError(
          'GAT requires node features')  # pylint: disable=raise-missing-from

    # pylint: disable=g-long-lambda
    # We compute the softmax logits using a function that takes the
    # embedded sender and receiver attributes.
    sent_attributes = nodes[senders]
    received_attributes = nodes[receivers]
    # First pass nodes through the node updater.
    sent_attributes, received_attributes = attention_query_fn(
        sent_attributes, received_attributes)
    softmax_logits = attention_logit_fn(
        sent_attributes, received_attributes, edges)

    # Compute the softmax weights on the entire tree.
    weights = jraph.segment_softmax(softmax_logits, segment_ids=receivers,
                                    num_segments=sum_n_node)
    # Apply weights
    messages = sent_attributes * weights
    # Aggregate messages to nodes.
    nodes = jraph.segment_sum(messages, receivers, num_segments=sum_n_node)

    # Apply an update function to the aggregated messages.
    nodes = node_update_fn(nodes)
    return graph._replace(nodes=nodes)
  # pylint: enable=g-long-lambda
  return _ApplyGATv2


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
      W_i = nn.DenseGeneral(features=(self.num_heads, head_dim))
      W_j = nn.DenseGeneral(features=(self.num_heads, head_dim))

      sender_attr = W_i(sender_attr)
      receiver_attr = W_j(receiver_attr)
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
      return rearrange(node, '... h d -> ... (h d)')

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
