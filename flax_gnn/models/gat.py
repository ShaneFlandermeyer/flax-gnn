from typing import Optional
import jraph



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
