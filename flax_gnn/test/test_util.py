from flax_gnn.util import add_self_edges, batch_graphs
import pytest
import jax.numpy as jnp
import jax

def test_batch_graphs():
  from flax_gnn.test.util import build_toy_graph
  graph1 = build_toy_graph()
  graph2 = build_toy_graph()

  graphs = batch_graphs((graph1, graph2), 0)
  
  assert graphs.nodes.shape == (2, 4, 1)
  assert graphs.edges.shape == (2, 5, 1)
  assert graphs.receivers.shape == (2, 5)
  assert graphs.senders.shape == (2, 5)
  assert graphs.globals.shape == (2, 1, 1)
  assert jnp.all(graphs.n_node == jnp.array([[4], [4]]))
  assert jnp.all(graphs.n_edge == jnp.array([[5], [5]]))

if __name__ == '__main__':
  pytest.main([__file__])
  