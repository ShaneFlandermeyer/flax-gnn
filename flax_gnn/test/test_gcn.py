import time
from typing import Any, Dict, Tuple
import jraph
import flax.linen as nn
import jax
import optax
from flax_gnn.layers.activations import mish
from flax_gnn.test.util import get_ground_truth_assignments_for_zacharys_karate_club, get_zacharys_karate_club
import jax.numpy as jnp
from flax_gnn.layers.gcn import GCN
import pytest


def test():
  class Model(nn.Module):

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
      graph = dict(
          node_features=graph.nodes,
          edge_features=graph.edges,
          global_features=graph.globals,
          senders=graph.senders,
          receivers=graph.receivers
      )
      graph = GCN(embed_dim=8, normalize=True, self_edges=True)(**graph)
      # graph['node_features'] = nn.relu(graph['node_features'])
      graph = GCN(embed_dim=2, normalize=True, self_edges=True)(**graph)

      return graph['node_features']

  def optimize_club(network: nn.Module, num_steps: int, seed) -> jnp.ndarray:
    karate_club = get_zacharys_karate_club()
    labels = get_ground_truth_assignments_for_zacharys_karate_club()
    network = Model()
    params = network.init(jax.random.PRNGKey(seed), get_zacharys_karate_club())

    @jax.jit
    def predict(params: Dict) -> jnp.ndarray:
      nodes = network.apply(params, karate_club)
      return jnp.argmax(nodes, axis=1)

    @jax.jit
    def prediction_loss(params: Dict) -> jnp.ndarray:
      nodes = network.apply(params, karate_club)
      log_prob = jax.nn.log_softmax(nodes)
      # The only two assignments we know a-priori are those of Mr. Hi (Node 0)
      # and John A (Node 33).
      return -(log_prob[0, 0] + log_prob[33, 1])

    opt_init, opt_update = optax.adam(1e-2)
    opt_state = opt_init(params)

    @jax.jit
    def update(params: Dict, opt_state: Any) -> Tuple[Dict, Any]:
      grads = jax.grad(prediction_loss)(params)
      updates, opt_state = opt_update(grads, opt_state)
      return optax.apply_updates(params, updates), opt_state

    @jax.jit
    def accuracy(params: Dict) -> jnp.ndarray:
      nodes = network.apply(params, karate_club)
      return jnp.mean(jnp.argmax(nodes, axis=1) == labels)

    for i in range(num_steps):
      params, opt_state = update(params, opt_state)

    return predict(params), accuracy(params).item()

  for i in range(5):
    model = Model()
    club, accuracy = optimize_club(model, num_steps=25, seed=i)
    print(accuracy)
    assert accuracy > 0.9


if __name__ == '__main__':
  # test()
  pytest.main([__file__])
