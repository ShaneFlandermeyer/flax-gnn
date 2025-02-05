import time
from typing import Any, Dict, Tuple
import jraph
import flax.linen as nn
import jax
import optax
from flax_gnn.test.util import get_ground_truth_assignments_for_zacharys_karate_club, get_zacharys_karate_club
import jax.numpy as jnp
from flax_gnn.layers.gin import GIN
import pytest


def test():
  class Model(nn.Module):

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
      nodes = GIN(mlp=nn.Sequential([
          nn.Dense(8),
          nn.relu,
          nn.Dense(6)
      ]), epsilon=0.0)(
          nodes=graph.nodes,
          edges=graph.edges,
          globals_=graph.globals,
          senders=graph.senders,
          receivers=graph.receivers
      )
      nodes = GIN(mlp=nn.Sequential([
          nn.Dense(2),
          nn.relu,
          nn.Dense(2)
      ]), epsilon=None)(
          nodes=nodes,
          edges=graph.edges,
          globals_=graph.globals,
          senders=graph.senders,
          receivers=graph.receivers
      )

      return nodes

  def optimize_club(network: nn.Module, num_steps: int) -> jnp.ndarray:
    karate_club = get_zacharys_karate_club()
    labels = get_ground_truth_assignments_for_zacharys_karate_club()
    network = Model()
    params = network.init(jax.random.PRNGKey(0), get_zacharys_karate_club())

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

    start = time.time()
    for i in range(num_steps):
      params, opt_state = update(params, opt_state)
    print("Training time: ", time.time() - start)
    return predict(params), accuracy(params).item()

  model = Model()
  club, accuracy = optimize_club(model, num_steps=15)
  print(accuracy)
  assert accuracy > 0.9


if __name__ == '__main__':
  test()
  pytest.main([__file__])
