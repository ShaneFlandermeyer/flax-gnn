import flax.linen as nn
from flax_gnn.networks.linear import NormedLinear
from flax_gnn.networks.activation import mish


def mlp(
    embed_dim: int,
    num_layers: int,
    activation: nn.activation = mish,
    kernel_init: nn.initializers.Initializer = nn.initializers.truncated_normal(
        0.02
    )
) -> nn.Module:
  return nn.Sequential([
      NormedLinear(embed_dim, activation=activation, kernel_init=kernel_init) for _ in range(num_layers)
  ])
