
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from tdmpc2_jax.common.activations import mish
from tdmpc2_jax.networks.mlp import NormedLinear


class AttentionBlock(nn.Module):
  embed_dim: int
  hidden_dim: int
  num_heads: int
  use_layernorm: bool = True
  dtype: jnp.dtype = jnp.bfloat16

  @nn.compact
  def __call__(self,
               q: jax.Array,
               kv: jax.Array,
               q_mask: jax.Array = None,
               kv_mask: jax.Array = None):
    if q_mask is None:
      q_mask = jnp.ones(q.shape[:-1], dtype=bool)
    if kv_mask is None:
      kv_mask = jnp.ones(kv.shape[:-1], dtype=bool)

    # Attention
    mha = nn.MultiHeadAttention(num_heads=self.num_heads,
                                dtype=self.dtype,
                                use_bias=False)
    if self.use_layernorm:
      q_normed = nn.LayerNorm()(q, mask=q_mask[..., None])
      kv_normed = nn.LayerNorm()(kv, mask=kv_mask[..., None])
      q_normed = jnp.where(q_mask[..., None], q_normed, 0.0)
      kv_normed = jnp.where(kv_mask[..., None], kv_normed, 0.0)
    else:
      q_normed, kv_normed = q, kv
    x = q + mha(inputs_q=q_normed,
                inputs_kv=kv_normed,
                mask=nn.make_attention_mask(q_mask, kv_mask))

    # FFN
    if self.use_layernorm:
      x_normed = nn.LayerNorm()(x, mask=q_mask[..., None])
      x_normed = jnp.where(q_mask[..., None], x_normed, 0.0)
    else:
      x_normed = x
    ffn = nn.Sequential([
        nn.Dense(self.hidden_dim, dtype=self.dtype),
        nn.relu,
        nn.Dense(self.embed_dim, dtype=self.dtype),
    ])
    x = x + ffn(x_normed)

    return x


class PMA(nn.Module):
  attention_base: nn.Module
  num_seeds: int = 1

  @nn.compact
  def __call__(self, x: jax.Array, valid: jax.Array = None):
    batch_dims, embed_dim = x.shape[:-2], x.shape[-1]

    S = self.param('S', nn.initializers.xavier_uniform(),
                   (self.num_seeds, embed_dim))
    S = jnp.tile(S, [*batch_dims, 1, 1])

    x = self.attention_base(
        q=S, kv=x, q_mask=jnp.ones(self.num_seeds, dtype=bool), kv_mask=valid)
    return x
