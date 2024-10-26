
from functools import partial
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


class AttentionBlock(nn.Module):
  embed_dim: int
  hidden_dim: int
  num_heads: int
  pre_norm: bool = True
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               query: jax.Array,
               key: jax.Array,
               query_mask: jax.Array = None,
               key_mask: jax.Array = None):
    if query_mask is None:
      query_mask = jnp.ones(query.shape[:-1], dtype=bool)
    if key_mask is None:
      key_mask = jnp.ones(key.shape[:-1], dtype=bool)
    mask = nn.make_attention_mask(query_mask, key_mask)

    # Attention
    mha = nn.MultiHeadAttention(
        num_heads=self.num_heads,
        normalize_qk=self.pre_norm,
        dtype=self.dtype)
    x = query + mha(inputs_q=query, inputs_kv=key, mask=mask)

    # FFN
    ffn = nn.Sequential([
        nn.LayerNorm() if self.pre_norm else lambda x: x,
        nn.Dense(self.hidden_dim, dtype=self.dtype),
        nn.gelu,
        nn.Dense(self.embed_dim, dtype=self.dtype),
    ], name='ffn')
    x = x + ffn(x)

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
        query=S,
        key=x,
        query_mask=jnp.ones(self.num_seeds, dtype=bool),
        key_mask=valid
    )
    return x


class PerceiverIO(nn.Module):
  attention_base: nn.Module
  num_latents: int
  num_latent_steps: int

  @nn.compact
  def __call__(self,
               input_tokens: jax.Array,
               output_query: jax.Array,
               valid_input_token: Optional[jax.Array] = None,
               output_query_mask: Optional[jax.Array] = None,
               ) -> jax.Array:
    # Encode
    latents = self.param('latents', nn.initializers.truncated_normal(0.02),
                         (self.num_latents, self.embed_dim))
    latents = self.attention_base(
        query=latents,
        key=input_tokens,
        query_mask=None,
        key_mask=valid_input_token
    )

    # Self attention
    for i in range(self.num_latent_steps):
      latents = self.attention_base(
          query=latents,
          key=latents,
          query_mask=None,
          key_mask=None
      )

    # Decode
    x = self.attention_base(
        query=output_query,
        key=latents,
        query_mask=output_query_mask,
        key_mask=None
    )

    return x
