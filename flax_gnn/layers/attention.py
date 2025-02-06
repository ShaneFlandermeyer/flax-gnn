
from functools import partial
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax_gnn.layers.activations import mish


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
    if self.pre_norm:
      q_norm = nn.LayerNorm(dtype=self.dtype)(query)
      k_norm = nn.LayerNorm(dtype=self.dtype)(key)
    else:
      q_norm = query
      k_norm = key
    mha = nn.MultiHeadAttention(num_heads=self.num_heads, dtype=self.dtype)
    x = query + mha(inputs_q=q_norm, inputs_kv=k_norm, mask=mask)

    # FFN
    ffn = nn.Sequential([
        nn.LayerNorm() if self.pre_norm else lambda x: x,
        nn.Dense(self.hidden_dim, dtype=self.dtype),
        mish,
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

    S = self.param('S', nn.initializers.xavier_normal(),
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
  embed_dim: int
  attention_base: nn.Module
  num_latents: int
  num_latent_steps: int
  share_latent_weights: bool = False

  @nn.compact
  def __call__(self,
               input_tokens: jax.Array,
               output_query: jax.Array,
               input_mask: Optional[jax.Array] = None,
               output_mask: Optional[jax.Array] = None,
               ) -> jax.Array:
    # Encode
    latents = self.param('latents', nn.initializers.xavier_normal(),
                         (self.num_latents, self.embed_dim))
    latents = jnp.tile(latents, (*input_tokens.shape[:-2], 1, 1))
    latents = self.attention_base()(
        query=latents,
        key=input_tokens,
        query_mask=None,
        key_mask=input_mask
    )

    # Latent self attention
    for i in range(self.num_latent_steps):
      if self.share_latent_weights:
        latent_attention = self.attention_base() if i == 0 else latent_attention
      else:
        latent_attention = self.attention_base()

      latents = self.attention_base()(
          query=latents,
          key=latents,
          query_mask=None,
          key_mask=None
      )

    # Decode
    x = self.attention_base()(
        query=output_query,
        key=latents,
        query_mask=output_mask,
        key_mask=None
    )

    return x
