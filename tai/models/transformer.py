from typing import Callable, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import lax


def sinusoidal_init(
    max_len: int, max_scale: float = 1.0, min_scale: float = 10_000.0
) -> jax.Array:
    def init(key, shape, dtype=np.float32):
        del key, dtype
        dim = shape[-1]
        position = jnp.arange(0, max_len)[:, jnp.newaxis]
        scale_factor = (max_scale / min_scale) ** (2 * jnp.arange(0, dim, 2) / dim)
        div_term = jnp.divide(1, scale_factor)
        pe = jnp.empty((max_len, dim), dtype=np.float32)
        pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe.at[:, 1::2].set(jnp.cos(position * div_term))
        pe = pe[np.newaxis, :, :]
        return pe

    return init


@struct.dataclass
class TransformerConfig:
    in_vocab: int
    out_vocab: int
    emb_dim: int
    num_heads: int
    num_layers: int
    qkv_dim: int
    mlp_dim: int
    max_len: int
    dropout: float
    deterministic: bool = False
    decode: bool = False
    kernel_init: nn.initializers = nn.initializers.he_uniform()
    bias_init: nn.initializers = nn.initializers.ones
    pos_emb_init: Optional[Callable] = None

    @staticmethod
    def fromDict(config: Dict):
        return TransformerConfig(
            in_vocab=config["in_vocab"],
            out_vocab=config["out_vocab"],
            emb_dim=config["emb_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            qkv_dim=config["qkv_dim"],
            mlp_dim=config["mlp_dim"],
            max_len=config["max_len"],
            dropout=config["dropout"],
            deterministic=config["deterministic"],
            decode=config["decode"],
        )


class PosEmbedding(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs) -> jax.Array:
        length = inputs.shape[1]
        pos_emb_shape = (self.config.max_len, self.config.emb_dim)
        if self.config.pos_emb_init is None:
            pos_emb = sinusoidal_init(self.config.max_len)(None, pos_emb_shape, None)
        else:
            pos_emb = self.param("pos_emb", self.config.pos_emb_init, pos_emb_shape)
        pe = pos_emb[:length, :]

        if self.config.decode:
            is_intialized = self.has_variable("cache", "cache_index")
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
            )
            if is_intialized:
                i = cache_index.value
                cache_index.value = i + 1
                _, df = pos_emb.shape
                pe = lax.dynamic_slice(pos_emb, jnp.array((i, 0)), (1, df))
        return inputs + pe


class MLP(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs) -> jax.Array:
        x = nn.Dense(
            features=self.config.mlp_dim,
            use_bias=False,
            kernel_init=self.config.kernel_init,
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )
        x = nn.Dense(
            features=self.config.emb_dim,
            use_bias=False,
            kernel_init=self.config.kernel_init,
        )(x)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )
        return x


class EncoderLayer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, mask) -> jax.Array:
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.qkv_dim,
            kernel_init=self.config.kernel_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=self.config.dropout,
            deterministic=self.config.deterministic,
        )(inputs, inputs, inputs, mask=mask)
        x = nn.LayerNorm()(x + inputs)

        y = MLP(self.config)(x)
        y = nn.LayerNorm(bias_init=nn.initializers.ones)(y + x)
        return y


class DecoderLayer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, enc_out, enc_mask, input_mask) -> jax.Array:
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.qkv_dim,
            kernel_init=self.config.kernel_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=self.config.dropout,
            deterministic=self.config.deterministic,
            decode=self.config.decode,
        )(inputs, inputs, inputs, mask=input_mask)
        x = nn.LayerNorm(bias_init=nn.initializers.ones)(x + inputs)

        y = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.qkv_dim,
            kernel_init=self.config.kernel_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=self.config.dropout,
            deterministic=self.config.deterministic,
        )(x, enc_out, enc_out, mask=enc_mask)
        y = nn.LayerNorm(bias_init=nn.initializers.ones)(y + x)

        z = MLP(self.config)(y)
        z = nn.LayerNorm()(z + y)
        return z


class Encoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, mask) -> jax.Array:
        x = inputs.astype(np.int32)
        x = nn.Embed(
            num_embeddings=self.config.in_vocab,
            features=self.config.emb_dim,
            embedding_init=nn.initializers.he_normal(),
        )(inputs)
        x = PosEmbedding(self.config)(x)
        for i in range(self.config.num_layers):
            x = EncoderLayer(self.config, name=f"encoder_layer_{i}")(x, mask)
        return x


class Decoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, enc_out, enc_mask, input_mask) -> jax.Array:
        x = inputs.astype(np.int32)
        x = nn.Embed(
            num_embeddings=self.config.out_vocab,
            features=self.config.emb_dim,
            embedding_init=nn.initializers.he_normal(),
        )(inputs)
        x = PosEmbedding(self.config)(x)
        for i in range(self.config.num_layers):
            x = DecoderLayer(self.config, name=f"decoder_layer_{i}")(
                x, enc_out, enc_mask, input_mask
            )
        return x


class Transformer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, targets) -> jax.Array:
        if self.config.decode:
            input_mask = nn.make_attention_mask(jnp.ones_like(targets) > 0, inputs > 0)
            target_mask = None
        else:
            input_mask = nn.make_attention_mask(inputs > 0, inputs > 0)
            target_mask = nn.combine_masks(
                nn.make_attention_mask(targets > 0, targets > 0),
                nn.make_causal_mask(targets),
            )
        enc = Encoder(self.config)(inputs, input_mask)
        dec = Decoder(self.config)(targets, enc, input_mask, target_mask)
        logits = nn.Dense(
            features=self.config.out_vocab,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )(dec)
        return logits


def create_model(config: TransformerConfig, rngs: Dict) -> Transformer:
    """
    Creates and initializes a Transformer model from the given configuration.

    Args:
        config (TransformerConfig): Transformer configuration.
        rngs (Dict): Random number generators.

    Returns:
        tuple: Tuple of model and parameters.
    """
    model = Transformer(config)
    params = model.init(
        rngs,
        jnp.ones((config.max_len,), dtype=jnp.int32),
        jnp.ones((config.max_len,), dtype=jnp.int32),
    )
    return model, params
